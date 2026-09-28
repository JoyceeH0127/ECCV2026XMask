#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XMask unified trainer: merges original Stage1/2/3 into one schedule.

  Phase-1  [0, phase1_end)           : density via P2R (encoder+decoder)
  Phase-2  [phase1_end, phase2_end)  : MaskHead discriminative + online pseudo-masks
  Phase-3  [phase2_end, epochs)      : mask-constraint on density using pseudo-masks

Semi-supervised unlabeled branch opens after `semi_warmup` epochs into each phase
(Phase-3 always uses pseudo masks for constraint).
"""
from __future__ import annotations

import argparse
import datetime
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import StepLR
from timm.utils import AverageMeter

# allow `python train.py` from XMask root
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from xmask.config import load_config, dump_config
from xmask.logger import create_logger
from xmask.models import build_model
from xmask.data import build_loader
from xmask.losses import build_p2r, build_discriminative, build_mask_constraint
from xmask.pseudo import generate_pseudo_masks
from xmask.utils import (
    set_seed,
    get_grad_norm,
    save_checkpoint,
    load_checkpoint,
    restore_training_state,
    auto_resume,
    ema_update,
    set_requires_grad,
)


def parse_args():
    p = argparse.ArgumentParser("XMask unified training")
    p.add_argument("--config", required=True, help="path to YAML config")
    p.add_argument("--opts", default=None, nargs="+", help="override KEY VALUE pairs")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--resume", default="", help="checkpoint path")
    return p.parse_args()


def current_phase(epoch, cfg):
    p1, p2 = cfg["train"]["phase1_end"], cfg["train"]["phase2_end"]
    if epoch < p1:
        return 1
    if epoch < p2:
        return 2
    return 3


def phase_local_epoch(epoch, cfg):
    p1, p2 = cfg["train"]["phase1_end"], cfg["train"]["phase2_end"]
    if epoch < p1:
        return epoch
    if epoch < p2:
        return epoch - p1
    return epoch - p2


def configure_trainable(student, phase, logger):
    """Freeze / unfreeze modules per phase (mirrors original 3-stage practice)."""
    set_requires_grad(student, False)
    if phase == 1:
        set_requires_grad(student.decoders, True)
        set_requires_grad(student.fuse_layer, True)
        set_requires_grad(student.encoders, True)
        logger.info("[phase1] train encoder + fuse + decoder (P2R)")
    elif phase == 2:
        set_requires_grad(student.maskhead, True)
        # late phase2 also fine-tunes fuse/decoder (as in original stage2)
        set_requires_grad(student.decoders, True)
        set_requires_grad(student.fuse_layer, True)
        logger.info("[phase2] train maskhead (+ fuse/decoder); encoder frozen")
    else:
        set_requires_grad(student.decoders, True)
        set_requires_grad(student.fuse_layer, True)
        logger.info("[phase3] train fuse + decoder under mask constraint; encoder+maskhead frozen")


def build_optimizer(student, cfg, phase):
    base_lr = cfg["train"]["base_lr"]
    bb_lr = cfg["train"]["backbone_lr"]
    wd = cfg["train"]["weight_decay"]
    phase_lr = cfg["train"].get("phase_lr", {}) or {}
    if phase == 1:
        params = [
            {"params": [p for n, p in student.named_parameters() if "encoders" not in n and p.requires_grad]},
            {
                "params": [p for n, p in student.named_parameters() if "encoders" in n and p.requires_grad],
                "lr": bb_lr,
            },
        ]
        return optim.Adam(params, lr=base_lr, weight_decay=wd)
    lr = float(phase_lr.get(str(phase), cfg["train"].get("phase3_lr", base_lr) if phase == 3 else base_lr))
    params = [p for p in student.parameters() if p.requires_grad]
    return optim.Adam(params, lr=lr, weight_decay=wd)


def build_phase_training_state(student, cfg, phase, logger):
    configure_trainable(student, phase, logger)
    optimizer = build_optimizer(student, cfg, phase)
    scheduler = StepLR(
        optimizer,
        step_size=cfg["train"]["lr_decay_epochs"],
        gamma=cfg["train"]["lr_decay_rate"],
    )
    if phase == 3 and "phase3_lr" in cfg["train"]:
        for group in optimizer.param_groups:
            group["lr"] = cfg["train"]["phase3_lr"]
    return optimizer, scheduler


def densify_points(den, img_hw):
    """(B,1,h,w) logits → (N,3) points at image resolution [b,y,x]."""
    down = img_hw[0] // den.size(-2)
    pts_low = torch.nonzero((den > 0).squeeze(1), as_tuple=False).float()
    if pts_low.numel() == 0:
        return pts_low
    pts = pts_low.clone()
    pts[:, 1:] = pts_low[:, 1:] * down + (down - 1) / 2
    return pts


@torch.no_grad()
def validate(loader, model):
    model.eval()
    mae_m, mse_m = AverageMeter(), AverageMeter()
    for images, dotseq, _, _masks in loader:
        images = images.cuda(non_blocking=True)
        cnt = torch.tensor([d.size(0) for d in dotseq], dtype=torch.float32, device="cuda")
        den, _, _ = model(images)
        outnum = (den > 0).sum(dim=(1, 2, 3)).float()
        diff = torch.abs(outnum - cnt)
        mae_m.update(diff.mean().item(), images.size(0))
        mse_m.update((diff ** 2).mean().item(), images.size(0))
    return mae_m.avg, math.sqrt(mse_m.avg)


def train_one_epoch(
    epoch,
    loader,
    student,
    teacher,
    optimizer,
    criteria,
    cfg,
    logger,
):
    phase = current_phase(epoch, cfg)
    local_ep = phase_local_epoch(epoch, cfg)
    warmup = cfg["train"]["semi_warmup"]
    enable_semi = local_ep >= warmup or phase == 3

    student.train()
    if phase >= 2:
        student.encoders.eval()
    if phase == 3:
        student.maskhead.eval()
    teacher.eval()

    p2r_crit, disc_crit, mc_crit = criteria
    loss_m, norm_m, bt_m = AverageMeter(), AverageMeter(), AverageMeter()
    num_steps = len(loader)
    end = time.time()
    start = end

    for idx, batch in enumerate(loader):
        limg, lseq, _lid, lmask, uimg, _uorig, umasks, _uid = batch
        limg = limg.cuda(non_blocking=True)
        lseq = [d.cuda(non_blocking=True) for d in lseq]
        lmask = lmask.cuda(non_blocking=True)
        down = limg.size(-1) // 64  # will recompute after forward

        try:
            lden, ldenmap, lemb = student(limg)
            down = limg.size(-1) // lden.size(-1)
            loss = None

            # -------- Phase 1: P2R on labeled --------
            if phase == 1:
                loss = p2r_crit(lden, lseq, down)
                if enable_semi:
                    wa = uimg[:, :3].cuda(non_blocking=True)
                    sa = uimg[:, 3:].cuda(non_blocking=True)
                    cut = umasks.cuda(non_blocking=True)
                    cut_den = F.avg_pool2d(cut, down, down) >= 0.5
                    with torch.inference_mode():
                        tden, _, _ = teacher(wa)
                        tseq = []
                        for d in tden:
                            pts = torch.nonzero((d >= 0).squeeze(), as_tuple=False).float()
                            tseq.append(pts * down + (down - 1) / 2)
                    sden, _, _ = student(sa * cut)
                    weight = min(max((local_ep - warmup) * 0.01, 0), 2)
                    semi = p2r_crit(sden, tseq, down, crop_den_masks=cut_den)
                    loss = (loss + weight * semi) / (1 + weight)

            # -------- Phase 2: discriminative + pseudo --------
            elif phase == 2:
                pts = densify_points(lden.detach(), limg.shape[-2:])
                lemb_h = F.interpolate(lemb, size=limg.shape[-2:], mode="bilinear", align_corners=False)
                loss = disc_crit(lemb_h, pts, lmask)
                if enable_semi:
                    wa = uimg[:, :3].cuda(non_blocking=True)
                    sa = uimg[:, 3:].cuda(non_blocking=True)
                    cut = umasks.cuda(non_blocking=True)
                    with torch.inference_mode():
                        tden, _, temb = teacher(wa)
                        masks, pp, conf, _ = generate_pseudo_masks(
                            tden,
                            temb,
                            sa.shape[-2:],
                            energy_threshold=cfg["pseudo"]["energy_threshold"],
                            lambda_geo=cfg["pseudo"]["lambda_geo"],
                            p_low=cfg["pseudo"]["p_low"],
                            p_high=cfg["pseudo"]["p_high"],
                            use_density_gate=False,
                        )
                        mask_gt = torch.stack(masks, 0)
                    sden, _, semb = student(sa * cut)
                    semb_h = F.interpolate(semb, size=sa.shape[-2:], mode="bilinear", align_corners=False)
                    spts = densify_points(sden, sa.shape[-2:])
                    Hh, Wh = sa.shape[-2:]
                    crop = F.interpolate(cut.float(), size=(Hh, Wh), mode="nearest") >= 0.5
                    weight = min(max((local_ep - warmup) * 0.01, 0), 2)
                    semi = disc_crit(
                        semb_h, spts, mask_gt, crop_masks=crop, pseudo_conf=conf, pseudo_points=pp
                    )
                    loss = (loss + weight * semi) / (1 + weight)

            # -------- Phase 3: mask constraint --------
            else:
                # supervised: P2R + optional MC on GT masks
                loss_sup, _ = mc_crit(lden, ldenmap, lseq, lmask, down)
                wa = uimg[:, :3].cuda(non_blocking=True)
                sa = uimg[:, 3:].cuda(non_blocking=True)
                cut = umasks.cuda(non_blocking=True)
                cut_den = F.avg_pool2d(cut, down, down) >= 0.5
                with torch.inference_mode():
                    tden, _, temb = teacher(wa)
                    masks, pp, conf, tseq = generate_pseudo_masks(
                        tden,
                        temb,
                        sa.shape[-2:],
                        energy_threshold=cfg["pseudo"]["energy_threshold"],
                        lambda_geo=cfg["pseudo"]["lambda_geo"],
                        p_low=cfg["pseudo"]["p_low"],
                        p_high=cfg["pseudo"]["p_high"],
                        use_density_gate=cfg["pseudo"].get("use_density_gate", True),
                    )
                    mask_gt = torch.stack(masks, 0)
                sden, sdenmap, _ = student(sa * cut)
                weight = min(max(local_ep * 0.01, 0), 2) if local_ep < warmup else min(max((local_ep - 0) * 0.01, 0), 2)
                # phase3 always semi from the start (like original STAGE_2=0)
                weight = min(max(local_ep * 0.01 + 0.1, 0.1), 2)
                semi, _ = mc_crit(
                    sden,
                    sdenmap,
                    tseq,
                    mask_gt,
                    down,
                    crop_den_masks=cut_den,
                    pseudo_conf=conf,
                    pseudo_points=pp,
                )
                loss = (loss_sup + weight * semi) / (1 + weight)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg["train"]["clip_grad"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in student.parameters() if p.requires_grad],
                    cfg["train"]["clip_grad"],
                )
            gn = get_grad_norm(student.parameters())
            optimizer.step()
            ema_update(teacher, student, cfg["train"]["ema_momentum"])

            loss_m.update(loss.item(), limg.size(0))
            norm_m.update(gn)
            bt_m.update(time.time() - end)
            end = time.time()

            if idx % cfg["print_freq"] == 0:
                lr = optimizer.param_groups[0]["lr"]
                mem = torch.cuda.max_memory_allocated() / 1024 ** 2
                eta = bt_m.avg * (num_steps - idx)
                logger.info(
                    f"Train p{phase} [{epoch}/{cfg['train']['epochs']}][{idx}/{num_steps}] "
                    f"eta {datetime.timedelta(seconds=int(eta))} lr {lr:.2e} "
                    f"loss {loss_m.val:.3f} ({loss_m.avg:.3f}) "
                    f"gn {norm_m.avg:.2f} mem {mem:.0f}MB"
                )
        except RuntimeError as err:
            if "out of memory" in str(err).lower():
                logger.warning(f"OOM epoch {epoch} batch {idx}, skip")
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue
            raise

    logger.info(
        f"EPOCH {epoch} phase={phase} takes {datetime.timedelta(seconds=int(time.time() - start))}"
    )


def main():
    args = parse_args()
    cfg = load_config(args.config, args.opts)
    if args.resume:
        cfg["model"]["resume"] = os.path.abspath(args.resume)

    set_seed(cfg["seed"])
    logger = create_logger(cfg["output"], name=cfg["tag"])
    dump_config(cfg, os.path.join(cfg["output"], "config.yaml"))
    logger.info(f"config dumped to {cfg['output']}/config.yaml")
    logger.info(
        f"schedule: phase1=[0,{cfg['train']['phase1_end']}) "
        f"phase2=[{cfg['train']['phase1_end']},{cfg['train']['phase2_end']}) "
        f"phase3=[{cfg['train']['phase2_end']},{cfg['train']['epochs']})"
    )

    train_loader = build_loader(cfg["data"], mode="train")
    val_loader = build_loader(cfg["data"], mode="test")

    student, teacher = build_model(cfg["model"]["name"])
    student.cuda()
    teacher.cuda()
    for p in teacher.parameters():
        p.requires_grad = False

    p2r_crit = build_p2r().cuda()
    disc_crit = build_discriminative(cfg["loss"]["disc"]).cuda()
    mc_crit = build_mask_constraint(cfg["loss"]["mask_constraint"]).cuda()
    criteria = (p2r_crit, disc_crit, mc_crit)

    best = {"mae": 1e9, "mse": 1e9}
    start_epoch = 0
    checkpoint = None
    resume = cfg["model"].get("resume") or ""
    if cfg["train"].get("auto_resume") and not resume:
        latest = auto_resume(cfg["output"])
        if latest:
            resume = latest
            logger.info(f"auto-resume {resume}")
    if resume:
        checkpoint, teacher_status, student_status = load_checkpoint(
            resume, [teacher, student]
        )
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best = checkpoint.get("best", best)
        logger.info(f"resumed from epoch {start_epoch - 1}, best MAE={best['mae']:.2f}")
        logger.info(f"[load teacher]: {teacher_status}")
        logger.info(f"[load student]: {student_status}")

    if args.eval:
        mae, mse = validate(val_loader, student)
        logger.info(f"Eval MAE {mae:.3f} MSE {mse:.3f}")
        return

    current_p = None
    optimizer = None
    scheduler = None

    if checkpoint is not None and start_epoch < cfg["train"]["epochs"]:
        current_p = current_phase(start_epoch, cfg)
        optimizer, scheduler = build_phase_training_state(
            student, cfg, current_p, logger
        )
        saved_epoch = int(checkpoint.get("epoch", -1))
        saved_phase = checkpoint.get("phase")
        if saved_phase is None and saved_epoch >= 0:
            saved_phase = current_phase(saved_epoch, cfg)
        same_phase = saved_phase == current_p
        restored = restore_training_state(
            checkpoint,
            optimizer=optimizer if same_phase else None,
            scheduler=scheduler if same_phase else None,
        )
        if same_phase:
            logger.info(
                "resume state: optimizer=%s scheduler=%s rng=%s",
                restored["optimizer"],
                restored["scheduler"],
                restored["rng"],
            )
        else:
            logger.info(
                "resume crosses phase boundary %s -> %s; initialized the new "
                "phase optimizer and scheduler (rng=%s)",
                saved_phase,
                current_p,
                restored["rng"],
            )

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        phase = current_phase(epoch, cfg)
        if phase != current_p:
            optimizer, scheduler = build_phase_training_state(
                student, cfg, phase, logger
            )
            current_p = phase

        train_one_epoch(epoch, train_loader, student, teacher, optimizer, criteria, cfg, logger)
        scheduler.step()

        mae, mse = validate(val_loader, student)
        logger.info(f"* MAE {mae:.3f} MSE {mse:.3f}")
        improved = mae < best["mae"]
        if improved:
            best = {"mae": mae, "mse": mse}
            save_checkpoint(
                os.path.join(cfg["output"], "ckpt_best.pth"),
                epoch,
                [teacher, student],
                best,
                optimizer,
                scheduler,
                phase,
            )
            logger.info(f"best updated → MAE {mae:.3f}")
        if (epoch + 1) % cfg["save_freq"] == 0:
            save_checkpoint(
                os.path.join(cfg["output"], f"ckpt_epoch_{epoch}.pth"),
                epoch,
                [teacher, student],
                best,
                optimizer,
                scheduler,
                phase,
            )

    logger.info(f"Done. Best MAE {best['mae']:.3f} MSE {best['mse']:.3f}")


if __name__ == "__main__":
    main()
