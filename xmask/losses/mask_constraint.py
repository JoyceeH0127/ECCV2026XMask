# -*- coding: utf-8 -*-
"""
Mask Constraint Loss (Phase-3) on density logits given pseudo instance masks.

Three terms (weights from config):
  bg:      mean ReLU(logits) on background  → suppress FP
  fg:      |sum ReLU(logits) - 1| per valid instance
  one_pt:  |#{logits>0} - 1| per valid instance

Valid instances = those hit by high-confidence pseudo points (conf_mask=True).
Optionally also includes P2R point-matching (combined mode).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .p2r import L2DIS

eps = 1e-10


class MaskConstraintLoss(nn.modules.loss._Loss):
    def __init__(
        self,
        bg_weight=0.0,
        fg_weight=0.005,
        one_point_weight=0.0,
        with_p2r=True,
        factor=1,
    ):
        super().__init__()
        self.bg_weight = bg_weight
        self.fg_weight = fg_weight
        self.one_point_weight = one_point_weight
        self.with_p2r = with_p2r
        self.cost = L2DIS(1)
        self.min_radis = 8
        self.max_radis = 96
        self.cost_class = 1
        self.cost_point = 8

    def _p2r_term(self, den_ori, seq, down, crop_den_mask=None):
        den = den_ori.permute(1, 2, 0)
        H, W = den.shape[:2]
        if seq.size(0) < 1:
            return F.binary_cross_entropy_with_logits(
                den, torch.zeros_like(den), weight=torch.ones_like(den) * 0.5
            )
        A_coord = (
            torch.stack(
                torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij"), dim=-1
            ).view(1, -1, 2)
            * down
            + (down - 1) / 2
        )
        A = den.view(1, -1, 1)
        A_coord = A_coord.to(seq).float()
        B_coord = seq[None, :, :2].float()
        with torch.no_grad():
            C = self.cost(A_coord, B_coord)
            minC, mcidx = C.min(dim=-1, keepdim=True)
            M = torch.zeros_like(C).scatter_(-1, mcidx, 1.0) * (C < self.max_radis)
            maxC = torch.clip((minC.view_as(A) * M).amax(dim=1, keepdim=True), self.min_radis, self.max_radis)
            C = (C / maxC) * self.cost_point - A * self.cost_class
            vid = (M.sum(dim=1) > 0).view(-1)
            C, M = C[..., vid], M[..., vid]
            C2 = M * C + (1 - M) * (C.max() + 1)
            _, mcidx2 = C2.min(dim=1, keepdim=True)
            T = (torch.zeros_like(C2).scatter_(1, mcidx2, 1.0).sum(dim=-1).view_as(A) > 0.5).to(A)
            W = T + 1
        if crop_den_mask is not None:
            W = W * crop_den_mask.view_as(W)
        return F.binary_cross_entropy_with_logits(A, T, weight=W)

    def _constraint_terms(self, den_ori, mask_gt, conf_mask=None, pseudo_points=None, crop_den_mask=None):
        device = den_ori.device
        mask_gt = mask_gt.squeeze(0).to(device)
        bg_mask = (mask_gt == 0).float()
        if crop_den_mask is not None:
            cdm = F.interpolate(
                crop_den_mask.float().unsqueeze(0), size=mask_gt.shape[-2:], mode="nearest"
            ).squeeze()
            bg_mask = bg_mask * cdm

        den_logits = den_ori.squeeze(0)
        up = F.interpolate(
            den_logits.unsqueeze(0).unsqueeze(0),
            size=mask_gt.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        scale = (torch.relu(den_logits).sum() / (torch.relu(up).sum() + eps)).detach()
        up_scaled = up * scale

        bg_pen = torch.tensor(0.0, device=device)
        bg_bool = bg_mask > 0.5
        if bg_bool.any():
            bg_pen = torch.relu(up[bg_bool]).sum() / max(1, bg_bool.sum())

        mg2 = mask_gt.long()
        instance_ids = torch.unique(mg2)
        instance_ids = instance_ids[instance_ids > 0]
        mask_sum = torch.tensor(0.0, device=device)
        one_pt = torch.tensor(0.0, device=device)

        if instance_ids.numel() == 0:
            return bg_pen, mask_sum, one_pt

        valid_ids = instance_ids
        if conf_mask is not None and pseudo_points is not None and pseudo_points.numel() > 0:
            pts = pseudo_points.long().to(mg2.device)
            n = min(pts.shape[0], conf_mask.numel())
            if n > 0:
                keep = conf_mask[:n].nonzero(as_tuple=False).squeeze(1)
                if keep.numel() > 0:
                    rows = pts[keep, 0].clamp(0, mg2.shape[0] - 1)
                    cols = pts[keep, 1].clamp(0, mg2.shape[1] - 1)
                    sampled = mg2[rows, cols]
                    sampled = sampled[sampled > 0]
                    valid_ids = (
                        torch.unique(sampled)
                        if sampled.numel() > 0
                        else torch.empty(0, device=device, dtype=torch.long)
                    )
                else:
                    valid_ids = torch.empty(0, device=device, dtype=torch.long)

        if valid_ids.numel() > 0:
            n = valid_ids.numel()
            one_hot = (mg2.unsqueeze(0) == valid_ids.view(n, 1, 1)).view(n, -1).float()
            inst_sum = (one_hot @ torch.relu(up_scaled.view(-1, 1))).squeeze(1)
            mask_sum = (torch.relu(1.0 - inst_sum) + torch.relu(inst_sum - 1.0)).sum() / max(1, n)
            pos_count = (one_hot @ (up.view(-1) > 0).float().view(-1, 1)).squeeze(1)
            one_pt = (torch.relu(1.0 - pos_count) + torch.relu(pos_count - 1.0)).sum() / max(1, n)
        return bg_pen, mask_sum, one_pt

    def forward(
        self,
        dens,
        denmap,
        seqs,
        masks_gt,
        down,
        crop_den_masks=None,
        pseudo_conf=None,
        pseudo_points=None,
    ):
        bs = len(seqs)
        device = dens[0].device
        p2r = torch.tensor(0.0, device=device)
        bg_all = torch.tensor(0.0, device=device)
        fg_all = torch.tensor(0.0, device=device)
        op_all = torch.tensor(0.0, device=device)

        for i in range(bs):
            if self.with_p2r:
                cdm = crop_den_masks[i] if crop_den_masks is not None else None
                p2r = p2r + self._p2r_term(dens[i], seqs[i], down, cdm)
            if masks_gt is not None:
                conf = pseudo_conf[i] if pseudo_conf is not None else None
                pts = pseudo_points[i] if pseudo_points is not None else None
                cdm = crop_den_masks[i] if crop_den_masks is not None else None
                bg, fg, op = self._constraint_terms(dens[i], masks_gt[i], conf, pts, cdm)
                bg_all = bg_all + bg
                fg_all = fg_all + fg
                op_all = op_all + op

        mc = (
            self.bg_weight * bg_all + self.fg_weight * fg_all + self.one_point_weight * op_all
        ) / bs
        if self.with_p2r:
            return p2r / bs + mc, mc
        return mc, mc
