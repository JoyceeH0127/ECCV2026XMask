# -*- coding: utf-8 -*-
"""Shared counting and instance-segmentation evaluation utilities."""
from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from xmask.config import load_config
from xmask.data import build_loader
from xmask.models import build_single_model
from xmask.pseudo import generate_pseudo_masks
from xmask.utils import set_seed


SEG_ENERGY_THRESHOLD = 1.1
SEG_GEOMETRY_WEIGHT = 0.8
SEG_GAUSSIAN_KERNEL_SIZE = 7
SEG_GAUSSIAN_SIGMA = 3.0


def add_common_arguments(parser):
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument("--checkpoint", required=True, help="path to a .pth checkpoint")
    parser.add_argument(
        "--weights",
        choices=("student", "teacher"),
        default="student",
        help="checkpoint model to evaluate (default: student)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="torch device, for example cuda, cuda:0, or cpu (default: auto)",
    )
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--opts", nargs="+", default=None, help="override KEY VALUE pairs")
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="allow missing or unexpected checkpoint parameters",
    )


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return device


def load_weights(model, checkpoint_path, weights, strict):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or weights not in checkpoint:
        available = sorted(checkpoint) if isinstance(checkpoint, dict) else []
        raise KeyError(
            f"checkpoint has no '{weights}' weights; available keys: {available}"
        )
    incompatible = model.load_state_dict(checkpoint[weights], strict=strict)
    if not strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        print(f"Missing keys: {incompatible.missing_keys}")
        print(f"Unexpected keys: {incompatible.unexpected_keys}")
    return checkpoint


def prepare_evaluation(args):
    checkpoint_path = os.path.abspath(args.checkpoint)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    cfg = load_config(args.config, args.opts)
    set_seed(cfg["seed"])
    device = resolve_device(args.device)
    loader = build_loader(cfg["data"], mode="test")
    model = build_single_model(cfg["model"]["name"], pretrained=False)
    checkpoint = load_weights(
        model, checkpoint_path, args.weights, strict=not args.non_strict
    )
    model.to(device)

    print(f"Config: {os.path.abspath(args.config)}")
    print(f"Checkpoint: {checkpoint_path} ({args.weights})")
    print(f"Device: {device}")
    if "epoch" in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
    return loader, model, device


class CountingMetrics:
    def __init__(self):
        self.absolute_error = 0.0
        self.squared_error = 0.0
        self.num_samples = 0

    def update(self, predicted, targets):
        for pred_count, target_count in zip(predicted, targets):
            error = float(pred_count - target_count)
            self.absolute_error += abs(error)
            self.squared_error += error * error
            self.num_samples += 1

    def compute(self):
        if self.num_samples == 0:
            raise RuntimeError("the test split is empty")
        return {
            "images": self.num_samples,
            "mae": self.absolute_error / self.num_samples,
            "rmse": math.sqrt(self.squared_error / self.num_samples),
        }


def instance_iou_matrix(prediction, target):
    """Compute pairwise instance IoU directly from two integer label maps."""
    prediction = prediction.long().reshape(-1)
    target = target.long().reshape(-1)
    if prediction.numel() != target.numel():
        raise ValueError("prediction and target masks must have the same number of pixels")

    pred_ids = torch.unique(prediction)
    pred_ids = pred_ids[pred_ids > 0]
    target_ids = torch.unique(target)
    target_ids = target_ids[target_ids > 0]
    num_pred, num_target = pred_ids.numel(), target_ids.numel()
    if num_pred == 0 or num_target == 0:
        return torch.zeros(
            (num_pred, num_target), dtype=torch.float32, device=prediction.device
        )

    pred_index = torch.zeros_like(prediction)
    pred_positive = prediction > 0
    pred_index[pred_positive] = (
        torch.searchsorted(pred_ids, prediction[pred_positive]) + 1
    )
    target_index = torch.zeros_like(target)
    target_positive = target > 0
    target_index[target_positive] = (
        torch.searchsorted(target_ids, target[target_positive]) + 1
    )

    joint_index = pred_index * (num_target + 1) + target_index
    intersections = torch.bincount(
        joint_index,
        minlength=(num_pred + 1) * (num_target + 1),
    ).reshape(num_pred + 1, num_target + 1)[1:, 1:].float()
    pred_area = torch.bincount(pred_index, minlength=num_pred + 1)[1:].float()
    target_area = torch.bincount(target_index, minlength=num_target + 1)[1:].float()
    unions = pred_area[:, None] + target_area[None, :] - intersections
    return torch.where(unions > 0, intersections / unions, torch.zeros_like(unions))


class SegmentationMetrics:
    """Mean IoU C from P2RLoss/validation.py using Hungarian matching."""

    def __init__(self):
        self.matched_iou_sum = 0.0
        self.total_gt_instances = 0
        self.total_pred_instances = 0
        self.num_images = 0

    def update(self, prediction, target):
        if prediction.shape != target.shape:
            prediction = F.interpolate(
                prediction[None, None].float(),
                size=target.shape[-2:],
                mode="nearest",
            )[0, 0].long()
        iou = instance_iou_matrix(prediction, target)
        num_pred, num_target = iou.shape
        if num_pred > 0 and num_target > 0:
            row_indices, column_indices = linear_sum_assignment(
                1.0 - iou.detach().cpu().numpy()
            )
            self.matched_iou_sum += float(
                iou[row_indices, column_indices].sum().item()
            )
        self.total_pred_instances += num_pred
        self.total_gt_instances += num_target
        self.num_images += 1

    def compute(self):
        mean_iou_c = (
            self.matched_iou_sum / self.total_gt_instances
            if self.total_gt_instances > 0
            else 0.0
        )
        return {
            "images": self.num_images,
            "pred_instances": self.total_pred_instances,
            "gt_instances": self.total_gt_instances,
            "mean_iou_c": mean_iou_c,
        }


def _progress(step, total, image_id, count_metrics, seg_metrics):
    fields = [f"[{step}/{total}]", f"id={image_id}"]
    if count_metrics is not None:
        result = count_metrics.compute()
        fields.extend((f"MAE={result['mae']:.3f}", f"RMSE={result['rmse']:.3f}"))
    if seg_metrics is not None:
        result = seg_metrics.compute()
        fields.append(f"MeanIoU_C={result['mean_iou_c']:.4f}")
    print(" ".join(fields))


@torch.inference_mode()
def evaluate(
    loader,
    model,
    device,
    tasks=("count", "seg"),
    count_threshold=0.0,
    print_freq=50,
):
    unknown = set(tasks) - {"count", "seg"}
    if unknown:
        raise ValueError(f"unknown evaluation tasks: {sorted(unknown)}")
    model.eval()
    count_metrics = CountingMetrics() if "count" in tasks else None
    seg_metrics = SegmentationMetrics() if "seg" in tasks else None

    for step, (images, dotseq, image_ids, gt_masks) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=device.type == "cuda")
        density, _, embedding = model(images)

        if count_metrics is not None:
            predicted = (density > count_threshold).sum(dim=(1, 2, 3)).cpu().tolist()
            targets = [points.size(0) for points in dotseq]
            count_metrics.update(predicted, targets)

        if seg_metrics is not None:
            predicted_masks, _, _, _ = generate_pseudo_masks(
                density,
                embedding,
                images.shape[-2:],
                energy_threshold=SEG_ENERGY_THRESHOLD,
                lambda_geo=SEG_GEOMETRY_WEIGHT,
                use_density_gate=False,
                gaussian_kernel_size=SEG_GAUSSIAN_KERNEL_SIZE,
                gaussian_sigma=SEG_GAUSSIAN_SIGMA,
            )
            gt_masks = gt_masks.to(device, non_blocking=device.type == "cuda")
            for prediction, target in zip(predicted_masks, gt_masks):
                seg_metrics.update(prediction.squeeze(0), target.squeeze(0))

        if print_freq > 0 and (step % print_freq == 0 or step == len(loader)):
            _progress(
                step,
                len(loader),
                image_ids[-1],
                count_metrics,
                seg_metrics,
            )

    results = {}
    if count_metrics is not None:
        results["count"] = count_metrics.compute()
    if seg_metrics is not None:
        results["seg"] = seg_metrics.compute()
    return results


def print_count_result(result):
    print(
        f"Count evaluation: images={result['images']} "
        f"MAE={result['mae']:.3f} RMSE={result['rmse']:.3f}"
    )


def print_seg_result(result):
    print(
        f"Seg evaluation: images={result['images']} "
        f"pred_instances={result['pred_instances']} "
        f"gt_instances={result['gt_instances']} "
        f"MeanIoU_C={result['mean_iou_c']:.4f}"
    )
