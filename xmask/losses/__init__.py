# -*- coding: utf-8 -*-
from .p2r import P2RLoss
from .discriminative import DiscriminativeLoss
from .mask_constraint import MaskConstraintLoss


def build_p2r(cfg=None):
    return P2RLoss(factor=1)


def build_discriminative(cfg):
    return DiscriminativeLoss(
        l2_threshold=cfg.get("l2_threshold", 0.6),
        margin=cfg.get("margin", 0.1),
        point_mask_weight=cfg.get("point_mask_weight", 1.0),
        invalid_disk_radius=cfg.get("invalid_disk_radius", 16),
    )


def build_mask_constraint(cfg):
    mode = cfg.get("loss_mode", "p2r_mc")  # p2r_mc | points | mc_only
    with_p2r = mode in ("p2r_mc", "points", "P2R")
    if mode == "mc_only":
        with_p2r = False
    return MaskConstraintLoss(
        bg_weight=cfg.get("bg_weight", 0.0),
        fg_weight=cfg.get("fg_weight", 0.005),
        one_point_weight=cfg.get("one_point_weight", 0.0),
        with_p2r=with_p2r,
    )
