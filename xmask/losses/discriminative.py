# -*- coding: utf-8 -*-
"""
Discriminative (pull–push) loss on MaskHead embeddings + optional point–mask consistency.

Per predicted point on a GT foreground instance:
  radius = nearest-neighbor distance among points
  L_pos = [d - (τ - m)]_+    inside same-instance disc
  L_neg = [(τ + 4m) - d]_+   inside other-instance disc
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from xmask.pseudo import gaussian_blur_feat


class DiscriminativeLoss(nn.modules.loss._Loss):
    def __init__(
        self,
        l2_threshold=0.6,
        margin=0.1,
        point_mask_weight=1.0,
        invalid_disk_radius=16,
        min_radius=3.0,
        max_radius_factor=0.5,
    ):
        super().__init__()
        self.l2_threshold = l2_threshold
        self.margin = margin
        self.point_mask_weight = point_mask_weight
        self.invalid_disk_radius = int(invalid_disk_radius)
        self.min_radius = min_radius
        self.max_radius_factor = max_radius_factor

    def _invalid_from_pseudo(self, H, W, points, conf_mask, device):
        invalid = torch.zeros((H, W), dtype=torch.bool, device=device)
        if points is None or conf_mask is None or len(conf_mask) == 0:
            return invalid
        r = max(1, self.invalid_disk_radius)
        pp = points.to(device).float()
        for i in range(min(len(pp), len(conf_mask))):
            if bool(conf_mask[i]):
                continue
            py = int(pp[i, 0].round().clamp(0, H - 1).item())
            px = int(pp[i, 1].round().clamp(0, W - 1).item())
            y0, y1 = max(0, py - r), min(H, py + r + 1)
            x0, x1 = max(0, px - r), min(W, px + r + 1)
            yy, xx = torch.meshgrid(
                torch.arange(y0, y1, device=device),
                torch.arange(x0, x1, device=device),
                indexing="ij",
            )
            invalid[y0:y1, x0:x1] |= (yy - py) ** 2 + (xx - px) ** 2 <= r * r
        return invalid

    def _point_mask_loss(self, pts, mask, crop_mask, invalid):
        if pts.numel() == 0:
            pts_int = torch.zeros((0, 2), dtype=torch.long, device=mask.device)
        else:
            pts_int = pts.round().long()
            pts_int[:, 0].clamp_(0, mask.shape[0] - 1)
            pts_int[:, 1].clamp_(0, mask.shape[1] - 1)
        loss = torch.zeros((), device=mask.device)
        if pts_int.shape[0] > 0:
            py, px = pts_int[:, 0], pts_int[:, 1]
            loss = loss + (crop_mask[py, px] & (mask[py, px] == 0) & (~invalid[py, px])).float().sum()
        valid_region = crop_mask & (~invalid)
        if valid_region.any():
            any_pt = False if pts_int.shape[0] == 0 else valid_region[pts_int[:, 0], pts_int[:, 1]].any().item()
            if not any_pt:
                loss = loss + 1.0
        fg_ids = torch.unique(mask[crop_mask & (mask > 0)])
        denom = float(max(pts_int.shape[0], 1) + max(fg_ids.numel(), 1))
        return loss / denom

    def forward(
        self,
        embedding,
        points,
        mask_gt,
        crop_masks=None,
        pseudo_conf=None,
        pseudo_points=None,
    ):
        device = embedding.device
        B, D, H_emb, W_emb = embedding.shape
        total = (embedding[:, 0, 0, 0] * 0.0).sum()

        for b in range(B):
            pts = points[points[:, 0] == b][:, 1:]
            N = pts.shape[0]
            mask = mask_gt[b].squeeze(0).to(device)
            H_img, W_img = mask.shape
            crop = (
                crop_masks[b].squeeze().to(device).bool()
                if crop_masks is not None
                else torch.ones((H_img, W_img), dtype=torch.bool, device=device)
            )
            valid_mask = mask * crop.long()

            pm = pseudo_conf[b] if pseudo_conf is not None and len(pseudo_conf) > b else None
            pp = pseudo_points[b] if pseudo_points is not None and len(pseudo_points) > b else None
            invalid = self._invalid_from_pseudo(H_img, W_img, pp, pm, device)

            if self.point_mask_weight > 0:
                total = total + self.point_mask_weight * self._point_mask_loss(
                    pts, valid_mask, crop, invalid
                )
            if N == 0:
                continue

            if N == 1:
                radii = torch.full((1,), min(H_img, W_img) * self.max_radius_factor, device=device)
            else:
                dist_mat = torch.cdist(pts, pts)
                dist_mat.fill_diagonal_(float("inf"))
                radii = dist_mat.min(dim=1)[0]
            radii = radii.clamp(min=self.min_radius, max=min(H_img, W_img) * self.max_radius_factor)

            emb_up = (
                F.interpolate(embedding[b].unsqueeze(0), size=(H_img, W_img), mode="bilinear", align_corners=False).squeeze(0)
                if (H_emb, W_emb) != (H_img, W_img)
                else embedding[b]
            )
            emb_up = gaussian_blur_feat(emb_up, kernel_size=7, sigma=1.5)
            emb_flat = emb_up.permute(1, 2, 0)

            y_grid, x_grid = torch.meshgrid(
                torch.arange(H_img, device=device),
                torch.arange(W_img, device=device),
                indexing="ij",
            )
            grid_yx = torch.stack([y_grid, x_grid], dim=-1).float()
            dist = torch.norm(grid_yx.unsqueeze(0) - pts[:, None, None, :], dim=-1)
            circle = (dist <= radii[:, None, None]) & crop[None]

            pts_int = pts.round().long()
            pts_int[:, 0].clamp_(0, H_img - 1)
            pts_int[:, 1].clamp_(0, W_img - 1)
            inst_ids = mask[pts_int[:, 0], pts_int[:, 1]]
            valid_point = (inst_ids > 0) & crop[pts_int[:, 0], pts_int[:, 1]]
            if valid_point.sum() == 0:
                continue
            inst_ids = inst_ids[valid_point]
            circle = circle[valid_point]
            pts_int = pts_int[valid_point]
            point_mask = valid_mask.unsqueeze(0) == inst_ids[:, None, None]
            if invalid.any():
                circle = circle & (~invalid.unsqueeze(0))

            pos_region = circle & point_mask
            neg_region = circle & (~point_mask)

            grid_norm = (
                pts_int.view(1, -1, 1, 2).float().flip(-1)
                * 2.0
                / torch.tensor([W_img - 1, H_img - 1], device=device)
                - 1.0
            )
            center_feats = F.grid_sample(
                emb_up.unsqueeze(0), grid_norm, mode="bilinear", align_corners=True
            ).squeeze(0).squeeze(-1).T
            l2 = torch.norm(emb_flat.unsqueeze(0) - center_feats[:, None, None, :], dim=-1)
            tau, m = self.l2_threshold, self.margin
            pos_v = torch.clamp(l2 - (tau - m), min=0.0)
            neg_v = torch.clamp((tau + 4 * m) - l2, min=0.0)
            pos_cnt = pos_region.sum(dim=(1, 2))
            neg_cnt = neg_region.sum(dim=(1, 2))
            valid = pos_cnt > 0
            if valid.any():
                pos_loss = (pos_v * pos_region).sum(dim=(1, 2))[valid] / (pos_cnt[valid] + 1e-6)
                neg_loss = (neg_v * neg_region).sum(dim=(1, 2))[valid] / (neg_cnt[valid] + 1e-6)
                total = total + (pos_loss + neg_loss).mean()

        return total / max(1, B)
