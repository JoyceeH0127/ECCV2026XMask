# -*- coding: utf-8 -*-
"""Online pseudo instance-mask generation from density peaks + embeddings."""
import math
import torch
import torch.nn.functional as F


def gaussian_blur_feat(feat, kernel_size=7, sigma=1.5):
    """Separable Gaussian blur on feature map (D,H,W) or (B,D,H,W)."""
    if feat.dim() == 3:
        feat = feat.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    B, D, H, W = feat.shape
    device, dtype = feat.device, feat.dtype
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    coords = torch.arange(k, device=device, dtype=dtype) - k // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_x = g.view(1, 1, 1, k).repeat(D, 1, 1, 1)
    kernel_y = g.view(1, 1, k, 1).repeat(D, 1, 1, 1)
    pad = k // 2
    feat = F.conv2d(feat, kernel_x, padding=(0, pad), groups=D)
    feat = F.conv2d(feat, kernel_y, padding=(pad, 0), groups=D)
    return feat.squeeze(0) if squeeze else feat


@torch.no_grad()
def build_instance_map(
    feature_embedding,
    points,
    energy_threshold=0.8,
    lambda_geo=1.0,
    tden=None,
    down_rate=4,
    use_density_gate=False,
    max_points=50,
    gaussian_kernel_size=7,
    gaussian_sigma=1.5,
):
    """
    Build instance id map from embedding energy discs around pseudo points.

    Args:
        feature_embedding: (D, H, W) L2-normalized embedding at image resolution
        points: (N, 2) float (y, x) at image resolution
        energy_threshold / lambda_geo: energy = L2 + λ * (dist/r)^2
        tden: optional density logits at low resolution for stage3 gating
        use_density_gate: if True, mid-confidence points use full circle;
                          high-confidence use energy gate (stage3 style)
    Returns:
        instance_mask: LongTensor (H, W), 0=bg, >=1 instance id
    """
    device = feature_embedding.device
    dim, H, W = feature_embedding.shape
    N = points.shape[0]
    if N == 0:
        return torch.zeros((H, W), dtype=torch.long, device=device)

    pts = points.clone().to(device).float()
    if N == 1:
        radii = torch.tensor([min(H, W) / 4.0], device=device)
    else:
        dist_matrix = torch.cdist(pts, pts, p=2)
        dist_matrix.fill_diagonal_(float("inf"))
        radii = dist_matrix.min(dim=1)[0]
    radii = radii.clamp(min=3.0, max=min(H, W) / 2.0)

    instance_mask = torch.zeros((H, W), dtype=torch.long, device=device)
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
    )
    grid_coords = torch.stack([y_grid, x_grid], dim=-1).float()

    feature_embedding = gaussian_blur_feat(
        feature_embedding,
        kernel_size=gaussian_kernel_size,
        sigma=gaussian_sigma,
    )
    feat_flat = feature_embedding.permute(1, 2, 0)

    step = max(1, N // max_points) if N > max_points else 1
    p_low, p_mid = 0.1, 0.5
    low_val = math.log(p_low / (1 - p_low))
    mid_val = math.log(p_mid / (1 - p_mid))

    for idx in range(0, N, step):
        center = pts[idx]
        radius = radii[idx]
        if radius <= 0:
            continue
        dist_to_center = torch.norm(grid_coords - center, dim=-1)
        circle_mask = dist_to_center <= radius
        if circle_mask.sum() == 0:
            continue

        if use_density_gate and tden is not None:
            cy = int((center[0] - (down_rate - 1) / 2) / down_rate + 0.5)
            cx = int((center[1] - (down_rate - 1) / 2) / down_rate + 0.5)
            cy = max(0, min(cy, tden.shape[0] - 1))
            cx = max(0, min(cx, tden.shape[1] - 1))
            tden_val = tden[cy, cx]
            if low_val <= tden_val <= mid_val:
                candidate_mask = circle_mask
            elif tden_val > mid_val:
                geo_dist = dist_to_center ** 2 / (radius + 1e-6) ** 2
                center_feat = F.grid_sample(
                    feature_embedding.unsqueeze(0),
                    center.view(1, 1, 1, 2).flip(-1)
                    * 2.0
                    / torch.tensor([W - 1, H - 1], device=device)
                    - 1.0,
                    mode="bilinear",
                    align_corners=True,
                ).squeeze()
                l2_dist = torch.norm(feat_flat - center_feat, dim=-1)
                energy = l2_dist + lambda_geo * geo_dist
                candidate_mask = circle_mask & (energy < energy_threshold)
            else:
                continue
        else:
            geo_dist = dist_to_center ** 2 / (radius + 1e-6) ** 2
            center_feat = F.grid_sample(
                feature_embedding.unsqueeze(0),
                center.view(1, 1, 1, 2).flip(-1)
                * 2.0
                / torch.tensor([W - 1, H - 1], device=device)
                - 1.0,
                mode="bilinear",
                align_corners=True,
            ).squeeze()
            l2_dist = torch.norm(feat_flat - center_feat, dim=-1)
            energy = l2_dist + lambda_geo * geo_dist
            candidate_mask = circle_mask & (energy < energy_threshold)

        instance_mask[candidate_mask] = idx + 1
    return instance_mask


@torch.no_grad()
def generate_pseudo_masks(
    den_logits,
    emb,
    img_hw,
    energy_threshold=0.8,
    lambda_geo=1.0,
    p_low=0.1,
    p_high=0.9,
    use_density_gate=False,
    gaussian_kernel_size=7,
    gaussian_sigma=1.5,
):
    """
    From teacher density + embedding, produce per-image pseudo instance masks.

    Returns:
        masks: list of (1,H,W) float instance maps
        points_batch: list of (N,2) high-res points (y,x)
        conf_mask_batch: list of bool tensors (high-confidence peaks)
        point_seqs: list of (M,2) points with den>=0 (for P2R semi)
    """
    B = den_logits.size(0)
    Hh, Wh = img_hw
    down_rate = Hh // den_logits.size(-2)
    emb_high = F.interpolate(emb, size=(Hh, Wh), mode="bilinear", align_corners=False)

    low = math.log(p_low / (1.0 - p_low))
    high = math.log(p_high / (1.0 - p_high))

    masks, points_batch, conf_batch, seqs = [], [], [], []
    for i in range(B):
        tden = den_logits[i]
        tseq = torch.nonzero((tden >= 0).squeeze(), as_tuple=False)
        low_pts = torch.nonzero((tden >= low).squeeze(), as_tuple=False)
        if low_pts.numel() == 0:
            points_batch.append(torch.zeros((0, 2), device=tden.device))
            conf_batch.append(torch.tensor([], device=tden.device, dtype=torch.bool))
            masks.append(torch.zeros((1, Hh, Wh), device=tden.device))
            seqs.append(tseq.float() * down_rate + (down_rate - 1) / 2)
            continue

        pts_h = low_pts.float() * down_rate + (down_rate - 1) / 2
        peaks = tden.squeeze()[low_pts[:, 0], low_pts[:, 1]]
        conf = peaks > high
        points_batch.append(pts_h)
        conf_batch.append(conf)
        inst = build_instance_map(
            emb_high[i],
            pts_h,
            energy_threshold=energy_threshold,
            lambda_geo=lambda_geo,
            tden=tden.squeeze(0) if tden.dim() == 3 else tden,
            down_rate=down_rate,
            use_density_gate=use_density_gate,
            gaussian_kernel_size=gaussian_kernel_size,
            gaussian_sigma=gaussian_sigma,
        )
        masks.append(inst.unsqueeze(0).float())
        seqs.append(tseq.float() * down_rate + (down_rate - 1) / 2)
    return masks, points_batch, conf_batch, seqs
