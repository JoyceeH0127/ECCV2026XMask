# -*- coding: utf-8 -*-
"""P2R point-matching density loss (Phase-1)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class L2DIS:
    def __init__(self, factor=1):
        self.factor = factor

    def __call__(self, X, Y):
        return torch.cdist(X, Y) / self.factor


class P2RLoss(nn.modules.loss._Loss):
    def __init__(self, factor=1):
        super().__init__()
        self.factor = factor
        self.cost = L2DIS(1)
        self.min_radis = 8
        self.max_radis = 96
        self.cost_class = 1
        self.cost_point = 8

    def forward(self, dens, seqs, down, masks=None, crop_den_masks=None):
        bs = len(seqs)
        cnt_loss = 0
        for i in range(bs):
            den, seq = dens[i], seqs[i]
            den = den.permute(1, 2, 0)
            H, W = den.shape[:2]
            if seq.size(0) < 1:
                cnt_loss = cnt_loss + F.binary_cross_entropy_with_logits(
                    den, torch.zeros_like(den), weight=torch.ones_like(den) * 0.5
                )
                continue
            A_coord = (
                torch.stack(
                    torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij"),
                    dim=-1,
                )
                .view(1, -1, 2)
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
                maxC = (minC.view_as(A) * M).amax(dim=1, keepdim=True)
                maxC = torch.clip(maxC, min=self.min_radis, max=self.max_radis)
                C = C / maxC
                C = C * self.cost_point - A * self.cost_class
                vid = (M.sum(dim=1) > 0).view(-1)
                C, M = C[..., vid], M[..., vid]
                C2 = M * C + (1 - M) * (C.max() + 1)
                _, mcidx2 = C2.min(dim=1, keepdim=True)
                T = torch.zeros_like(C2).scatter_(1, mcidx2, 1.0).sum(dim=-1).view_as(A)
                T = (T > 0.5).to(A)
                W = T + 1
                if masks is not None:
                    MB = masks[i].view(1, -1, 1).to(A)
                    M = (M @ MB[:, vid, :]) + 1 - M.sum(dim=-1).view_as(A)
                    W = W * M
            if crop_den_masks is not None:
                W = W * crop_den_masks[i].view_as(W)
            cnt_loss = cnt_loss + F.binary_cross_entropy_with_logits(A, T, weight=W)
        return cnt_loss / bs
