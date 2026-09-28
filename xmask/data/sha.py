# -*- coding: utf-8 -*-
"""SHA / UCF semi-supervised loaders with point + instance mask annotations."""
import os
import random
import numpy as np
import torch
from torch.utils import data
from PIL import Image

from .transforms import NormalSample


class CrowdSemiDataset(data.Dataset):
    """
    Expected layout under root_path:
      train_data|test_data/
        images/*.jpg
        new-anno/{anno_fmt}
        masks/{id}.npy   # instance id map, 0=bg
    """

    def __init__(self, root_path, mode, protocol_path, anno_fmt, crop_size=(256, 256)):
        self.training = mode == "train"
        self.label, self.unlabel = [], []
        assert protocol_path and os.path.isfile(protocol_path), protocol_path
        with open(protocol_path) as f:
            imgids = set(f.read().strip().split())

        imtype = "jpg"
        img_dir = os.path.join(root_path, mode + "_data", "images")
        for imgf in sorted(os.listdir(img_dir)):
            if not imgf.endswith(imtype):
                continue
            sid = imgf.replace("." + imtype, "")
            if (not self.training) or (imgf in imgids):
                self.label.append(sid)
            self.unlabel.append(sid)

        self.imgpath = os.path.join(root_path, mode + "_data", "images", "{}" + f".{imtype}")
        self.dotpath = os.path.join(root_path, mode + "_data", "new-anno", anno_fmt)
        self.maskpath = os.path.join(root_path, mode + "_data", "masks", "{}.npy")
        self.norm_func = NormalSample(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            crop_size=crop_size,
            train=self.training,
        )
        print(
            f"[training={self.training}] labeled={len(self.label)} unlabeled={len(self.unlabel)}"
        )

    def __len__(self):
        return len(self.unlabel) if self.training else len(self.label)

    def __getitem__(self, index):
        if self.training:
            lid = random.choice(self.label)
            limg, lseq, lmask = self.read_labeled(lid)
            uid = self.unlabel[index]
            uimg, uorig, umask = self.read_unlabeled(uid)
            return limg, lseq, lid, lmask, uimg, uorig, umask, uid
        lid = self.label[index]
        limg, lseq, lmask = self.read_labeled(lid)
        return limg, lseq, lid, lmask

    def read_labeled(self, smpid):
        img = Image.open(self.imgpath.format(smpid)).convert("RGB")
        img = self.norm_func.im2tensor(img)
        mask_path = self.maskpath.format(smpid)
        if os.path.exists(mask_path):
            mask = torch.from_numpy(np.load(mask_path)).float().unsqueeze(0)
        else:
            mask = torch.zeros(1, img.size(1), img.size(2))
        dots = np.load(self.dotpath.format(smpid))
        if dots.ndim == 1:
            dots = dots.reshape(-1, 2)
        dotseq = torch.from_numpy(dots[:, :2]).float()
        return self.norm_func.process_label(img, dotseq, mask)

    def read_unlabeled(self, smpid):
        img = Image.open(self.imgpath.format(smpid)).convert("RGB")
        img_orig = img.copy()
        wa = self.norm_func.im2tensor(img)
        sa = self.norm_func.strong_aug(img)
        img, img_orig = self.norm_func.process_unlabel(torch.cat((wa, sa), 0), img_orig)
        return img, img_orig, self.random_cutout(img)

    @staticmethod
    def random_cutout(uimgs):
        bsize, _, img_h, img_w = uimgs.shape
        cut = torch.ones((bsize, 1, img_h, img_w))
        for i in range(bsize):
            cut_w = int(img_w * (1 / 8 + random.random() * (1 / 4 - 1 / 8)))
            cut_h = int(img_h * (1 / 8 + random.random() * (1 / 4 - 1 / 8)))
            top = random.randint(0, img_h - cut_h)
            left = random.randint(0, img_w - cut_w)
            cut[i, :, top : top + cut_h, left : left + cut_w] = 0
        return cut

    @staticmethod
    def collate_fn(samples):
        if len(samples[0]) > 4:
            limgs, lseqs, lids, lmasks, uimgs, uorig, umask, uids = zip(*samples)
            return (
                torch.cat(limgs, 0),
                sum(lseqs, []),
                lids,
                torch.cat(lmasks, 0),
                torch.cat(uimgs, 0),
                list(uorig),
                torch.cat(umask, 0),
                uids,
            )
        limgs, lseqs, lids, lmasks = zip(*samples)
        return torch.cat(limgs, 0), sum(lseqs, []), lids, torch.cat(lmasks, 0)


class SHADataset(CrowdSemiDataset):
    def __init__(self, root_path, mode, protocol_path, crop_size=(256, 256)):
        super().__init__(
            root_path, mode, protocol_path, anno_fmt="GT_{}.npy", crop_size=crop_size
        )


class UCFDataset(CrowdSemiDataset):
    def __init__(self, root_path, mode, protocol_path, crop_size=(256, 256)):
        super().__init__(
            root_path, mode, protocol_path, anno_fmt="{}_ann.npy", crop_size=crop_size
        )


class JHUDataset(CrowdSemiDataset):
    def __init__(self, root_path, mode, protocol_path, crop_size=(256, 256)):
        super().__init__(
            root_path, mode, protocol_path, anno_fmt="{}.npy", crop_size=crop_size
        )
