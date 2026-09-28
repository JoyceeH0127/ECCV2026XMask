# -*- coding: utf-8 -*-
import random
from PIL import ImageFilter, Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF


class GaussianBlur:
    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))


class NormalSample:
    def __init__(self, mean, std, crop_size=(256, 256), resize_factor=0.3, train=False):
        self.half_h, self.half_w = crop_size
        self.train = train
        self.scale_factor = resize_factor
        self.im2tensor = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )
        self.strong_aug = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.25),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.8),
                self.im2tensor,
            ]
        )

    def process_label(self, image, dotseq, mask_gt):
        if self.train:
            images, dotseqs, masks_gt = self.crop_and_resize(image, dotseq, mask_gt)
        else:
            images, dotseqs, masks_gt = image.unsqueeze(0), [dotseq], mask_gt.unsqueeze(0)

        h, w = images.shape[-2:]
        if h % 32 != 0 or w % 32 != 0:
            ph = (32 - h % 32) % 32
            pw = (32 - w % 32) % 32
            images = F.pad(images, (0, pw, 0, ph))
            masks_gt = F.pad(masks_gt, (0, pw, 0, ph))
            h, w = images.shape[-2:]

        for i in range(images.size(0)):
            if self.train and random.randint(0, 1):
                images[i] = torch.flip(images[i], dims=(-1,))
                masks_gt[i] = torch.flip(masks_gt[i], dims=(-1,))
                dotseqs[i][:, 0] = w - dotseqs[i][:, 0] - 1

        for i, seq in enumerate(dotseqs):
            u = self.nearest(seq)
            # store as (y, x, nearest_dist)
            dotseqs[i] = torch.cat((seq[:, [1, 0]], u), dim=1)
        return images, dotseqs, masks_gt

    def process_unlabel(self, image, img_orig):
        images, imgs_orig = self.crop_and_resize_pro(image, img_orig)
        h, w = images.shape[-2:]
        if h % 32 != 0 or w % 32 != 0:
            ph = (32 - h % 32) % 32
            pw = (32 - w % 32) % 32
            images = F.pad(images, (0, pw, 0, ph))
            imgs_orig = [TF.pad(p, padding=(0, 0, pw, ph), fill=0) for p in imgs_orig]
            h, w = images.shape[-2:]
        for i in range(images.size(0)):
            if self.train and random.randint(0, 1):
                images[i] = torch.flip(images[i], dims=(-1,))
                imgs_orig = [p.transpose(Image.FLIP_LEFT_RIGHT) for p in imgs_orig]
        return images, imgs_orig

    def crop_and_resize_pro(self, image, image_orig, num_patches=1):
        imh, imw = image.shape[-2:]
        image_orig_t = TF.to_tensor(image_orig)
        scale = random.random() * (self.scale_factor * 2) + (1 - self.scale_factor)
        crop_h = int(self.half_h / scale + 0.5)
        crop_w = int(self.half_w / scale + 0.5)
        if crop_h > imh or crop_w > imw:
            image = F.pad(image, (0, crop_w - imw, 0, crop_h - imh), value=0)
            image_orig_t = F.pad(image_orig_t, (0, crop_w - imw, 0, crop_h - imh), value=0)
            imh, imw = image.shape[-2:]
        crop_imgs, crop_imgs_orig = [], []
        for _ in range(num_patches):
            sh = random.randint(0, imh - crop_h)
            sw = random.randint(0, imw - crop_w)
            crop_imgs.append(image[:, sh : sh + crop_h, sw : sw + crop_w])
            c = image_orig_t[:, sh : sh + crop_h, sw : sw + crop_w].unsqueeze(0)
            c = F.interpolate(c, (self.half_h, self.half_w), mode="bilinear", align_corners=False)
            crop_imgs_orig.append(TF.to_pil_image(c.squeeze(0)))
        crop_imgs = F.interpolate(
            torch.stack(crop_imgs, 0),
            (self.half_h, self.half_w),
            mode="bilinear",
            align_corners=False,
        )
        return crop_imgs, crop_imgs_orig

    def crop_and_resize(self, image, dotseq=None, mask=None, num_patches=1):
        imh, imw = image.shape[-2:]
        scale = random.random() * (self.scale_factor * 2) + (1 - self.scale_factor)
        crop_h = int(self.half_h / scale + 0.5)
        crop_w = int(self.half_w / scale + 0.5)
        if crop_h > imh or crop_w > imw:
            image = F.pad(image, (0, crop_w - imw, 0, crop_h - imh), value=0)
            if mask is not None:
                mask = F.pad(mask, (0, crop_w - imw, 0, crop_h - imh), value=0)
            imh, imw = image.shape[-2:]
        crop_imgs, crop_dots, crop_masks = [], [], []
        rh, rw = self.half_h / crop_h, self.half_w / crop_w
        for _ in range(num_patches):
            sh = random.randint(0, imh - crop_h)
            sw = random.randint(0, imw - crop_w)
            crop_imgs.append(image[:, sh : sh + crop_h, sw : sw + crop_w])
            if mask is not None:
                c_mask = mask[..., sh : sh + crop_h, sw : sw + crop_w]
                crop_masks.append(c_mask)
                crop_masks = torch.stack(crop_masks, 0)
                crop_masks = F.interpolate(
                    crop_masks, (self.half_h, self.half_w), mode="nearest"
                )
            if dotseq is not None:
                idx = (
                    (dotseq[:, 0] >= sw)
                    & (dotseq[:, 0] <= sw + crop_w)
                    & (dotseq[:, 1] >= sh)
                    & (dotseq[:, 1] <= sh + crop_h)
                )
                selected = dotseq[idx].clone()
                selected[:, 0] = (selected[:, 0] - sw) * rw
                selected[:, 1] = (selected[:, 1] - sh) * rh
                crop_dots.append(selected)
        crop_imgs = F.interpolate(
            torch.stack(crop_imgs, 0),
            (self.half_h, self.half_w),
            mode="bilinear",
            align_corners=False,
        )
        return crop_imgs, crop_dots, crop_masks

    @staticmethod
    def nearest(seq):
        seqlen = seq.size(0)
        if seqlen <= 1:
            return torch.zeros(seqlen, 1) + 32
        xx = (seq ** 2).sum(dim=-1, keepdim=True)
        L2 = (xx - 2 * (seq @ seq.T) + xx.T).relu()
        m = torch.kthvalue(L2, 2, dim=1).values
        return m.view(-1, 1)
