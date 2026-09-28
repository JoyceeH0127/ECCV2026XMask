# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader
from .sha import JHUDataset, SHADataset, UCFDataset


def build_loader(cfg, mode="train"):
    name = cfg["dataset"].lower()
    Dataset = {
        "sha": SHADataset,
        "shha": SHADataset,
        "ucf": UCFDataset,
        "jhu": JHUDataset,
    }[name]
    ds = Dataset(
        root_path=cfg["data_path"],
        mode=mode if mode != "val" else "test",
        protocol_path=cfg["protocol"],
        crop_size=tuple(cfg.get("crop_size", [256, 256])),
    )
    return DataLoader(
        ds,
        batch_size=cfg["batch_size"] if mode == "train" else 1,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=cfg.get("pin_memory", False),
        shuffle=(mode == "train"),
        collate_fn=Dataset.collate_fn,
    )
