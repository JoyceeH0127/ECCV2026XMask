# -*- coding: utf-8 -*-
import os
import copy
import yaml


def _deep_update(base, override):
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _cast(val):
    if not isinstance(val, str):
        return val
    low = val.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        if any(c in val for c in (".", "e", "E")):
            return float(val)
        return int(val)
    except ValueError:
        return val


def _apply_opts(cfg, opts):
    if not opts:
        return
    if len(opts) % 2 != 0:
        raise ValueError("--opts requires KEY VALUE pairs")
    for key, raw_value in zip(opts[0::2], opts[1::2]):
        keys = [part.lower() for part in key.split(".")]
        node = cfg
        for part in keys[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[keys[-1]] = _cast(raw_value)


DEFAULTS = {
    "tag": "xmask",
    "seed": 2024,
    "output": "outputs",
    "print_freq": 50,
    "save_freq": 50,
    "data": {
        "dataset": "sha",
        "data_path": "datasets/SHA",
        "protocol": "protocols/sha-5.txt",
        "label_percent": 0.05,
        "batch_size": 16,
        "num_workers": 4,
        "pin_memory": False,
        "crop_size": [256, 256],
    },
    "model": {"name": "vgg16bn", "resume": ""},
    "train": {
        "base_lr": 5.0e-5,
        "backbone_lr": 1.0e-5,
        "weight_decay": 1.0e-4,
        "clip_grad": 5.0,
        "ema_momentum": 0.998,
        "auto_resume": True,
        "phase1_end": 1500,
        "phase2_end": 2000,
        "epochs": 2100,
        "semi_warmup": 50,
        "lr_decay_epochs": 3500,
        "lr_decay_rate": 0.9,
        "phase_lr": {},
    },
    "pseudo": {
        "energy_threshold": 0.8,
        "lambda_geo": 0.6,
        "p_low": 0.1,
        "p_high": 0.9,
        "use_density_gate": False,
    },
    "loss": {
        "disc": {
            "l2_threshold": 0.6,
            "margin": 0.1,
            "point_mask_weight": 1.0,
            "invalid_disk_radius": 16,
        },
        "mask_constraint": {
            "loss_mode": "p2r_mc",
            "bg_weight": 0.0,
            "fg_weight": 0.005,
            "one_point_weight": 0.0,
        },
    },
}


def load_config(path, opts=None):
    cfg = copy.deepcopy(DEFAULTS)
    with open(path, "r") as f:
        user = yaml.safe_load(f) or {}
    _deep_update(cfg, user)
    _apply_opts(cfg, opts)

    root = os.path.abspath(os.path.join(os.path.dirname(path), ".."))
    for key in ("data_path", "protocol"):
        p = cfg["data"][key]
        if p and not os.path.isabs(p):
            cfg["data"][key] = os.path.normpath(os.path.join(root, p))
    resume = cfg["model"].get("resume") or ""
    if resume and not os.path.isabs(resume):
        cfg["model"]["resume"] = os.path.normpath(os.path.join(root, resume))

    out = cfg["output"]
    if not os.path.isabs(out):
        out = os.path.join(root, out, cfg["tag"])
    else:
        out = os.path.join(out, cfg["tag"])
    cfg["output"] = out
    os.makedirs(cfg["output"], exist_ok=True)
    return cfg


def dump_config(cfg, path):
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
