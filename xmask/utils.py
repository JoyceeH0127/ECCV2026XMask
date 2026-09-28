# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import torch


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_grad_norm(parameters, norm_type=2):
    parameters = [p for p in parameters if p.grad is not None]
    if not parameters:
        return 0.0
    total = 0.0
    for p in parameters:
        total += p.grad.data.norm(norm_type).item() ** norm_type
    return total ** (1.0 / norm_type)


def _rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not state:
        return False
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda"):
        for device, rng in enumerate(state["cuda"][: torch.cuda.device_count()]):
            torch.cuda.set_rng_state(rng, device=device)
    return True


def save_checkpoint(
    path,
    epoch,
    models,
    best,
    optimizer=None,
    scheduler=None,
    phase=None,
):
    """Save everything required to continue from the next epoch."""
    teacher, student = models
    state = {
        "checkpoint_version": 2,
        "epoch": epoch,
        "teacher": teacher.state_dict(),
        "student": student.state_dict(),
        "best": best,
        "phase": phase,
        "rng_state": _rng_state(),
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    temporary_path = path + ".tmp"
    torch.save(state, temporary_path)
    os.replace(temporary_path, path)


def load_checkpoint(path, models):
    """Load teacher and student weights in the order used by P2RLoss."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    teacher, student = models
    teacher_status = teacher.load_state_dict(checkpoint["teacher"], strict=False)
    student_status = student.load_state_dict(checkpoint["student"], strict=False)
    return checkpoint, teacher_status, student_status


def restore_training_state(checkpoint, optimizer=None, scheduler=None):
    """Restore optimizer, scheduler, and RNG state from a training checkpoint."""
    restored = {
        "optimizer": False,
        "scheduler": False,
        "rng": restore_rng_state(checkpoint.get("rng_state")),
    }
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        restored["optimizer"] = True
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
        restored["scheduler"] = True
    return restored


def auto_resume(output_dir):
    if not os.path.isdir(output_dir):
        return None
    ckpts = [f for f in os.listdir(output_dir) if f.endswith(".pth")]
    if not ckpts:
        return None
    return max((os.path.join(output_dir, f) for f in ckpts), key=os.path.getmtime)


def ema_update(teacher, student, momentum=0.998):
    """Apply the parameter-only EMA update used in P2RLoss/main_backup.py."""
    with torch.no_grad():
        for para_t, para_s in zip(teacher.parameters(), student.parameters()):
            para_t.data.copy_(
                para_t.data * momentum + para_s.data * (1.0 - momentum)
            )


def set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad = flag
