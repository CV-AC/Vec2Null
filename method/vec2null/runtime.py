import argparse
import copy
from contextlib import nullcontext
from pathlib import Path
import random

import numpy as np
import torch
import timm

from method.dataset import get_datasets, get_transforms
from .core import TargetWithFeatures


def setup(seed, device="auto"):
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "auto"
        else torch.device(device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu for a local check")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Vec2Null supports CPU and CUDA devices")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    from timm.layers import set_fused_attn

    set_fused_attn(False)
    if device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    return device


def autocast(device):
    if isinstance(device, int):
        device = torch.device("cuda", device)
    else:
        device = torch.device(device)
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


def load_target(path, device, data_root=None, with_data=True):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not all(key in ck for key in ("args", "state_dict", "splits")):
        raise ValueError("Target checkpoint requires args, state_dict and splits")
    args = copy.deepcopy(ck["args"])
    if isinstance(args, dict):
        args = argparse.Namespace(**args)
    if data_root:
        args.data_dir = data_root
    args.device = device
    args.distributed = False
    args.world_size = 1
    get_transforms(args)
    expected_classes = args.num_classes
    manager = get_datasets(args, ck["splits"]) if with_data else None
    if args.num_classes != expected_classes:
        raise ValueError("Dataset class count differs from the target checkpoint")
    model = timm.create_model(
        args.model, num_classes=args.num_classes, pretrained=False
    )
    model.load_state_dict(ck["state_dict"], strict=True)
    target = TargetWithFeatures(model).to(device).eval().requires_grad_(True)
    return target, manager, args, ck


def to_device(episode, device):
    return {
        k: v.to(device, non_blocking=True)
        if torch.is_tensor(v)
        else tuple(t.to(device, non_blocking=True) for t in v)
        for k, v in episode.items()
    }


def save_atomic(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)
