import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from method.vec2null.projection import calibrate_utility_gradient
from method.vec2null.runtime import load_target, save_atomic, setup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batches", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError("Use an independent calibration output")
    device = setup(args.seed, args.device)
    target, manager, settings, checkpoint = load_target(
        args.checkpoint, device, args.data_root
    )
    retain = manager.get_unlearning_data(train=False)["retain"]
    loader = DataLoader(
        retain,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    gradients, stats = calibrate_utility_gradient(
        target, loader, device, args.batches, settings.task
    )
    save_atomic(
        dict(
            utility_gradients={name: value.cpu() for name, value in gradients.items()},
            stats=stats,
            calibration_pool="retain",
            splits=checkpoint["splits"],
            seed=args.seed,
        ),
        args.output,
    )


if __name__ == "__main__":
    main()
