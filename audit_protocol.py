import argparse
import json
from pathlib import Path
from method.protocol import field, equal, validate_pair

import torch


def descriptor(checkpoint):
    args = checkpoint["args"]
    return {name: field(args, name) for name in ("dataset", "num_identities", "model")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol")
    parser.add_argument("--vec2null", required=True)
    parser.add_argument("--metaunlearn", required=True)
    parser.add_argument("--retrain", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    paths = {
        "vec2null": args.vec2null,
        "metaunlearn": args.metaunlearn,
        "retrain": args.retrain,
    }
    checkpoints = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in paths.items()
    }
    for checkpoint in checkpoints.values():
        validate_pair(checkpoint, checkpoints["retrain"])
    protocol = (
        json.loads(Path(args.protocol).read_text())
        if args.protocol
        else dict(
            dataset=field(checkpoints["retrain"]["args"], "dataset"),
            forget_identities=field(checkpoints["retrain"]["args"], "num_identities"),
            model=field(checkpoints["retrain"]["args"], "model"),
        )
    )
    expected = {
        "dataset": protocol["dataset"],
        "num_identities": protocol["forget_identities"],
        "model": protocol["model"],
    }
    descriptions = {
        name: descriptor(checkpoint) for name, checkpoint in checkpoints.items()
    }
    if any(value != expected for value in descriptions.values()):
        raise ValueError(f"Checkpoint setting mismatch: {descriptions}")
    split = checkpoints["retrain"]["splits"]
    if not all(
        equal(split, checkpoint["splits"]) for checkpoint in checkpoints.values()
    ):
        raise ValueError(
            "Vec2Null, MetaUnlearn and retrain checkpoints must use identical splits"
        )
    report = {
        "passed": True,
        "protocol": protocol,
        "checkpoint_settings": descriptions,
        "identical_splits": True,
        "common_evaluator": "evaluate.py",
    }
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
