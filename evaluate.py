import argparse
import json
from pathlib import Path

import torch

from method import metrics
from method.protocol import field, validate_pair
from method.vec2null.runtime import load_target, setup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--retrain-checkpoint", required=True)
    parser.add_argument("--retrain-results")
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    retrain = torch.load(
        args.retrain_checkpoint, map_location="cpu", weights_only=False
    )
    validate_pair(checkpoint, retrain)
    protocol = dict(
        dataset=field(checkpoint["args"], "dataset"),
        forget_identities=field(checkpoint["args"], "num_identities"),
        model=field(checkpoint["args"], "model"),
        evaluation_seed=field(checkpoint["args"], "seed"),
    )
    if args.protocol:
        supplied = json.loads(Path(args.protocol).read_text())
        for name in ("dataset", "forget_identities", "model"):
            if supplied[name] != protocol[name]:
                raise ValueError(f"Checkpoint protocol mismatch: {name}")
        protocol = supplied
    device = setup(protocol["evaluation_seed"], args.device)
    model, manager, checkpoint_args, _ = load_target(
        args.checkpoint, device, args.data_root
    )
    reference_model, reference_manager, reference_args, _ = load_target(
        args.retrain_checkpoint, device, args.data_root
    )
    for setting in (checkpoint_args, reference_args):
        setting.num_workers = args.workers
        setting.debug = False
        setting.pin_memory = device.type == "cuda"
    criterion = (
        torch.nn.BCEWithLogitsLoss()
        if checkpoint_args.task == "multilabel"
        else torch.nn.CrossEntropyLoss()
    )
    reference = metrics.compute_unlearning_metrics(
        reference_model.model, reference_manager, criterion, None, reference_args
    )
    reference = {name: float(value) for name, value in reference.items()}
    if args.retrain_results:
        supplied = json.loads(Path(args.retrain_results).read_text())
        for name, value in reference.items():
            if name not in supplied or abs(float(supplied[name]) - value) > 1e-5:
                raise ValueError(
                    f"Stored retrain results differ from the current evaluator: {name}"
                )
    result = metrics.compute_unlearning_metrics(
        model.model, manager, criterion, None, checkpoint_args, reference
    )
    result = {name: float(value) for name, value in result.items()}
    result["forget_gap"] = abs(result["forget/mAP"] - reference["forget/mAP"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    record = dict(
        protocol=protocol,
        evaluator="shared task metrics; fixed 100-tree gini MIA, seed 0",
        retrain_metrics_recomputed=True,
    )
    (output / "protocol.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
