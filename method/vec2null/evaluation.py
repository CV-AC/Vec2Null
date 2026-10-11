import torch
from torch.utils.data import Subset

from method.metrics import compute_unlearning_metrics
from .runtime import load_target


@torch.no_grad()
def evaluate_checkpoint(path, data_root, device, workers=6, limit=0):
    target, manager, args, _ = load_target(path, device, data_root)
    args.num_workers = workers
    args.pin_memory = device.type == "cuda"
    args.debug = False
    if limit:
        for name in ("get_unlearning_data", "get_val_data", "get_test_data"):
            original = getattr(manager, name)

            def limited(*positional, _original=original, **keywords):
                value = _original(*positional, **keywords)

                def subset(dataset):
                    return Subset(dataset, range(min(len(dataset), limit)))

                return (
                    {key: subset(ds) for key, ds in value.items()}
                    if isinstance(value, dict)
                    else subset(value)
                )

            setattr(manager, name, limited)
    criterion = (
        torch.nn.BCEWithLogitsLoss()
        if args.task == "multilabel"
        else torch.nn.CrossEntropyLoss()
    )
    metrics = compute_unlearning_metrics(target.model, manager, criterion, None, args)
    result = {}
    for split in ("retain", "forget", "test"):
        result[split] = dict(
            loss=float(metrics[f"{split}/loss"]),
            score=float(metrics[f"{split}/mAP"]),
            metric="mAP" if args.task == "multilabel" else "accuracy",
        )
    result["mia"] = dict(
        auc=float(metrics["mia/auc"]),
        forget_nonmember_rate=float(metrics["mia/acc"]),
        unit="percent",
    )
    result["partial_evaluation"] = bool(limit)
    result["evaluator"] = "shared task metrics; fixed 100-tree gini MIA, seed 0"
    return result
