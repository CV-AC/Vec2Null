import numpy as np
import torch


def field(args, name):
    return args[name] if isinstance(args, dict) else getattr(args, name)


def equal(left, right):
    if torch.is_tensor(left) and torch.is_tensor(right):
        return torch.equal(left, right)
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return left == right


def validate_pair(candidate, reference):
    for name in ("dataset", "num_identities", "model", "seed", "num_classes"):
        if field(candidate["args"], name) != field(reference["args"], name):
            raise ValueError(f"Checkpoint setting mismatch: {name}")
    if not equal(candidate["splits"], reference["splits"]):
        raise ValueError(
            "Checkpoints must use identical identity splits and support images"
        )
    left, right = candidate["state_dict"], reference["state_dict"]
    if left.keys() != right.keys() or any(
        left[name].shape != right[name].shape for name in left
    ):
        raise ValueError("Checkpoint model structures differ")
