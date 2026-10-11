import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from piu.joint_piu import repair
from method.protocol import equal, validate_pair
from method.vec2null.runtime import setup


def names(tensors):
    return {name.removeprefix("model."): value for name, value in tensors.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--basis", required=True)
    parser.add_argument("--original-gradient", required=True)
    parser.add_argument("--candidate-gradient", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--strength", type=float, default=0.25)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    if not 0 <= args.strength <= 1:
        raise ValueError("strength must be in [0, 1]")
    device = setup(0, args.device)
    paths = {
        "target": args.target,
        "candidate": args.candidate,
        "basis": args.basis,
        "original_gradient": args.original_gradient,
        "candidate_gradient": args.candidate_gradient,
    }
    artifacts = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in paths.items()
    }
    target = artifacts["target"]
    candidate = artifacts["candidate"]
    validate_pair(target, candidate)
    for key, source in (
        ("original_gradient", "target"),
        ("candidate_gradient", "candidate"),
    ):
        calibration = artifacts[key]
        if calibration.get("calibration_pool") != "retain":
            raise ValueError(f"Gradient calibration must use the retain pool: {key}")
        if not equal(calibration.get("splits"), artifacts[source]["splits"]):
            raise ValueError(f"Gradient calibration protocol mismatch: {key}")
    if artifacts["original_gradient"].get("seed") != artifacts[
        "candidate_gradient"
    ].get("seed"):
        raise ValueError("PIU gradients must use the same retain sampling seed")
    bases = {
        name: value.to(device)
        for name, value in names(artifacts["basis"]["bases"]).items()
    }
    gradients = []
    for key in ("original_gradient", "candidate_gradient"):
        gradients.append(
            {
                name: value.to(device)
                for name, value in names(artifacts[key]["utility_gradients"]).items()
            }
        )
    weights = {
        name: value.to(device)
        for name, value in candidate["state_dict"].items()
        if value.is_floating_point()
    }
    anchors = {name: target["state_dict"][name].to(device) for name in weights}
    updates = {name: value - anchors[name] for name, value in weights.items()}
    repaired, audit = repair(updates, bases, gradients[0], gradients[1], args.strength)
    state = dict(candidate["state_dict"])
    changed = []
    for name in bases:
        state[name] = (anchors[name] + repaired[name]).to(state[name].dtype).cpu()
        if not torch.equal(state[name], candidate["state_dict"][name]):
            changed.append(name)
    provenance = {
        "operator": "joint task-sensitive PIU",
        "strength": args.strength,
        "changed_parameters": changed,
        "sources": {name: {"path": str(Path(path))} for name, path in paths.items()},
        "operator_audit": audit,
    }
    result = dict(candidate)
    result["state_dict"] = state
    result["method"] = "vec2null-joint-task-piu"
    result["joint_piu_provenance"] = provenance
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Use an independent output checkpoint")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(result, temporary)
    temporary.replace(output)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
