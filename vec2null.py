import argparse
import json
import os
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from PIL import Image
from torchvision import transforms as T
from method.unoface.model import load_reference
from method.unoface.alignment import reference_alignment, reference_exclusions
from method.vec2null.core import (
    StructuredMetaLoss,
    vectorize,
    inner_update,
    outer_objective,
)
from method.vec2null.projection import calibrate_subspaces, calibrate_utility_gradient
from method.vec2null.data import MetaEpisodes, support_views
from method.vec2null.runtime import (
    setup,
    load_target,
    to_device,
    save_atomic,
    autocast,
)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="Optional JSON defaults; CLI flags override")
    p.add_argument("command", choices=["meta-train", "unlearn", "evaluate"])
    p.add_argument(
        "--target",
        required=True,
        help="Trusted ONE-SHUI pretrain or unlearned checkpoint",
    )
    p.add_argument("--output", required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--mufac-annotations", default=None)
    p.add_argument("--unoface")
    p.add_argument(
        "--reference-detector", help="SCRFD ONNX file used to align UnoFace inputs"
    )
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument(
        "--reference-alignment-root",
        default="external/insightface",
        help="Pinned alignment implementation and isolated CPU dependencies",
    )
    p.add_argument(
        "--reference-failures",
        help="Audited detector-failure manifest for reference inputs only",
    )
    p.add_argument("--artifact", help="Learned Vec2Null artifact for deployment")
    p.add_argument(
        "--request", help='JSON list: [{"identity":"request-id","image":"path"}]'
    )
    p.add_argument("--resume")
    p.add_argument(
        "--probe-artifact",
        help="Reuse a frozen same-seed identity probe",
    )
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--identities", type=int, default=1)
    p.add_argument("--views", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--adapter-dim", type=int, default=64)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--lambda-distribution", type=float, default=0.0)
    p.add_argument("--meta-hidden-norm", action="store_true")
    p.add_argument("--meta-dropout", type=float, default=0.0)
    p.add_argument("--meta-positive-output", action="store_true")
    p.add_argument("--inner-lr", type=float, default=0.1)
    p.add_argument("--inner-weight-decay", type=float, default=1e-5)
    p.add_argument(
        "--utility-guard", choices=["none", "global", "layer"], default="none"
    )
    p.add_argument("--utility-guard-strength", type=float, default=1.0)
    p.add_argument(
        "--utility-gradient-pool",
        choices=["calibration", "retain"],
        default="calibration",
    )
    p.add_argument(
        "--projection-scope", choices=["all", "backbone", "early"], default="all"
    )
    p.add_argument("--projection-mode", choices=["hard", "ridge"], default="hard")
    p.add_argument("--projection-ridge-ratio", type=float, default=0.1)
    p.add_argument("--unprojected-lr-scale", type=float, default=1.0)
    p.add_argument("--utility-bypass", action="store_true")
    p.add_argument("--projection-strength", type=float, default=1.0)
    p.add_argument("--direct-identity-weight", type=float, default=0.0)
    p.add_argument("--retain-target", choices=["teacher", "labels"], default="teacher")
    p.add_argument("--support-supervised-weight", type=float, default=0.0)
    p.add_argument("--lambda-validation", type=float, default=0.0)
    p.add_argument("--lambda-match", type=float, default=1.0)
    p.add_argument("--lambda-retain", type=float, default=0.5)
    p.add_argument("--lambda-orth", type=float, default=0.5)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--energy", type=float, default=0.95)
    p.add_argument("--calibration-rows", type=int, default=2048)
    p.add_argument("--calibration-batches", type=int, default=64)
    p.add_argument("--alignment-steps", type=int, default=500)
    p.add_argument("--matching-mode", choices=["mean", "pairwise"], default="mean")
    p.add_argument(
        "--probe-mode",
        choices=["frozen", "joint"],
        default="frozen",
        help="Freeze the aligned identity probe to prevent nulling by probe rotation",
    )
    p.add_argument("--context-strength", type=float, default=1.0)
    p.add_argument("--calibration-strength", type=float, default=1.0)
    p.add_argument("--identity-interaction", type=float, default=0.0)
    p.add_argument(
        "--reference-views", choices=["paired", "clean", "flip"], default="paired"
    )
    p.add_argument("--anchor-mode", choices=["mean", "medoid"], default="mean")
    p.add_argument(
        "--deployment-preprocessing", choices=["legacy", "target"], default="target"
    )
    p.add_argument(
        "--update-scope", choices=["all", "projected_only", "last4"], default="all"
    )
    p.add_argument(
        "--augmentation", choices=["original", "none", "weak"], default="original"
    )
    p.add_argument("--save-every", type=int, default=100)
    for flag in (
        "anchor",
        "context",
        "features",
        "predictions",
        "projection",
        "confidence",
        "match-pre",
        "match-post",
    ):
        p.add_argument("--no-" + flag, action="store_true")
    p.add_argument("--retrain", help="Optional retain-only checkpoint for ToW")
    p.add_argument(
        "--max-eval-samples",
        type=int,
        default=0,
        help="Smoke only; default full evaluation",
    )
    return p


def read_probe_reuse(output):
    path = Path(output) / "probe_reuse.json"
    return json.loads(path.read_text()) if path.exists() else None


def meta_train(a, device):
    if not a.unoface:
        raise ValueError("--unoface required")
    os.environ["VEC2NULL_AUGMENT"] = a.augmentation
    os.environ["VEC2NULL_REFERENCE_VIEWS"] = a.reference_views
    os.environ["VEC2NULL_ANCHOR_MODE"] = a.anchor_mode
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "meta-last.pth").exists() and not a.resume:
        raise FileExistsError("Use new output or --resume")
    target, manager, args, target_ck = load_target(a.target, device, a.data_root)
    reference, ref_ck = load_reference(a.unoface, device)

    reference_size = ref_ck.get("args", {}).get("size", 112)
    align_reference = reference_alignment(
        ref_ck, a.reference_alignment_root, a.reference_detector
    )
    known_failures = {
        str(Path(path).resolve()) for path in reference_exclusions(a.reference_failures)
    }
    if align_reference is not None:
        align_reference.fallback_paths = known_failures
    if a.episodes <= 0:
        import math

        a.episodes = math.ceil(len(manager.TRAIN) / a.identities)
    episodes = MetaEpisodes(
        manager,
        a.episodes * a.epochs,
        a.identities,
        a.views,
        a.batch_size,
        a.seed,
        reference_alignment=align_reference,
        excluded_reference_paths=(),
    )
    coverage = {
        episodes.row_identity[row]
        for _, request in episodes.requests[: a.episodes]
        for row in request["support"].indices
    }
    if a.episodes >= __import__("math").ceil(len(manager.TRAIN) / a.identities):
        assert coverage == set(manager.TRAIN), (
            "TRAIN identity coverage differs from upstream"
        )
    assert set(episodes.reference_ids) == set(manager.VAL)
    assert not episodes.excluded_reference_rows
    (out / "protocol_audit.json").write_text(
        json.dumps(
            dict(
                train_identity_count=len(manager.TRAIN),
                pseudo_identity_count=len(coverage),
                actual_forget_identities_in_train=sorted(
                    set(manager.FORGET) & set(manager.TRAIN)
                ),
                reference_identity_count=len(episodes.reference_ids),
                reference_image_count=len(episodes.reference_rows),
                extra_reference_exclusions=0,
                support_labels=True,
                support_identity_tokens=True,
                sampler="upstream get_simulated_unlearning_data",
                optimizer="AdamW AMSGrad; weight_decay=0; no gradient clipping",
                requests_per_epoch=a.episodes,
            ),
            indent=2,
        )
    )
    loader = DataLoader(
        episodes,
        batch_size=None,
        num_workers=a.workers,
        pin_memory=True,
        persistent_workers=a.workers > 0,
        generator=torch.Generator().manual_seed(a.seed + 104729),
    )
    meta = StructuredMetaLoss(
        args.num_classes,
        target.model.num_features,
        reference.embedding_dim,
        a.adapter_dim,
        a.hidden_size,
        context_strength=a.context_strength,
        identity_interaction=a.identity_interaction,
        direct_identity_weight=a.direct_identity_weight,
        support_supervised_weight=a.support_supervised_weight,
        hidden_norm=a.meta_hidden_norm,
        dropout=a.meta_dropout,
        positive_output=a.meta_positive_output,
        num_identity_tokens=len(manager.TRAIN),
        no_anchor=a.no_anchor,
        no_context=a.no_context,
        no_features=a.no_features,
        no_predictions=a.no_predictions,
        task=args.task,
    ).to(device)
    optimizer = torch.optim.AdamW(
        meta.parameters(), lr=a.lr, weight_decay=0.0, amsgrad=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs)
    start = 0
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        if ck.get("algorithm_version") != "vec2null-v2":
            raise ValueError("Resume requires a Vec2Null v2 artifact")
        if ck["meta_config"] != meta.config:
            raise ValueError("Resume architecture mismatch")
        for key in (
            "identities",
            "batch_size",
            "matching_mode",
            "probe_mode",
            "views",
            "inner_lr",
            "inner_weight_decay",
            "no_projection",
            "no_confidence",
            "no_match_pre",
            "no_match_post",
            "probe_artifact",
            "meta_hidden_norm",
            "meta_dropout",
            "meta_positive_output",
            "projection_scope",
            "projection_mode",
            "projection_ridge_ratio",
            "unprojected_lr_scale",
            "utility_bypass",
            "utility_guard",
            "utility_guard_strength",
            "utility_gradient_pool",
            "projection_strength",
            "retain_target",
            "support_supervised_weight",
            "lambda_validation",
            "lambda_distribution",
            "lambda_match",
            "lambda_retain",
            "lambda_orth",
            "epsilon",
            "seed",
        ):
            if ck["args"][key] != vars(a)[key]:
                raise ValueError(f"Resume setting changed: {key}")
        meta.load_state_dict(ck["meta"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        bases = {n: u.to(device) for n, u in ck["bases"].items()}
        stats = ck["projection_stats"]
        utility_gradients = {
            n: g.to(device) for n, g in ck.get("utility_gradients", {}).items()
        }
        utility_stats = ck.get("utility_gradient_stats", {})
        start = ck["step"]
        torch.set_rng_state(ck["rng"]["torch"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
    else:
        calibration = DataLoader(
            Subset(manager.test_dataset, episodes.retain_rows),
            batch_size=a.batch_size,
            shuffle=True,
            num_workers=a.workers,
            pin_memory=True,
        )
        bases, stats = calibrate_subspaces(
            target,
            calibration,
            device,
            a.energy,
            a.calibration_rows,
            a.calibration_batches,
            mode=a.projection_mode,
            ridge_ratio=a.projection_ridge_ratio,
        )
        (out / "projection.json").write_text(json.dumps(stats, indent=2))
        utility_gradients, utility_stats = {}, {}
        if a.utility_guard != "none":
            utility_subset = (
                manager.get_unlearning_data(train=False)["retain"]
                if a.utility_gradient_pool == "retain"
                else Subset(manager.test_dataset, episodes.retain_rows)
            )
            pool_rows = set(int(i) for i in utility_subset.indices)
            forget_rows = set(
                int(i)
                for i in manager.get_unlearning_data(train=False)["forget"].indices
            )
            support_rows = set(int(i) for i in manager.support_set)
            if a.utility_gradient_pool == "retain":
                assert not pool_rows & (forget_rows | support_rows), (
                    "Utility pool includes forget/support rows"
                )
            utility_loader = DataLoader(
                utility_subset,
                batch_size=a.batch_size,
                shuffle=True,
                num_workers=a.workers,
                pin_memory=True,
                generator=torch.Generator().manual_seed(a.seed + 9173),
            )
            utility_gradients, utility_stats = calibrate_utility_gradient(
                target, utility_loader, device, a.calibration_batches, args.task
            )
            utility_stats.update(
                pool=a.utility_gradient_pool,
                pool_size=len(pool_rows),
                overlap_forget=len(pool_rows & forget_rows),
                overlap_support=len(pool_rows & support_rows),
            )
            (out / "utility_gradient.json").write_text(
                json.dumps(utility_stats, indent=2)
            )

        if a.probe_artifact:
            probe_ck = torch.load(
                a.probe_artifact, map_location="cpu", weights_only=False
            )
            if (
                a.probe_mode != "frozen"
                or probe_ck["args"].get("probe_mode") != "frozen"
            ):
                raise ValueError("Probe reuse requires frozen probe mode")
            if probe_ck["args"]["seed"] != a.seed:
                raise ValueError("Probe reuse requires matching meta-training seed")
            if probe_ck["identity_map"] != episodes.identity_map:
                raise ValueError("Probe TRAIN identity map mismatch")
            if probe_ck["args"].get("alignment_steps", 0) < 1:
                raise ValueError("Source probe has not been aligned")
            if probe_ck.get("algorithm_version") != "vec2null-v2":
                raise ValueError(
                    "Source probe must use the Vec2Null v2 artifact format"
                )
            meta.pi_s.load_state_dict(
                {"weight": probe_ck["meta"]["pi_s.weight"]}, strict=True
            )
            (out / "probe_reuse.json").write_text(
                json.dumps(
                    dict(
                        source=str(Path(a.probe_artifact).resolve()),
                        source_step=probe_ck["step"],
                        source_alignment_steps=probe_ck["args"]["alignment_steps"],
                        only_frozen_pi_s_reused=True,
                        same_seed=True,
                    ),
                    indent=2,
                )
            )
            del probe_ck
        else:
            alignment_opt = torch.optim.Adam(meta.pi_s.parameters(), lr=a.lr)
            for i, episode in enumerate(loader):
                if i >= a.alignment_steps:
                    break
                e = to_device(episode, device)
                x = e["retain"][0][: a.batch_size]
                if "reference_retain_mask" in e:
                    x = x[e["reference_retain_mask"][: a.batch_size]]
                    if len(x) == 0:
                        continue
                with torch.no_grad():
                    _, f = target(x)
                    reference_x = e.get("reference_retain")
                    if reference_x is not None:
                        reference_x = reference_x[: a.batch_size]
                    if reference_x is None:
                        reference_x = F.interpolate(
                            x,
                            size=(reference_size, reference_size),
                            mode="bilinear",
                            align_corners=False,
                        )
                    z = reference(reference_x)
                loss = (
                    1
                    - (
                        F.normalize(meta.pi_s(F.layer_norm(f, (f.shape[-1],))), dim=-1)
                        * z
                    ).sum(-1)
                ).mean()
                alignment_opt.zero_grad(set_to_none=True)
                loss.backward()
                alignment_opt.step()
    if a.probe_mode == "frozen":
        if not a.resume and not a.probe_artifact and a.alignment_steps < 1:
            raise ValueError("A frozen identity probe must first be aligned")
        meta.pi_s.requires_grad_(False)
    trainable_meta = [
        parameter for parameter in meta.parameters() if parameter.requires_grad
    ]
    config = dict(
        args=vars(a),
        torch=torch.__version__,
        hip=torch.version.hip,
        device=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        development_ids=episodes.development,
        reference_ids=episodes.reference_ids,
        calibration_pool="full TRAIN excluding only the actual request support images, as upstream",
        episode_pool="upstream TRAIN requests / TRAIN meta-retain / full VAL reference",
        probe_reuse=read_probe_reuse(out),
        outer_matching=a.matching_mode,
        identity_probe=a.probe_mode,
        reference_preprocessing=ref_ck.get(
            "preprocessing", "resize and normalize to [-1,1]"
        ),
        excluded_reference_rows=sorted(episodes.excluded_reference_rows),
    )
    (out / "config.json").write_text(json.dumps(config, indent=2))
    t0 = time.time()

    def save(step):
        payload = dict(
            format="vec2null-v1",
            algorithm_version="vec2null-v2",
            probe_reuse=read_probe_reuse(out),
            meta=meta.state_dict(),
            meta_config=meta.config,
            identity_map=episodes.identity_map,
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            step=step,
            args=vars(a),
            bases={n: u.cpu() for n, u in bases.items()},
            projection_stats=stats,
            utility_gradients={n: g.cpu() for n, g in utility_gradients.items()},
            utility_gradient_stats=utility_stats,
            target_args=target_ck["args"],
            reference_size=reference_size,
            rng={
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            },
        )
        save_atomic(payload, out / "meta-last.pth")
        if step % a.episodes == 0:
            save_atomic(payload, out / f"meta-step-{step}.pth")

    for i, episode in enumerate(loader):
        if i < start:
            continue
        e = to_device(episode, device)
        anchor, context = vectorize(
            reference, e.get("reference_support", e["support"]), reference_size
        )
        updated, inner = inner_update(
            target,
            meta,
            e["support"],
            anchor,
            context,
            bases,
            a.inner_lr,
            a.inner_weight_decay,
            create_graph=True,
            no_projection=a.no_projection,
            update_scope=a.update_scope,
            projection_strength=a.projection_strength,
            projection_scope=a.projection_scope,
            unprojected_lr_scale=a.unprojected_lr_scale,
            utility_bypass=a.utility_bypass,
            utility_guard=a.utility_guard,
            utility_guard_strength=a.utility_guard_strength,
            utility_gradients=utility_gradients,
            support_targets=e["support_targets"],
            support_ids=e["support_ids"],
        )

        total = len(e["forget"][0])
        outer = torch.zeros((), device=device)
        parts = {}
        optimizer.zero_grad(set_to_none=True)
        for offset in range(0, total, a.batch_size):
            end = min(offset + a.batch_size, total)
            take = lambda pair, start, stop: tuple(t[start:stop] for t in pair)
            with autocast(device):
                chunk, chunk_stats = outer_objective(
                    target,
                    updated,
                    meta,
                    take(e["forget"], offset, end),
                    take(e["retain"], offset, offset + a.batch_size),
                    take(e["validation"], offset, offset + a.batch_size),
                    anchor,
                    args.task,
                    a.lambda_retain,
                    a.lambda_orth,
                    a.epsilon,
                    not a.no_confidence,
                    not a.no_match_pre,
                    not a.no_match_post,
                    forget_anchor_indices=e["forget_anchor_indices"][offset:end],
                    matching_mode=a.matching_mode,
                    calibration_strength=a.calibration_strength,
                    lambda_match=a.lambda_match,
                    retain_target=a.retain_target,
                    lambda_validation=a.lambda_validation,
                    lambda_distribution=a.lambda_distribution,
                )
            if not torch.isfinite(chunk):
                raise FloatingPointError("Non-finite outer objective")
            weight = (end - offset) / total
            outer += weight * chunk.detach()
            for key, value in chunk_stats.items():
                parts[key] = parts.get(key, 0) + weight * value
            gradients = torch.autograd.grad(
                weight * chunk,
                trainable_meta,
                allow_unused=True,
                retain_graph=end < total,
            )
            for parameter, gradient in zip(trainable_meta, gradients):
                if gradient is not None:
                    if parameter.grad is None:
                        parameter.grad = gradient
                    else:
                        parameter.grad.add_(gradient)
            del chunk, gradients
        gradnorm = torch.linalg.vector_norm(
            torch.stack(
                [p.grad.detach().norm() for p in trainable_meta if p.grad is not None]
            )
        )
        if not torch.isfinite(gradnorm):
            raise FloatingPointError("Non-finite meta gradient")
        optimizer.step()
        if (i + 1) % a.episodes == 0:
            scheduler.step()
        record = dict(
            step=i + 1,
            epoch=(i // a.episodes) + 1,
            inner=float(inner.detach()),
            outer=float(outer.detach()),
            gradnorm=float(gradnorm),
            seconds=time.time() - t0,
            **{k: float(v) for k, v in parts.items()},
        )
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        if i % 10 == 0:
            print(json.dumps(record), flush=True)
        if (i + 1) % a.save_every == 0 or (i + 1) % a.episodes == 0:
            save(i + 1)
        del updated, inner, outer, e
    save(len(episodes))


def unlearn(a, device):
    if not all((a.artifact, a.unoface, a.request)):
        raise ValueError("--artifact --unoface --request required")
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "unlearned.pth").exists():
        raise FileExistsError("Use an independent output directory")
    artifact = torch.load(a.artifact, map_location="cpu", weights_only=False)
    if artifact.get("algorithm_version") != "vec2null-v2":
        raise ValueError("Deployment requires a Vec2Null v2 artifact")
    target, manager, args, ck = load_target(
        a.target, device, a.data_root, with_data=True
    )
    reference, ref_ck = load_reference(a.unoface, device)
    align_reference = reference_alignment(
        ref_ck, a.reference_alignment_root, a.reference_detector
    )
    known_failures = {
        str(Path(path).resolve()) for path in reference_exclusions(a.reference_failures)
    }
    if align_reference is not None:
        align_reference.fallback_paths = known_failures
    meta = StructuredMetaLoss(**artifact["meta_config"]).to(device)
    meta.load_state_dict(artifact["meta"], strict=True)
    meta.eval()
    request = json.loads(Path(a.request).read_text())
    if not request or len({r["identity"] for r in request}) != len(request):
        raise ValueError(
            "Exactly one support image for each distinct requested identity"
        )
    tf = T.Compose(
        [
            T.Resize((args.size, args.size)),
            T.ToTensor(),
            T.Normalize((0.5,) * 3, (0.5,) * 3),
        ]
    )
    if artifact["args"].get("deployment_preprocessing", "legacy") == "target":
        from method.dataset.dataset_utils import get_transforms

        _, tf = get_transforms(args)
    os.environ["VEC2NULL_AUGMENT"] = artifact["args"].get("augmentation", "original")
    os.environ["VEC2NULL_REFERENCE_VIEWS"] = artifact["args"].get(
        "reference_views", "paired"
    )
    os.environ["VEC2NULL_ANCHOR_MODE"] = artifact["args"].get("anchor_mode", "mean")
    views, reference_views = [], []
    support_targets, support_ids = [], []
    dataset = manager.test_dataset
    if hasattr(dataset, "data"):
        paths = dataset.data
    else:
        paths = [
            Path(dataset.root) / dataset.base_folder / "img_align_celeba" / name
            for name in dataset.filename
        ]
    lookup = {str(Path(path).resolve()): i for i, path in enumerate(paths)}
    request_rows = [lookup.get(str(Path(row["image"]).resolve())) for row in request]
    if None in request_rows or set(request_rows) != set(manager.support_set):
        raise ValueError(
            "Requests must contain exactly the checkpoint protocol support images"
        )
    for row, index in zip(request, request_rows):
        _, identity, label = dataset[index]
        if str(row["identity"]) != str(int(identity)):
            raise ValueError("Request identity does not match its protocol annotation")
        support_targets.append(torch.as_tensor(label))
        support_ids.append(artifact["identity_map"][int(identity)])
        with Image.open(row["image"]) as image:
            image = image.convert("RGB")
            if align_reference is None:
                v = support_views(tf(image), artifact["args"]["views"])
                rv = v
            else:
                v, rv = support_views(
                    tf(image), artifact["args"]["views"], align_reference(row["image"])
                )
        views.append(v)
        reference_views.append(rv)
    views = torch.stack(views).to(device)
    reference_views = torch.stack(reference_views).to(device)
    anchor, context = vectorize(reference, reference_views, artifact["reference_size"])
    cfg = artifact["args"]
    updated, loss = inner_update(
        target,
        meta,
        views,
        anchor,
        context,
        artifact["bases"],
        cfg["inner_lr"],
        cfg["inner_weight_decay"],
        create_graph=False,
        no_projection=cfg["no_projection"],
        update_scope=cfg.get("update_scope", "all"),
        projection_strength=cfg.get("projection_strength", 1.0),
        projection_scope=cfg.get("projection_scope", "all"),
        unprojected_lr_scale=cfg.get("unprojected_lr_scale", 1.0),
        utility_bypass=cfg.get("utility_bypass", False),
        utility_guard=cfg.get("utility_guard", "none"),
        utility_guard_strength=cfg.get("utility_guard_strength", 1.0),
        utility_gradients=artifact.get("utility_gradients", {}),
        support_targets=torch.stack(support_targets).to(device),
        support_ids=torch.tensor(support_ids, device=device),
    )

    state = target.model.state_dict()
    for name, weight in updated.items():
        state[name.removeprefix("model.")] = weight.detach().cpu()
    save_atomic(
        dict(
            args=ck["args"],
            splits=ck["splits"],
            state_dict=state,
            method="vec2null",
            request=request,
        ),
        out / "unlearned.pth",
    )
    (out / "request_report.json").write_text(
        json.dumps(
            dict(
                identities=len(request),
                inner_loss=float(loss.detach()),
                context=context.cpu().tolist(),
                seed=a.seed,
                uses_target_labels=True,
                uses_train_identity_tokens=True,
            ),
            indent=2,
        )
    )
    print(f"Saved {out / 'unlearned.pth'}", flush=True)


def parse_options(argv=None):
    p = parser()
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config")
    preliminary, _ = bootstrap.parse_known_args(argv)
    if preliminary.config:
        defaults = json.loads(Path(preliminary.config).read_text())
        unknown = set(defaults) - {action.dest for action in p._actions}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {unknown}")
        actions = {action.dest: action for action in p._actions}
        for key, value in defaults.items():
            action = actions[key]
            if isinstance(
                action, (argparse._StoreTrueAction, argparse._StoreFalseAction)
            ) and not isinstance(value, bool):
                p.error(f"Configuration flag {key} must be a JSON boolean")
            if action.type is not None and value is not None:
                try:
                    defaults[key] = action.type(value)
                except (TypeError, ValueError):
                    p.error(f"Invalid configuration value for {key}")
            if action.choices is not None and defaults[key] not in action.choices:
                p.error(f"Invalid configuration choice for {key}")
        for key in (
            "unoface",
            "reference_alignment_root",
            "reference_detector",
            "data_root",
            "mufac_annotations",
        ):
            if defaults.get(key):
                defaults[key] = str(
                    (
                        Path(preliminary.config).resolve().parent / defaults[key]
                    ).resolve()
                )
        p.set_defaults(**defaults)
    a = p.parse_args(argv)
    for name in (
        "epochs",
        "identities",
        "views",
        "batch_size",
        "save_every",
        "calibration_rows",
        "calibration_batches",
    ):
        if getattr(a, name) <= 0:
            p.error(f"--{name.replace(chr(95), chr(45))} must be positive")
    if a.workers < 0 or a.alignment_steps < 0 or a.max_eval_samples < 0:
        p.error("Worker, alignment and evaluation counts cannot be negative")
    return a


def main():
    a = parse_options()
    if a.mufac_annotations:
        os.environ["MUFAC_ANNOTATIONS"] = str(Path(a.mufac_annotations).resolve())
    device = setup(a.seed, a.device)
    if a.command == "meta-train":
        meta_train(a, device)
    elif a.command == "unlearn":
        unlearn(a, device)
    else:
        from method.vec2null.evaluation import evaluate_checkpoint

        result = evaluate_checkpoint(
            a.target, a.data_root, device, a.workers, a.max_eval_samples
        )
        if a.retrain:
            from method.protocol import validate_pair

            candidate = torch.load(a.target, map_location="cpu", weights_only=False)
            retrain = torch.load(a.retrain, map_location="cpu", weights_only=False)
            validate_pair(candidate, retrain)
            ref = evaluate_checkpoint(
                a.retrain, a.data_root, device, a.workers, a.max_eval_samples
            )
            result["retrain"] = ref
            result["tow"] = 100.0 * float(
                torch.tensor(
                    [
                        1 - abs(result[s]["score"] - ref[s]["score"]) / 100
                        for s in ("retain", "forget", "test")
                    ]
                ).prod()
            )
        out = Path(a.output)
        out.mkdir(parents=True, exist_ok=True)
        (out / "evaluation.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
