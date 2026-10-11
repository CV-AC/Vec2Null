import torch
from torch import nn


def calibrate_utility_gradient(
    target, loader, device, max_batches=32, task="multilabel"
):

    from .core import task_loss

    if max_batches <= 0:
        raise ValueError("Utility calibration requires positive max_batches")
    parameters = [(n, p) for n, p in target.named_parameters() if p.requires_grad]
    accumulated, examples = {}, 0
    was_training = target.training
    target.eval()
    try:
        for index, batch in enumerate(loader):
            if index >= max_batches:
                break
            x, y = batch[0].to(device), batch[2].to(device)
            output = target(x)
            logits = output[0] if isinstance(output, tuple) else output
            loss = task_loss(logits, y, task).mean()
            gradients = torch.autograd.grad(
                loss, [p for _, p in parameters], allow_unused=True
            )
            for (name, _), gradient in zip(parameters, gradients):
                if gradient is None:
                    continue
                value = gradient.detach().float() * len(x)
                if name in accumulated:
                    accumulated[name].add_(value)
                else:
                    accumulated[name] = value.clone()
            examples += len(x)
    finally:
        target.train(was_training)
    if examples == 0 or not accumulated:
        raise ValueError("Empty utility gradient calibration")
    result = {
        name: value.div_(examples).detach() for name, value in accumulated.items()
    }
    if not all(torch.isfinite(value).all() for value in result.values()):
        raise ValueError("Nonfinite utility calibration gradient")
    norm = sum(float(value.square().sum()) for value in result.values()) ** 0.5
    if norm <= 1e-12:
        raise ValueError("Degenerate utility calibration gradient")
    return result, dict(
        examples=examples,
        parameter_tensors=len(result),
        gradient_norm=norm,
        task=task,
        precision="FP32",
        target_modified=False,
    )


@torch.no_grad()
def calibrate_subspaces(
    target,
    loader,
    device,
    energy=0.95,
    max_rows=512,
    max_batches=32,
    mode="hard",
    ridge_ratio=0.1,
):

    if max_rows <= 0 or max_batches <= 0:
        raise ValueError("Projection calibration counts must be positive")
    if not 0 < energy <= 1:
        raise ValueError("energy must be in (0,1]")
    if mode not in ("hard", "ridge") or ridge_ratio <= 0:
        raise ValueError("Invalid projection mode or ridge ratio")
    samples = {}
    handles = []

    def hook(name):
        def collect(module, args, output):
            x = args[0].detach().reshape(-1, module.in_features)

            n = min(max(1, (max_rows + max_batches - 1) // max_batches), len(x))
            ix = torch.linspace(0, len(x) - 1, n, device=x.device).long()
            values = samples.setdefault(name, [])
            remaining = max_rows - sum(len(v) for v in values)
            if remaining > 0:
                values.append(x[ix[:remaining]].float().cpu())

        return collect

    for name, module in target.named_modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(hook(name + ".weight")))
    try:
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            x = batch[0].to(device, non_blocking=True)
            target(x)
    finally:
        for h in handles:
            h.remove()
    bases = {}
    stats = {}
    for name, parts in samples.items():
        x = torch.cat(parts).to(device)
        _, s, vh = torch.linalg.svd(x, full_matrices=False)
        sq = s.square()
        total = sq.sum()
        if total <= 1e-12:
            bases[name] = torch.empty(x.shape[1], 0, device=device)
            rank = 0
            kept = 0.0
        elif mode == "ridge":
            ridge = ridge_ratio * sq[0]
            weights = sq / (sq + ridge)
            bases[name] = (vh.T * weights.sqrt().unsqueeze(0)).contiguous()
            rank = len(s)
            kept = float((weights * sq).sum() / total)
        else:
            curve = sq.cumsum(0) / total
            rank = min(int(torch.searchsorted(curve, energy).item()) + 1, len(s))
            bases[name] = vh[:rank].T.contiguous()
            kept = float(curve[rank - 1])
        stats[name] = dict(rank=rank, input_dim=x.shape[1], rows=len(x), energy=kept)
        if mode == "ridge":
            stats[name].update(
                operator="spectral_ridge",
                ridge_ratio=ridge_ratio,
                energy_definition="fraction of activation energy weighted by retention operator",
            )
    if not bases:
        raise ValueError("No Linear activation subspaces collected")
    return bases, stats
