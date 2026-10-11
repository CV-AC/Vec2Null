import torch


def project_two(delta, normals, strength=1.0):
    if len(normals) != 2 or not 0 <= strength <= 1:
        raise ValueError("Expected two task normals and strength in [0,1]")
    if not torch.isfinite(delta).all() or any(
        not torch.isfinite(normal).all() for normal in normals
    ):
        raise ValueError("PIU inputs must be finite")
    work = delta.double()
    unit = []
    for normal in normals:
        normal = normal.detach().to(work)
        norm = normal.norm()
        if norm > 1e-24:
            unit.append(normal / norm)
    if not unit or strength == 0:
        return delta, dict(active=False)
    score = torch.stack([(work * q).sum() for q in unit])
    if bool((score <= 0).all()):
        return delta, dict(active=False)
    gram = torch.stack([torch.stack([(a * b).sum() for b in unit]) for a in unit])
    tolerance = 1e-10 * max(1.0, float(work.detach().norm()))
    candidates = []

    def consider(multipliers):
        if bool((multipliers < -1e-12).any()):
            return
        correction = sum(lam * q for lam, q in zip(multipliers, unit))
        residual = score - gram @ multipliers
        if bool((residual <= tolerance).all()):
            candidates.append((correction, multipliers, residual))

    for index in range(len(unit)):
        mask = torch.zeros_like(score)
        mask[index] = 1
        consider(mask * score[index].clamp_min(0) / gram[index, index])
    if len(unit) == 2 and float(torch.linalg.det(gram).detach()) > 1e-12:
        consider(torch.linalg.solve(gram, score))
    if not candidates:
        raise RuntimeError(
            "No numerically feasible two-halfspace projection; do not silently weaken constraint"
        )
    correction, multipliers, residual = min(
        candidates, key=lambda item: float(item[0].detach().square().sum())
    )
    assert float((multipliers * residual).detach().abs().max()) <= tolerance * 10
    result = (work - strength * correction).to(delta.dtype)
    audit = dict(
        active=True,
        before=score.detach().tolist(),
        full_strength_after=residual.detach().tolist(),
        normalized_multipliers=multipliers.detach().tolist(),
        full_correction_norm=float(correction.detach().norm()),
        complementarity_max=float((multipliers * residual).detach().abs().max()),
    )
    return result, audit


def repair(updates, bases, gradient_a, gradient_b, strength):
    result, audit = dict(updates), {}
    for name, basis in bases.items():
        u = basis.detach().to(device=updates[name].device, dtype=torch.float64)

        normals = [
            (g[name].detach().double() @ u) @ u.T for g in [gradient_a, gradient_b]
        ]
        result[name], audit[name] = project_two(updates[name], normals, strength)
    return result, audit


def verify(device="cpu"):
    torch.manual_seed(283)
    options = dict(device=device, dtype=torch.float64)
    a, b = (
        torch.tensor([1.0, 0.0, 0.0], **options),
        torch.tensor([0.0, 1.0, 0.0], **options),
    )
    delta = torch.tensor([1.0, 2.0, 3.0], **options)
    result, _ = project_two(delta, [a, b])
    assert torch.equal(result, torch.tensor([0.0, 0.0, 3.0], **options))
    result, _ = project_two(delta, [a, b], 0.25)
    assert torch.equal(result, torch.tensor([0.75, 1.5, 3.0], **options))
    for normals in [[a, a], [a, -a], [a, torch.zeros_like(a)]]:
        result, _ = project_two(delta, normals)
        assert all(float((result * n).sum()) <= 1e-12 for n in normals)
    for _ in range(100):
        d, a, b = [torch.randn(12, **options) for _ in range(3)]
        out, audit = project_two(d, [a, b])
        assert max(float((out * n).sum()) for n in [a, b]) < 1e-8
        assert out.norm() <= d.norm() + 1e-10
        twice, _ = project_two(out, [a, b])
        assert torch.allclose(twice, out, atol=1e-9, rtol=0)
        if audit["active"]:
            q = [a / a.norm(), b / b.norm()]
            lam = torch.tensor(audit["normalized_multipliers"], **options)
            assert float(lam.min()) >= -1e-12
            assert torch.allclose(
                d - out, sum(v * n for v, n in zip(lam, q)), atol=1e-9
            )
            assert max(abs(float(lam[i] * (out * q[i]).sum())) for i in range(2)) < 1e-8
    d = torch.tensor([1.0, 2.0, 3.0], requires_grad=True, **options)
    a, b = (
        torch.tensor([1.0, 0.0, 0.0], **options),
        torch.tensor([-0.5, 0.8, 0.0], **options),
    )
    fn = lambda x: project_two(x, [a, b], 0.25)[0]
    assert torch.autograd.gradcheck(fn, (d,))
    assert torch.autograd.gradgradcheck(fn, (d,))
    return dict(
        analytic_cases=True,
        degenerate_normals=True,
        feasibility=True,
        norm_nonincrease=True,
        idempotence=True,
        kkt_optimality=True,
        gradcheck=True,
        gradgradcheck=True,
    )
