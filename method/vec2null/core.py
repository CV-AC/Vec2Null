from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call


class TargetWithFeatures(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        features = self.model.forward_head(
            self.model.forward_features(x), pre_logits=True
        )
        return self.model.get_classifier()(features), features


class StructuredMetaLoss(nn.Module):
    def __init__(
        self,
        classes,
        feature_dim,
        embedding_dim=512,
        adapter_dim=64,
        hidden=128,
        context_strength=1.0,
        identity_interaction=0.0,
        num_identity_tokens=0,
        direct_identity_weight=0.0,
        support_supervised_weight=0.0,
        hidden_norm=False,
        dropout=0.0,
        positive_output=False,
        no_anchor=False,
        no_context=False,
        no_features=False,
        no_predictions=False,
        task="multilabel",
    ):
        super().__init__()
        self.config = dict(
            classes=classes,
            feature_dim=feature_dim,
            embedding_dim=embedding_dim,
            adapter_dim=adapter_dim,
            hidden=hidden,
            context_strength=context_strength,
            identity_interaction=identity_interaction,
            num_identity_tokens=num_identity_tokens,
            direct_identity_weight=direct_identity_weight,
            support_supervised_weight=support_supervised_weight,
            hidden_norm=hidden_norm,
            dropout=dropout,
            positive_output=positive_output,
            no_anchor=no_anchor,
            no_context=no_context,
            no_features=no_features,
            no_predictions=no_predictions,
            task=task,
        )
        self.pi_f = nn.Linear(feature_dim, adapter_dim)
        self.pi_z = nn.Linear(embedding_dim, adapter_dim)
        self.pi_c = nn.Sequential(nn.Linear(3, adapter_dim), nn.GELU())
        self.pi_s = nn.Linear(feature_dim, embedding_dim, bias=False)
        self.pi_y = nn.Linear(classes, adapter_dim, bias=False)
        self.pi_id = (
            nn.Embedding(num_identity_tokens, adapter_dim)
            if num_identity_tokens
            else None
        )
        if self.pi_id is not None:
            nn.init.normal_(self.pi_id.weight, std=0.02)
        self.gate = nn.Linear(adapter_dim, 1)
        if not 0 <= dropout < 1:
            raise ValueError("Meta dropout must be in [0,1)")
        layers = []
        for input_dim in [classes + 3 * adapter_dim, hidden]:
            layers.append(nn.Linear(input_dim, hidden))
            if hidden_norm:
                layers.append(nn.LayerNorm(hidden))
            layers.append(nn.GELU())
            if dropout:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, 1))
        if positive_output:
            layers.append(nn.Softplus())
        self.h = nn.Sequential(*layers)

    def forward(
        self, logits, features, anchor, context, support_targets=None, support_ids=None
    ):

        f = self.pi_f(F.layer_norm(features, (features.shape[-1],)))
        if support_targets is not None:
            labels = (
                F.one_hot(support_targets.long(), self.config["classes"])
                if self.config["task"] == "classification"
                else support_targets
            )
            f = f + self.pi_y(labels.float())
        z = self.pi_z(anchor)
        if support_ids is not None and self.pi_id is not None:
            z = z + self.pi_id(support_ids.long())
        if self.config["identity_interaction"]:
            identity_feature = F.normalize(
                self.pi_s(F.layer_norm(features, (features.shape[-1],))), dim=-1
            )
            z = z + self.config["identity_interaction"] * self.pi_z(
                identity_feature * anchor
            )
        strength = self.config["context_strength"]
        c = strength * self.pi_c(context)
        prediction = (
            logits.sigmoid()
            if self.config["task"] == "multilabel"
            else logits.softmax(-1)
        )
        if self.config["no_features"]:
            f = torch.zeros_like(f)
        if self.config["no_anchor"]:
            z = torch.zeros_like(z)
        if self.config["no_predictions"]:
            prediction = torch.zeros_like(prediction)
        gamma = F.softplus(self.gate(c))
        if strength != 1.0:
            gamma = 1 + strength * (gamma - 1)
        if self.config["no_context"]:
            c = torch.zeros_like(c)
            gamma = torch.ones_like(gamma)
        learned = (gamma * self.h(torch.cat([prediction, f, z, c], dim=-1))).mean()
        if self.config["direct_identity_weight"]:
            projected = F.normalize(
                self.pi_s(F.layer_norm(features, (features.shape[-1],))), dim=-1
            )
            learned = (
                learned
                + self.config["direct_identity_weight"]
                * (projected * anchor).sum(-1).mean()
            )
        if self.config["support_supervised_weight"]:
            if support_targets is None:
                raise ValueError("Supervised residual requires support labels")
            learned = (
                learned
                + self.config["support_supervised_weight"]
                * task_loss(logits, support_targets, self.config["task"]).mean()
            )
        return learned


@torch.no_grad()
def vectorize(reference, views, size=112):
    b, k, c, h, w = views.shape
    x = F.interpolate(
        views.flatten(0, 1), size=(size, size), mode="bilinear", align_corners=False
    )
    z = reference(x).float().view(b, k, -1)
    import os

    if os.environ.get("VEC2NULL_ANCHOR_MODE", "mean") == "medoid":
        scores = (z @ z.transpose(1, 2)).sum(-1)
        anchor = z[torch.arange(b, device=z.device), scores.argmax(-1)]
    else:
        anchor = F.normalize(z.sum(1), dim=-1)
    r = (z * anchor[:, None]).sum(-1)
    context = torch.stack(
        [r.mean(1), r.std(1, unbiased=False), 1 - r.min(1).values], dim=-1
    )
    return anchor.detach(), context.detach()


def guard_utility_directions(
    directions, utility_gradients, mode="global", strength=1.0
):

    if mode not in ("global", "layer") or not 0 <= strength <= 1:
        raise ValueError("Invalid utility guard mode or strength")
    if strength == 0:
        return dict(directions)
    missing = set(directions) - set(utility_gradients)
    if missing:
        raise ValueError(
            "Missing utility calibration gradients: " + str(sorted(missing))
        )
    gradients = {
        name: utility_gradients[name].detach().to(value)
        for name, value in directions.items()
    }
    if not directions:
        return {}
    groups = [[name] for name in directions] if mode == "layer" else [list(directions)]
    result = dict(directions)
    for names in groups:
        dot = sum((directions[name] * gradients[name]).sum() for name in names)
        norm = sum(gradients[name].square().sum() for name in names)
        coefficient = strength * F.relu(-dot) / norm.clamp_min(1e-24)
        for name in names:
            result[name] = directions[name] + coefficient * gradients[name]
    return result


def inner_update(
    target,
    meta,
    views,
    anchor,
    context,
    bases,
    lr=0.1,
    weight_decay=0.0,
    create_graph=True,
    no_projection=False,
    update_scope="all",
    projection_strength=1.0,
    support_targets=None,
    support_ids=None,
    projection_scope="all",
    unprojected_lr_scale=1.0,
    utility_bypass=False,
    utility_guard="none",
    utility_guard_strength=1.0,
    utility_gradients=None,
):

    b, k = views.shape[:2]
    params = OrderedDict(target.named_parameters())
    logits, features = target(views.flatten(0, 1))
    loss = meta(
        logits,
        features,
        anchor.repeat_interleave(k, 0),
        context.repeat_interleave(k, 0),
        support_targets.repeat_interleave(k, 0)
        if support_targets is not None
        else None,
        support_ids.repeat_interleave(k, 0) if support_ids is not None else None,
    )
    active = [(n, p) for n, p in params.items() if p.requires_grad]
    utility_grads = {}
    if utility_bypass:
        coefficient = meta.config.get("support_supervised_weight", 0.0)
        if coefficient <= 0 or support_targets is None:
            raise ValueError(
                "Utility bypass requires a positive supervised residual and labels"
            )
        supervised = (
            coefficient
            * task_loss(
                logits, support_targets.repeat_interleave(k, 0), meta.config["task"]
            ).mean()
        )
        values = torch.autograd.grad(
            supervised,
            [p for _, p in active],
            create_graph=create_graph,
            retain_graph=True,
            allow_unused=True,
        )
        utility_grads = {
            name: value for (name, _), value in zip(active, values) if value is not None
        }
    grads = torch.autograd.grad(
        loss, [p for _, p in active], create_graph=create_graph, allow_unused=True
    )
    updated = OrderedDict(params)
    guarded_directions = {}
    for (name, weight), grad in zip(active, grads):
        if grad is None:
            continue
        if update_scope == "projected_only" and name not in bases:
            continue
        if update_scope == "last4" and not (
            name.startswith(("model.head.", "model.norm."))
            or any(name.startswith(f"model.blocks.{i}.") for i in range(8, 12))
        ):
            continue
        direction = grad + weight_decay * weight
        selected = (
            projection_scope == "all"
            or (projection_scope == "backbone" and not name.startswith("model.head."))
            or (
                projection_scope == "early"
                and any(name.startswith(f"model.blocks.{i}.") for i in range(8))
            )
        )
        protected = not no_projection and name in bases and selected
        if protected:
            u = bases[name].to(weight.device, dtype=weight.dtype)
            forget_direction = direction - utility_grads.get(name, 0.0)
            direction = direction - projection_strength * ((forget_direction @ u) @ u.T)
        else:
            direction = unprojected_lr_scale * direction
        updated[name] = weight - lr * direction
        if utility_guard != "none":
            guarded_directions[name] = direction
    if utility_guard != "none":
        if utility_gradients is None:
            raise ValueError("Utility guard requires calibrated gradients")
        guarded = guard_utility_directions(
            guarded_directions, utility_gradients, utility_guard, utility_guard_strength
        )
        for name, direction in guarded.items():
            updated[name] = params[name] - lr * direction
    return updated, loss


def task_loss(logits, labels, task):
    if task == "multilabel":
        return F.binary_cross_entropy_with_logits(
            logits, labels.float(), reduction="none"
        ).mean(-1)
    return F.cross_entropy(logits, labels.long(), reduction="none")


def calibrated_loss(logits, labels, task, confidence=True, strength=1.0):
    loss = task_loss(logits, labels, task)
    if not confidence:
        return loss
    if task == "multilabel":
        active = labels.float()
        rho = (logits.sigmoid() * active).sum(-1) / active.sum(-1).clamp_min(1)
    else:
        rho = logits.softmax(-1).gather(1, labels.long()[:, None]).squeeze(1)
    return loss * (1 - strength * rho)


def reference_matching(forget_losses, reference_losses, mode="mean"):

    if mode == "mean":
        return F.smooth_l1_loss(forget_losses.mean(), reference_losses.detach().mean())
    if mode != "pairwise":
        raise ValueError("Unknown matching mode: " + mode)
    difference = forget_losses[:, None] - reference_losses.detach()[None, :]
    return F.smooth_l1_loss(difference, torch.zeros_like(difference))


def loss_distribution_distance(forget_losses, reference_losses):

    x, y = forget_losses.reshape(-1), reference_losses.detach().reshape(-1)
    if x.numel() == 0 or y.numel() == 0:
        raise ValueError("Loss-distribution matching needs nonempty sample sets")

    def kernel(a, b):
        distance = (a[:, None] - b[None, :]).square()
        return torch.stack(
            [
                torch.exp(-distance / (2 * bandwidth**2))
                for bandwidth in (0.02, 0.05, 0.1, 0.2)
            ]
        ).mean(0)

    return (
        kernel(x, x).mean() + kernel(y, y).mean() - 2 * kernel(x, y).mean()
    ).clamp_min(0)


def outer_objective(
    target,
    updated,
    meta,
    forget,
    retain,
    validation,
    anchor,
    task="multilabel",
    lambda_retain=0.5,
    lambda_orth=0.5,
    epsilon=0.1,
    confidence=True,
    match_pre=True,
    match_post=True,
    forget_anchor_indices=None,
    matching_mode="mean",
    calibration_strength=1.0,
    lambda_match=1.0,
    retain_target="teacher",
    lambda_validation=0.0,
    lambda_distribution=0.0,
):
    xf, yf = forget
    xr, yr = retain
    xv, yv = validation
    logits_f, feat_f = functional_call(target, updated, (xf,))
    logits_r, _ = functional_call(target, updated, (xr,))
    if lambda_validation:
        new_v_grad, _ = functional_call(target, updated, (xv,))
        validation_loss = task_loss(new_v_grad, yv, task).mean()
    else:
        new_v_grad = None
        validation_loss = logits_f.sum() * 0
    with torch.no_grad():
        old_r, _ = target(xr)
        old_v, _ = target(xv)
        if new_v_grad is None:
            new_v, _ = functional_call(target, updated, (xv,))
        else:
            new_v = new_v_grad.detach()
        losses_pre = calibrated_loss(old_v, yv, task, confidence, calibration_strength)
        losses_post = calibrated_loss(new_v, yv, task, confidence, calibration_strength)
    losses_f = calibrated_loss(logits_f, yf, task, confidence, calibration_strength)
    stats_f = losses_f.mean()
    matching = stats_f * 0
    if match_pre:
        matching = matching + reference_matching(losses_f, losses_pre, matching_mode)
    if match_post:
        matching = matching + reference_matching(losses_f, losses_post, matching_mode)
    if retain_target == "labels":
        retaining = task_loss(logits_r, yr, task).mean()
    elif task == "multilabel":
        retaining = F.binary_cross_entropy_with_logits(logits_r, old_r.sigmoid())
    else:
        retaining = F.kl_div(
            logits_r.log_softmax(-1), old_r.softmax(-1), reduction="batchmean"
        )
    direction = F.normalize(
        meta.pi_s(F.layer_norm(feat_f, (feat_f.shape[-1],))), dim=-1
    )
    if forget_anchor_indices is not None:
        anchor = anchor[forget_anchor_indices]
    orth = F.relu((direction * anchor).sum(-1) - epsilon).mean()
    loss = (
        lambda_match * matching
        + lambda_retain * retaining
        + lambda_orth * orth
        + lambda_validation * validation_loss
    )
    distribution = stats_f.detach() * 0
    if lambda_distribution:
        distribution = loss_distribution_distance(
            task_loss(logits_f, yf, task), task_loss(old_v, yv, task)
        )
        loss = loss + lambda_distribution * distribution
    return loss, dict(
        distribution=distribution.detach(),
        weighted_distribution=(lambda_distribution * distribution).detach(),
        match=matching.detach(),
        weighted_match=lambda_match * matching.detach(),
        validation=validation_loss.detach(),
        retain=retaining.detach(),
        orth=orth.detach(),
        forget_stat=stats_f.detach(),
        pre_stat=losses_pre.mean(),
        post_stat=losses_post.mean(),
    )
