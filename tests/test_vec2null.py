import argparse
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from method.dataset.dataset_classes import IdentityUnlearningDataset
from method.metrics import compute_basic_mia
from method.protocol import validate_pair
from method.vec2null.core import (
    StructuredMetaLoss,
    TargetWithFeatures,
    inner_update,
    outer_objective,
)
from method.vec2null.data import MetaEpisodes
from method.vec2null.projection import calibrate_subspaces
from piu.joint_piu import verify
from vec2null import parse_options


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 5)
        self.head = nn.Linear(5, 2)
        self.num_features = 5

    def forward_features(self, x):
        return self.encoder(x.mean((-2, -1)))

    def forward_head(self, x, pre_logits=False):
        return x if pre_logits else self.head(x)

    def get_classifier(self):
        return self.head

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.identities = [identity for identity in range(5) for _ in range(6)]

    def __len__(self):
        return len(self.identities)

    def __getitem__(self, row):
        identity = self.identities[row]
        image = torch.full((3, 8, 8), row / 30)
        return image, identity, torch.tensor([identity % 2, (identity + 1) % 2])

    def get_identities(self):
        return self.identities

    def get_identity_to_images(self):
        return {
            identity: list(range(identity * 6, (identity + 1) * 6))
            for identity in range(5)
        }


def manager():
    dataset = ToyDataset()
    return IdentityUnlearningDataset(
        dataset,
        dataset,
        1,
        dict(
            train=[0, 1, 2], val=[3], test=[4], support=[0], forget=[0], retain=[1, 2]
        ),
    )


class Vec2NullTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_config_loads_weight_and_cli_overrides(self):
        config = Path(__file__).resolve().parents[1] / "configs" / "vec2null.json"
        options = parse_options(
            [
                "--config",
                str(config),
                "meta-train",
                "--target",
                "target.pth",
                "--output",
                "output",
            ]
        )
        self.assertTrue(Path(options.unoface).is_file())
        self.assertEqual(options.deployment_preprocessing, "target")
        options = parse_options(
            [
                "--config",
                str(config),
                "meta-train",
                "--target",
                "target.pth",
                "--output",
                "output",
                "--unoface",
                "override.pth",
            ]
        )
        self.assertEqual(options.unoface, "override.pth")

    def test_episode_is_deterministic_and_uses_train_and_val_only(self):
        episodes = MetaEpisodes(
            manager(), steps=3, identities=1, views=2, batch=2, seed=3
        )
        self.assertEqual(
            {
                episodes.row_identity[row]
                for _, req in episodes.requests
                for row in req["support"].indices
            },
            {0, 1, 2},
        )
        for train, request in episodes.requests:
            support = set(request["support"].indices)
            forget = set(request["forget"].indices)
            retain = set(train["retain"].indices)
            self.assertFalse(support & forget or support & retain or forget & retain)
            self.assertFalse((support | forget | retain) & {0})
            self.assertTrue(
                all(
                    episodes.row_identity[row] in {0, 1, 2}
                    for row in support | forget | retain
                )
            )
        first = episodes[0]
        torch.rand(100)
        second = episodes[0]
        self.assertTrue(torch.equal(first["support"], second["support"]))
        self.assertTrue(
            all(episodes.row_identity[row] == 3 for row in episodes.reference_rows)
        )

    def test_classification_and_multilabel_have_finite_second_derivatives(self):
        for task in ("multilabel", "classification"):
            target = TargetWithFeatures(ToyModel()).eval()
            meta = StructuredMetaLoss(2, 5, 4, 3, 6, task=task, num_identity_tokens=3)
            views = torch.randn(1, 2, 3, 8, 8)
            anchor = F.normalize(torch.randn(1, 4), dim=-1)
            context = torch.zeros(1, 3)
            label = (
                torch.tensor([[1, 0]]) if task == "multilabel" else torch.tensor([1])
            )
            updated, loss = inner_update(
                target,
                meta,
                views,
                anchor,
                context,
                {},
                support_targets=label,
                support_ids=torch.tensor([0]),
            )
            y = label.repeat(2, 1) if task == "multilabel" else label.repeat(2)
            pair = (torch.randn(2, 3, 8, 8), y)
            objective, _ = outer_objective(
                target,
                updated,
                meta,
                pair,
                pair,
                pair,
                anchor,
                task=task,
                forget_anchor_indices=torch.zeros(2, dtype=torch.long),
            )
            gradients = torch.autograd.grad(
                objective, tuple(meta.parameters()), allow_unused=True
            )
            self.assertTrue(torch.isfinite(loss) and torch.isfinite(objective))
            self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in gradients))
            self.assertTrue(
                all(torch.isfinite(g).all() for g in gradients if g is not None)
            )

    def test_request_rejects_support_forget_overlap(self):
        datasets = manager()
        groups = datasets.test_dataset.get_identity_to_images()
        groups[0] = [0, 1]
        with patch.object(
            datasets.test_dataset, "get_identity_to_images", return_value=groups
        ):
            with self.assertRaisesRegex(ValueError, "separate support and forget"):
                datasets.get_simulated_unlearning_data(num_ids=1)

    def test_projection_preserves_calibrated_activations(self):
        target = TargetWithFeatures(ToyModel()).eval()
        batches = [(torch.randn(2, 3, 8, 8), torch.zeros(2), torch.zeros(2, 2))]
        bases, _ = calibrate_subspaces(
            target, batches, "cpu", energy=1, max_rows=4, max_batches=1
        )
        for name, basis in bases.items():
            weight = dict(target.named_parameters())[name]
            direction = torch.randn_like(weight)
            projected = direction - (direction @ basis) @ basis.T
            self.assertTrue(
                torch.allclose(
                    projected @ basis, torch.zeros_like(direction @ basis), atol=2e-6
                )
            )

    def test_protocol_rejects_mismatched_support(self):
        args = argparse.Namespace(
            dataset="celebahq", num_identities=1, model="toy", seed=0, num_classes=2
        )
        first = dict(
            args=args,
            state_dict={"weight": torch.zeros(2, 3)},
            splits=manager().get_splits(),
        )
        second = dict(first, splits=dict(first["splits"], support=[1]))
        with self.assertRaises(ValueError):
            validate_pair(first, second)

    def test_mia_attacker_is_fixed_before_evaluation(self):
        class Forest:
            def __init__(self, **options):
                self.options = options

            def fit(self, x, y):
                self.train_x = x

            def predict_proba(self, x):
                return np.tile([0.5, 0.5], (len(x), 1))

            def predict(self, x):
                return np.zeros(len(x))

        with patch(
            "method.metrics.RandomForestClassifier", side_effect=Forest
        ) as factory:
            compute_basic_mia(
                torch.ones(3), torch.ones(2) * 9, torch.zeros(3), torch.ones(2) * 7
            )
            factory.assert_called_once_with(
                n_estimators=100, criterion="gini", n_jobs=1, random_state=0
            )

    def test_piu_geometry_and_second_derivatives(self):
        self.assertTrue(all(verify("cpu").values()))


if __name__ == "__main__":
    unittest.main()
