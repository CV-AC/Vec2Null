import random
from collections import OrderedDict
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


def support_views(image, k, reference_image=None):
    augment = T.Compose(
        [
            T.RandomResizedCrop(image.shape[-1], scale=(0.8, 1.0), antialias=True),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.0))], p=0.2),
        ]
    )
    import os

    mode = os.environ.get("VEC2NULL_AUGMENT", "original")
    if mode == "none":
        augment = T.Compose([])
    elif mode == "weak":
        augment = T.Compose([T.RandomHorizontalFlip()])
    elif mode != "original":
        raise ValueError(mode)
    native_reference = reference_image
    if reference_image is not None:
        reference_image = T.functional.resize(
            reference_image, list(image.shape[-2:]), antialias=True
        )

        image = torch.stack([image, reference_image])
    image = ((image + 1) / 2).clamp(0, 1)
    views = torch.stack([augment(image) * 2 - 1 for _ in range(k)])
    if native_reference is not None:
        reference_mode = os.environ.get("VEC2NULL_REFERENCE_VIEWS", "paired")
        if reference_mode in ("clean", "flip"):
            refs = torch.stack(
                [
                    native_reference.flip(-1)
                    if reference_mode == "flip" and i % 2
                    else native_reference
                    for i in range(k)
                ]
            )
            return views[:, 0], refs
    return (views[:, 0], views[:, 1]) if reference_image is not None else views


class MetaEpisodes(Dataset):
    def __init__(
        self,
        manager,
        steps=1000,
        identities=20,
        views=1,
        batch=64,
        seed=0,
        reference_alignment=None,
        excluded_reference_paths=(),
    ):
        if not 1 <= identities < len(manager.TRAIN):
            raise ValueError("Episode identity count must leave a nonempty retain set")
        if min(steps, views, batch) < 1:
            raise ValueError("Episode, view and batch counts must be positive")
        if excluded_reference_paths:
            raise ValueError("The protocol does not permit excluding reference images")
        self.manager = manager
        self.dataset = manager.test_dataset
        self.steps = steps
        self.identities = identities
        self.views = views
        self.batch = batch
        self.seed = seed
        self.reference_alignment = reference_alignment
        self.reference_cache = OrderedDict()
        self.development = list(manager.TRAIN)
        self.reference_ids = list(manager.VAL)
        self.identity_map = {
            int(identity): i for i, identity in enumerate(manager.TRAIN)
        }
        groups = self.dataset.get_identity_to_images()
        if not isinstance(groups, dict):
            groups = dict(enumerate(groups))
        self.row_identity = {
            row: int(identity) for identity, rows in groups.items() for row in rows
        }
        self.reference_rows = list(manager.get_val_data().indices)
        actual_support = set(manager.support_set)
        self.retain_rows = [
            row
            for identity in manager.TRAIN
            for row in groups[identity]
            if row not in actual_support
        ]
        self.excluded_reference_rows = set()
        self.requests = []
        state = random.getstate()
        try:
            random.seed(seed)
            while len(self.requests) < steps:
                manager.get_simulated_unlearning_data(num_ids=identities)
                self.requests.extend(
                    manager.get_simulated_unlearning_data(num_ids=identities)
                )
        finally:
            random.setstate(state)
        self.requests = self.requests[:steps]
        self.eligible = list(manager.TRAIN)

    def __len__(self):
        return self.steps

    def image_path(self, row):
        if hasattr(self.dataset, "data"):
            return self.dataset.data[row]
        return (
            Path(self.dataset.root)
            / self.dataset.base_folder
            / "img_align_celeba"
            / self.dataset.filename[row]
        )

    def reference_image(self, row):
        if row not in self.reference_cache:
            self.reference_cache[row] = self.reference_alignment(self.image_path(row))
            if len(self.reference_cache) > 256:
                self.reference_cache.popitem(last=False)
        self.reference_cache.move_to_end(row)
        return self.reference_cache[row]

    def batch_from_rows(self, rows, train=False):
        dataset = self.manager.train_dataset if train else self.dataset
        items = [dataset[row] for row in rows]
        return torch.stack([x[0] for x in items]), torch.stack(
            [torch.as_tensor(x[2]) for x in items]
        )

    def __getitem__(self, index):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed + index)
            return self._episode(index)

    def _episode(self, index):
        train_request, test_request = self.requests[index]
        support_rows = list(test_request["support"].indices)
        hidden = list(test_request["forget"].indices)
        retain_pool = list(train_request["retain"].indices)
        rng = random.Random(self.seed + index)
        rng.shuffle(hidden)
        rng.shuffle(retain_pool)
        validation = list(self.reference_rows)
        rng.shuffle(validation)
        assert hidden and retain_pool and validation

        count = ((len(hidden) + self.batch - 1) // self.batch) * self.batch
        retain = [retain_pool[i % len(retain_pool)] for i in range(count)]
        validation = [validation[i % len(validation)] for i in range(count)]
        support = []
        refs = []
        labels = []
        ids = [self.row_identity[row] for row in support_rows]
        positions = {identity: i for i, identity in enumerate(ids)}
        for row in support_rows:
            image, identity, label = self.dataset[row]
            if self.reference_alignment is None:
                support.append(support_views(image, self.views))
            else:
                v, rv = support_views(image, self.views, self.reference_image(row))
                support.append(v)
                refs.append(rv)
            labels.append(torch.as_tensor(label))
        episode = dict(
            support=torch.stack(support),
            support_targets=torch.stack(labels),
            support_ids=torch.tensor([self.identity_map[i] for i in ids]),
            forget=self.batch_from_rows(hidden),
            forget_anchor_indices=torch.tensor(
                [positions[self.row_identity[r]] for r in hidden]
            ),
            retain=self.batch_from_rows(retain, train=True),
            validation=self.batch_from_rows(validation),
        )
        if self.reference_alignment is not None:
            episode["reference_support"] = torch.stack(refs)
            episode["reference_retain_mask"] = torch.ones(len(retain), dtype=torch.bool)
            episode["reference_retain"] = torch.stack(
                [self.reference_image(row) for row in retain]
            )
        return episode
