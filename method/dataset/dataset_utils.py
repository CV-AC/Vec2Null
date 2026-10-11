import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from torchvision.transforms import v2 as transforms
from typing import Tuple

from method.dataset.dataset_classes import CelebA, CelebAHQ, MUFAC, IdentityUnlearningDataset


def get_transforms(args) -> Tuple[transforms.Compose, transforms.Compose]:
    if args.dataset in ("celeba", "celebahq"):
        args.size = 224
        args.mean = (0.5, 0.5, 0.5)
        args.std = (0.5, 0.5, 0.5)
        args.task = "multilabel"

        if "vit" in args.model:
            args.size = 224

        base_transform = [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(args.mean, args.std),
        ]

        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(args.size, antialias=None),
                transforms.RandomHorizontalFlip(),
                *base_transform,
            ]
        )
        test_transform = transforms.Compose(
            [
                *base_transform,
                transforms.Resize((args.size, args.size), antialias=None),
            ]
        )

    elif args.dataset == "mufac":
        args.size = 224
        args.mean = (0.5, 0.5, 0.5)
        args.std = (0.5, 0.5, 0.5)
        args.task = "classification"

        base_transform = [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(args.mean, args.std),
        ]
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(args.size, antialias=None),
                transforms.RandomHorizontalFlip(),
                *base_transform,
            ]
        )

        test_transform = transforms.Compose(
            [
                *base_transform,
                transforms.Resize((args.size, args.size), antialias=None),
            ]
        )

    return train_transform, test_transform


def get_datasets(args, splits: dict) -> IdentityUnlearningDataset:


    train_transform, test_transform = get_transforms(args)

    datasets = {}

    if args.dataset == "celeba":
        train_dataset = CelebA(
            root=args.data_dir,
            split="all",
            target_type=["attr", "identity"],
            transform=train_transform,
            download=False,
        )
        test_dataset = CelebA(
            root=args.data_dir,
            split="all",
            target_type=["attr", "identity"],
            transform=test_transform,
            download=False,
        )

        args.num_classes = train_dataset.get_num_classes()
        args.criterion = "binary_cross_entropy"

        datasets = IdentityUnlearningDataset(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            num_identities=args.num_identities,
            splits=splits,
        )

    elif args.dataset == "celebahq":
        train_dataset = CelebAHQ(args.data_dir, transform=train_transform)
        test_dataset = CelebAHQ(args.data_dir, transform=test_transform)

        args.num_classes = len(train_dataset.classes)

        if args.dataset == "celebahq":
            args.criterion = "binary_cross_entropy"
        else:
            args.criterion = None

        datasets = IdentityUnlearningDataset(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            num_identities=args.num_identities,
            splits=splits,
        )

    elif args.dataset == "mufac":
        train_dataset = MUFAC(args.data_dir, transform=train_transform)
        test_dataset = MUFAC(args.data_dir, transform=test_transform)
        args.num_classes = len(train_dataset.classes)
        args.criterion = "cross_entropy"

        datasets = IdentityUnlearningDataset(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            num_identities=args.num_identities,
            splits=splits,
        )

    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    return datasets


class DistributedEvalSampler(Sampler):


    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0


        self.total_size = len(self.dataset)
        indices = list(range(self.total_size))
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)

        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:

            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))


        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):


        self.epoch = epoch
