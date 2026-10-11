import csv
import os
import random
import torch
import torchvision
from PIL import Image
from typing import Optional


class IdentityUnlearningDataset:
    def __init__(
        self,
        train_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        num_identities: int,
        splits: Optional[dict] = None,
    ):

        super().__init__()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.num_identities = num_identities
        self.num_all_identities = len(self.train_dataset.get_identity_to_images())

        self.TRAIN = []
        self.VAL = []
        self.TEST = []

        self.support_set = []
        self.FORGET = []
        self.RETAIN = []

        self.simulated_support_set = []

        if splits is None:
            self._split_data()
            self.RETAIN, self.FORGET, self.support_set = self._init_unlearning(
                self.TRAIN
            )
        else:
            self.TRAIN = splits["train"]
            self.VAL = splits["val"]
            self.TEST = splits["test"]
            self.support_set = (
                splits["support"] if "support" in splits else splits["support_set"]
            )
            self.FORGET = splits["forget"]
            self.RETAIN = splits["retain"]

        train, val, test = set(self.TRAIN), set(self.VAL), set(self.TEST)
        forget, retain = set(self.FORGET), set(self.RETAIN)
        if train & test or train & val or test & val:
            raise ValueError("TRAIN, VAL and TEST identities must be disjoint")
        if forget & retain or forget | retain != train:
            raise ValueError("FORGET and RETAIN must partition TRAIN identities")
        if (
            len(forget) != self.num_identities
            or len(self.support_set) != self.num_identities
        ):
            raise ValueError(
                "The protocol requires one support image per requested identity"
            )
        identities = self.test_dataset.get_identities()
        if len(set(self.support_set)) != len(self.support_set):
            raise ValueError("Duplicate support images in the checkpoint protocol")
        if any(row < 0 or row >= len(identities) for row in self.support_set):
            raise ValueError("Support image index is outside the dataset")
        if {int(identities[row]) for row in self.support_set} != forget:
            raise ValueError("Support images do not correspond to FORGET identities")
        groups = self.test_dataset.get_identity_to_images()
        if any(not groups[identity] for identity in train | val | test):
            raise ValueError(
                "The protocol contains identities without images; check the upstream annotations"
            )

    def get_splits(self):
        return {
            "train": self.TRAIN,
            "val": self.VAL,
            "test": self.TEST,
            "support": self.support_set,
            "forget": self.FORGET,
            "retain": self.RETAIN,
        }

    def get_num_classes(self):
        return self.train_dataset.get_num_classes()

    def get_data(self, identities, train):

        dataset = self.train_dataset if train else self.test_dataset
        samples = []

        set_identities = set(identities)
        set_support_set = set(self.support_set)

        for img_id, identity in enumerate(dataset.get_identities()):
            if identity in set_identities and not img_id in set_support_set:
                samples.append(img_id)
        return torch.utils.data.Subset(dataset, samples)

    def get_train_data(self, train=True):

        return self.get_data(self.TRAIN, train)

    def get_simulated_unlearning_data(self, num_ids=None):

        set_support_set = set(self.support_set)
        identity2images = self.test_dataset.get_identity_to_images()

        if isinstance(identity2images, dict):
            identity2images = {
                identity: images
                for identity, images in identity2images.items()
                if identity in self.TRAIN
            }
        else:
            identity2images = {
                identity: images
                for identity, images in enumerate(identity2images)
                if identity in self.TRAIN
            }

        all_train_identities = set(self.TRAIN)

        import copy

        identities = copy.deepcopy(self.TRAIN)
        random.shuffle(identities)

        datasets = []
        self.simulated_support_set = {}

        num_identities = self.num_identities if num_ids is None else num_ids

        for i in range(0, len(identities), num_identities):
            support_set_split = []
            forget_set_split = []
            retain_set_split = []

            if i + num_identities < len(identities):
                ids = identities[i : i + num_identities]
            else:
                ids = (
                    identities[i:] + identities[: i + num_identities - len(identities)]
                )

            retain_ids = all_train_identities - set(ids)

            for identity in ids:
                set_images = set()
                set_images.update(identity2images[identity])
                set_images -= set_support_set
                valid_images = list(set_images)
                if len(valid_images) < 2:
                    raise ValueError(
                        f"Identity {identity} needs separate support and forget images after excluding the protocol support"
                    )

                image = random.sample(valid_images, 1)[0]
                support_set_split.append(image)
                self.simulated_support_set[identity] = image

                valid_images.remove(image)
                forget_set_split.extend(valid_images)

            for identity in retain_ids:
                retain_images = identity2images[identity]
                retain_images = list(set(retain_images) - set_support_set)
                retain_set_split.extend(retain_images)

            datasets.append(
                [
                    {
                        "support": torch.utils.data.Subset(
                            self.test_dataset, support_set_split
                        ),
                        "forget": torch.utils.data.Subset(
                            self.train_dataset, forget_set_split
                        ),
                        "retain": torch.utils.data.Subset(
                            self.train_dataset, retain_set_split
                        ),
                    },
                    {
                        "support": torch.utils.data.Subset(
                            self.test_dataset, support_set_split
                        ),
                        "forget": torch.utils.data.Subset(
                            self.test_dataset, forget_set_split
                        ),
                        "retain": torch.utils.data.Subset(
                            self.test_dataset, retain_set_split
                        ),
                    },
                ]
            )

        return datasets

    def get_val_data(self, train: bool = False) -> torch.utils.data.Subset:

        return self.get_data(self.VAL, train=train)

    def get_test_data(self):

        return self.get_data(self.TEST, train=False)

    def get_unlearning_data(self, train=False):

        datasets = {
            "support": torch.utils.data.Subset(self.test_dataset, self.support_set),
            "forget": self.get_data(self.FORGET, train),
            "retain": self.get_data(self.RETAIN, train),
        }
        return datasets

    def _split_using_target_ratio(self, ratio, identities):

        set_identities = set(identities)
        images = [
            img
            for img, identity in enumerate(self.train_dataset.get_identities())
            if identity in set_identities
        ]
        target = len(images) * ratio

        limit = int(len(identities) * ratio)
        test_ids = set(identities[:limit])
        num_test_images = len(
            [
                img
                for img, identity in enumerate(self.train_dataset.get_identities())
                if identity in test_ids
            ]
        )

        last_side = -1 if num_test_images > target else +1

        while (num_test_images > target and last_side == -1) or (
            num_test_images <= target and last_side == +1
        ):
            limit += last_side
            num_test_images += last_side * len(
                self.train_dataset.get_identity_to_images()[identities[limit - 1]]
            )

        print(
            f"Num images: {len(images)}({len(identities)}) - Ratio: {ratio} - Num test images: {num_test_images} - Target: {target}"
        )
        return identities[:limit], identities[limit:]

    def _split_data(self):

        if isinstance(self.train_dataset.get_identity_to_images(), dict):
            shuffled_identities = list(
                self.train_dataset.get_identity_to_images().keys()
            )
        else:
            shuffled_identities = list(
                range(len(self.train_dataset.get_identity_to_images()))
            )
        random.shuffle(shuffled_identities)

        intermediate, self.TRAIN = self._split_using_target_ratio(
            ratio=0.4,
            identities=shuffled_identities,
        )
        self.VAL, self.TEST = self._split_using_target_ratio(
            ratio=0.5,
            identities=intermediate,
        )

    def _init_unlearning(self, train_ids):

        num_identities_so_far = 0

        identity2images = self.train_dataset.get_identity_to_images()
        if isinstance(identity2images, dict):
            identities = list(identity2images.keys())
        else:
            identities = list(range(len(identity2images)))
        random.shuffle(identities)

        RETAIN = []
        FORGET = []
        support_set = []

        for identity in identities:
            images = identity2images[identity]
            if identity in train_ids:
                if (
                    len(images) > 1
                    and num_identities_so_far < self.num_identities
                    and (
                        (self.num_identities == 1 and len(images) > 5)
                        or self.num_identities > 1
                    )
                ):
                    image = random.sample(images, 1)[0]
                    support_set.append(image)
                    FORGET.append(identity)
                    num_identities_so_far += 1
                else:
                    RETAIN.append(identity)

        return RETAIN, FORGET, support_set


class IdentityDataset(torch.utils.data.Dataset):
    def get_num_classes(self):
        raise NotImplementedError

    def get_identities(self):
        raise NotImplementedError

    def get_identity_to_images(self):
        raise NotImplementedError


class MUFAC(IdentityDataset):
    def __init__(self, root, transform=None):
        super().__init__()

        self.root = root
        self.transform = transform

        self.fpath = os.path.join(self.root, "mufac")

        self.data = []
        self.identities = []
        self.ages = []
        self.age_map = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}
        self.identity_to_images = []
        self.classes = [
            "0-6 years old",
            "7-12 years old",
            "13-19 years old",
            "20-30 years old",
            "31-45 years old",
            "46-55 years old",
            "56-66 years old",
            "67-80 years old",
        ]

        self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        identity = self.identities[index]
        age = self.ages[index]
        return img, identity, age

    def get_num_classes(self):
        return len(self.classes)

    def get_identities(self):
        return self.identities

    def get_identity_to_images(self):
        return self.identity_to_images

    def _load_data(self):

        seen_ids = {}
        curr_id_num = 0

        for split in ("train", "val", "test"):
            annotation_file = f"custom_{split}_dataset.csv"
            annotation_path = os.path.join(self.fpath, annotation_file)

            if os.environ.get("MUFAC_ANNOTATIONS"):
                annotation_path = os.path.join(
                    os.environ["MUFAC_ANNOTATIONS"], annotation_file
                )

            with open(annotation_path, "r") as fp:
                reader = csv.reader(fp)
                for i, row in enumerate(reader):
                    if i > 0:
                        id_string = "_".join(row[:2])
                        if id_string not in seen_ids:
                            seen_ids[id_string] = curr_id_num
                            curr_id_num += 1

                        self.data.append(
                            os.path.join(self.fpath, f"{split}_images", row[-1])
                        )
                        self.identities.append(seen_ids[id_string])
                        self.ages.append(self.age_map[row[2]])

        self.identity_to_images = [[] for _ in range(max(self.identities) + 1)]
        for i, identity in enumerate(self.identities):
            self.identity_to_images[identity].append(i)


class CelebA(IdentityDataset, torchvision.datasets.CelebA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.identity_to_images = {}
        for i, identity in enumerate(self.get_identities()):
            if identity not in self.identity_to_images:
                self.identity_to_images[identity] = []
            self.identity_to_images[identity].append(i)

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        return image, target[1], target[0]

    def get_num_classes(self):
        return self.attr.size(1)

    def get_identities(self):
        return self.identity.squeeze().tolist()

    def get_identity_to_images(self):
        return self.identity_to_images


class CelebAHQ(IdentityDataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform

        self.fpath = os.path.join(self.root, "CelebAMask-HQ")
        self.image_folder = os.path.join(self.fpath, "CelebA-HQ-img")
        self.identity_file = os.path.join(self.fpath, "CelebA-HQ-identity.txt")
        self.attribute_file = os.path.join(
            self.fpath, "CelebAMask-HQ-attribute-anno.txt"
        )

        self.data = []
        self.identities = []
        self.attributes = []
        self.identity_to_images = []
        self.classes = None

        self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        identity = self.identities[index]
        attributes = torch.nn.functional.one_hot(
            torch.LongTensor(self.attributes[index]), num_classes=len(self.classes)
        ).sum(dim=0)
        return img, identity, attributes

    def get_num_classes(self):
        return len(self.classes)

    def get_identities(self):
        return self.identities

    def get_identity_to_images(self):
        return self.identity_to_images

    def _load_data(self):
        with open(self.attribute_file, "r") as f:
            self.num_images = int(f.readline().strip())
            self.data = [None] * self.num_images
            self.attributes = [None] * self.num_images

            self.classes = f.readline().strip().split()
            for line in f:
                entry = line.strip().split()
                img_id = int(entry[0].split(".")[0])

                img_path = os.path.join(self.image_folder, entry[0])
                self.data[img_id] = img_path
                self.attributes[img_id] = [
                    i for i, attr in enumerate(entry[1:]) if int(attr) != -1
                ]

        self.identities = [-1] * self.num_images
        with open(self.identity_file, "r") as f:
            for line in f:
                entry = line.strip().split()
                img_id = int(entry[0].split(".")[0])
                person_id = int(entry[1])

                self.identities[img_id] = person_id

        self.identity_to_images = [[] for _ in range(max(self.identities) + 1)]
        for i, identity in enumerate(self.identities):
            self.identity_to_images[identity].append(i)
