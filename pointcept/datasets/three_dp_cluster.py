import csv
import os
from copy import deepcopy

import numpy as np
from torch.utils.data import Dataset

from pointcept.utils.logger import get_root_logger
from .builder import DATASETS
from .transform import Compose


@DATASETS.register_module()
class ThreeDPClusterDataset(Dataset):
    def __init__(
        self,
        split="train",
        data_root="data/3dp_clusters_processed",
        class_names=None,
        transform=None,
        num_points=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.class_names = class_names or ["marche", "accroupi", "escalade"]
        self.transform = Compose(transform)
        self.num_points = num_points
        self.loop = loop if not test_mode else 1
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None
        if test_mode and self.test_cfg is not None:
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        self.labels = self._load_labels()
        self.data_list = sorted(self.labels.keys())

        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} set.".format(
                len(self.data_list), self.loop, split
            )
        )

    def _labels_path(self):
        split_name = "val" if self.split in {"val", "test"} else "train"
        return os.path.join(self.data_root, f"labels_{split_name}.csv")

    def _load_labels(self):
        labels_path = self._labels_path()
        if not os.path.exists(labels_path):
            raise FileNotFoundError(
                f"Missing labels file: {labels_path}. Run data/prepare_dataset.py."
            )
        labels = {}
        with open(labels_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "file" not in reader.fieldnames or "label" not in reader.fieldnames:
                raise ValueError(
                    f"labels file must contain header 'file,label': {labels_path}"
                )
            for row in reader:
                labels[row["file"]] = int(row["label"])
        return labels

    def _split_dir(self):
        return "val" if self.split in {"val", "test"} else "train"

    def _load_npz(self, npz_path):
        data = np.load(npz_path)
        if "coord" in data:
            coord = data["coord"]
        elif "points" in data:
            coord = data["points"]
        elif "xyz" in data:
            coord = data["xyz"]
        else:
            raise KeyError(f"Missing coord/points/xyz in {npz_path}")
        return coord.astype(np.float32)

    def get_data(self, idx):
        data_idx = idx % len(self.data_list)
        basename = self.data_list[data_idx]
        npz_path = os.path.join(self.data_root, self._split_dir(), basename)
        coord = self._load_npz(npz_path)
        if self.num_points is not None and len(coord) > self.num_points:
            choice = np.random.choice(len(coord), self.num_points, replace=False)
            coord = coord[choice]
        category = np.array([self.labels[basename]], dtype=np.int64)
        return dict(coord=coord, category=category, name=basename)

    def get_data_name(self, idx):
        data_idx = idx % len(self.data_list)
        return self.data_list[data_idx]

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop

    def prepare_train_data(self, idx):
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        assert idx < len(self.data_list)
        data_dict = self.get_data(idx)
        category = data_dict.pop("category")
        data_dict = self.transform(data_dict)
        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))
        for i in range(len(data_dict_list)):
            data_dict_list[i] = self.post_transform(data_dict_list[i])
        return dict(
            voting_list=data_dict_list,
            category=category,
            name=self.get_data_name(idx),
        )
