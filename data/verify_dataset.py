import argparse
import csv
import os
from collections import Counter

from pointcept.datasets.three_dp_cluster import ThreeDPClusterDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Verify 3DP cluster dataset.")
    parser.add_argument("--data-root", default="data/3dp_clusters_processed")
    return parser.parse_args()


def load_labels(path):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {row["file"]: int(row["label"]) for row in reader}


def verify_split(data_root, split):
    labels_path = os.path.join(data_root, f"labels_{split}.csv")
    split_dir = os.path.join(data_root, split)
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Missing labels file: {labels_path}")
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    labels = load_labels(labels_path)
    files_in_dir = {name for name in os.listdir(split_dir) if name.endswith(".npz")}

    missing_files = sorted(set(labels.keys()) - files_in_dir)
    extra_files = sorted(files_in_dir - set(labels.keys()))
    if missing_files:
        raise RuntimeError(f"{split} missing files listed in CSV: {missing_files[:5]}")
    if extra_files:
        raise RuntimeError(f"{split} extra files not in CSV: {extra_files[:5]}")

    dist = Counter(labels.values())
    print(f"{split} label distribution: {dict(dist)}")


def verify_sample(data_root):
    dataset = ThreeDPClusterDataset(
        split="train",
        data_root=data_root,
        class_names=["marche", "accroupi", "escalade"],
        transform=[],
    )
    sample = dataset.get_data(0)
    if "coord" not in sample or "category" not in sample:
        raise RuntimeError("Sample missing coord/category keys.")
    print("Sample keys:", list(sample.keys()))


def main():
    args = parse_args()
    for split in ("train", "val"):
        verify_split(args.data_root, split)
    verify_sample(args.data_root)
    print("✅ Dataset verification passed.")


if __name__ == "__main__":
    main()
