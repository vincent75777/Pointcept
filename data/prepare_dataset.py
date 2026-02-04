import argparse
import csv
import os
import shutil
import zipfile

import numpy as np
from tqdm import tqdm


DEFAULT_ZIP = (
    "/mnt/c/Users/vince/SynologyDrive/7 - Projets internes/3 - 3DP/"
    "2 - Modèles IA- classification/New_model/Pointcept/3dp_clusters.zip"
)

LABEL_MAP = {"marche": 0, "accroupi": 1, "escalade": 2}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare 3DP cluster dataset.")
    parser.add_argument("--zip-path", default=DEFAULT_ZIP, help="Path to 3dp_clusters.zip")
    parser.add_argument("--data-root", default="data/3dp_clusters_processed")
    parser.add_argument("--raw-root", default="data/3dp_clusters_raw")
    parser.add_argument("--clean", action="store_true", help="Remove existing output first")
    parser.add_argument("--augment", action="store_true", help="Enable train augmentations")
    parser.add_argument("--augment-count", type=int, default=1, help="Augmented copies per train sample")
    return parser.parse_args()


def infer_label(basename):
    prefix = basename.split("__")[0].lower()
    if prefix not in LABEL_MAP:
        raise ValueError(f"Unknown class prefix '{prefix}' in {basename}")
    return LABEL_MAP[prefix]


def infer_split(path):
    lowered = path.lower()
    if "/val/" in lowered or "/valid/" in lowered or "/test/" in lowered:
        return "val"
    if "/train/" in lowered:
        return "train"
    return "train"


def load_coord(npz_path):
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


def apply_augmentations(coord, aug_idx):
    rng = np.random.default_rng(seed=aug_idx)
    angle = rng.uniform(-np.pi, np.pi)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]])
    coord = coord @ rot.T
    scale = rng.uniform(0.9, 1.1)
    coord = coord * scale
    jitter = rng.normal(0.0, 0.005, size=coord.shape)
    return coord + jitter


def main():
    args = parse_args()

    if not os.path.exists(args.zip_path):
        raise FileNotFoundError(f"Zip not found: {args.zip_path}")

    if args.clean:
        shutil.rmtree(args.data_root, ignore_errors=True)
        shutil.rmtree(args.raw_root, ignore_errors=True)

    os.makedirs(args.raw_root, exist_ok=True)
    os.makedirs(args.data_root, exist_ok=True)
    for split in ("train", "val"):
        os.makedirs(os.path.join(args.data_root, split), exist_ok=True)

    if not os.listdir(args.raw_root):
        with zipfile.ZipFile(args.zip_path, "r") as zip_ref:
            zip_ref.extractall(args.raw_root)

    npz_paths = []
    for root, _, files in os.walk(args.raw_root):
        for name in files:
            if name.endswith(".npz"):
                npz_paths.append(os.path.join(root, name))

    if not npz_paths:
        raise RuntimeError(f"No .npz files found under {args.raw_root}")

    labels = {"train": [], "val": []}
    for npz_path in tqdm(npz_paths, desc="Processing"):
        basename = os.path.basename(npz_path)
        split = infer_split(npz_path)
        coord = load_coord(npz_path)
        dest_path = os.path.join(args.data_root, split, basename)
        np.savez_compressed(dest_path, coord=coord)
        labels[split].append((basename, infer_label(basename)))

        if split == "train" and args.augment:
            for idx in range(args.augment_count):
                coord_aug = apply_augmentations(coord, idx)
                aug_name = f"{os.path.splitext(basename)[0]}_aug{idx}.npz"
                aug_path = os.path.join(args.data_root, split, aug_name)
                np.savez_compressed(aug_path, coord=coord_aug)
                labels[split].append((aug_name, infer_label(basename)))

    for split in ("train", "val"):
        labels_path = os.path.join(args.data_root, f"labels_{split}.csv")
        with open(labels_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["file", "label"])
            for row in sorted(labels[split]):
                writer.writerow(row)

    print("✅ Dataset prepared.")
    for split in ("train", "val"):
        print(f"{split}: {len(labels[split])} samples")


if __name__ == "__main__":
    main()
