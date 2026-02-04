# 3DP Pointcept Classifier (WSL2)

This guide trains and runs a Pointcept classifier for 3DP LiDAR clusters (classes: `marche=0`, `accroupi=1`, `escalade=2`) in WSL2 while pointing to data on the Windows filesystem.

## 1) WSL2 + CUDA checks

```bash
nvidia-smi
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
PY
```

## 2) Conda environment

```bash
./scripts/setup_env.sh
```

## 3) pointops check

```bash
./scripts/check_env.sh
```

## 4) Dataset preparation

> The dataset zip must be on `/mnt/c/...` when running from WSL. Never use `C:\...` paths.

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
python data/prepare_dataset.py --zip-path "/mnt/c/Users/vince/SynologyDrive/7 - Projets internes/3 - 3DP/2 - Modèles IA- classification/New_model/Pointcept/3dp_clusters.zip"
python data/verify_dataset.py
```

This produces:

```
data/3dp_clusters_processed/train/*.npz
data/3dp_clusters_processed/val/*.npz
data/3dp_clusters_processed/labels_train.csv
data/3dp_clusters_processed/labels_val.csv
```

## 5) Training command

```bash
./scripts/run_train.sh
```

## 6) Realtime UDP inference + dry-run

Dry-run with one `.npz`:

```bash
./scripts/run_realtime.sh --test-npz "data/3dp_clusters_processed/train/<sample>.npz"
```

Realtime UDP:

```bash
./scripts/run_realtime.sh --ip 0.0.0.0 --port 8888 --device cuda
```

## 7) Troubleshooting

### Path issues (WSL)
- Always use `/mnt/c/...` paths under WSL.
- Do **not** pass `C:\Users\...` to python scripts.

### `KeyError: 'category'`
- Ensure `labels_train.csv` / `labels_val.csv` exist in `data/3dp_clusters_processed`.
- Verify each file listed in the CSV exists in the split directory.
- Run:

```bash
python data/verify_dataset.py
```

### `CUDA_HOME` / CUDA errors
- Ensure CUDA is available (`nvidia-smi`).
- Reinstall PyTorch with a CUDA build if `torch.cuda.is_available()` is false.
- If necessary:

```bash
export CUDA_HOME=/usr/local/cuda
```
