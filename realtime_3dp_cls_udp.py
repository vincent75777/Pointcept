import argparse
import os
import socket
import struct
import time
from pathlib import Path

import numpy as np
import torch

from pointcept.datasets.transform import Compose
from pointcept.engines.defaults import default_config_parser
from pointcept.models.builder import build_model


UDP_IP_DEFAULT = "0.0.0.0"
UDP_PORT_DEFAULT = 8888
OFFSET_ID = 11
OFFSET_COUNT = 15
OFFSET_DATA = 17
BYTES_PER_POINT = 12

CLASS_NAMES = ["marche", "accroupi", "escalade"]


def build_parser():
    parser = argparse.ArgumentParser(description="Realtime 3DP UDP classifier.")
    parser.add_argument(
        "--config",
        default="configs/3dp/cls-3dp-ptv3-v1m1-0-run.py",
        help="Config path (relative to repo root).",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Checkpoint path (relative to repo root).",
    )
    parser.add_argument("--ip", default=UDP_IP_DEFAULT)
    parser.add_argument("--port", type=int, default=UDP_PORT_DEFAULT)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--test-npz", default=None, help="Run inference on one .npz and exit.")
    return parser


def repo_root():
    return Path(__file__).resolve().parent


def resolve_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path
    return repo_root() / path


def default_ckpt_path():
    return repo_root() / "exp/3dp/cls-ptv3/model/model_last.pth"


def get_infer_transform():
    return Compose(
        [
            dict(type="NormalizeCoord"),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord"),
                feat_keys=["coord"],
            ),
        ]
    )


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


def prepare_input(coord, infer_tf, device, num_points):
    if num_points is not None and len(coord) > num_points:
        choice = np.random.choice(len(coord), num_points, replace=False)
        coord = coord[choice]
    data = {"coord": coord}
    data = infer_tf(data)
    for key, value in data.items():
        if torch.is_tensor(value):
            data[key] = value.to(device)
    return data


def load_model(cfg_path, ckpt_path, device):
    cfg = default_config_parser(str(cfg_path), options={})
    model = build_model(cfg.model).to(device)
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def infer(model, data):
    with torch.no_grad():
        output = model(data)
        logits = output.get("cls_logits", list(output.values())[0]) if isinstance(output, dict) else output
        probs = torch.softmax(logits, dim=-1)[0]
        pred = torch.argmax(probs).item()
        conf = probs[pred].item()
    return pred, conf


def print_pred(pred, conf, cluster_id=None):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cluster_info = f"cluster={cluster_id}" if cluster_id is not None else "cluster=unknown"
    print(f"{timestamp} | {cluster_info} | pred={CLASS_NAMES[pred]} | conf={conf:.2%}")


def run_test_npz(model, infer_tf, device, path, num_points):
    coord = load_coord(path)
    data = prepare_input(coord, infer_tf, device, num_points)
    pred, conf = infer(model, data)
    print_pred(pred, conf, cluster_id=Path(path).stem)


def run_udp(model, infer_tf, device, ip, port, num_points):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip, port))
    sock.settimeout(1.0)
    print(f"📡 Listening UDP on {ip}:{port}")

    storage = {}
    try:
        while True:
            try:
                payload, _ = sock.recvfrom(65535)
            except socket.timeout:
                continue

            if not payload.startswith(b"\x28\x2a"):
                continue

            cluster_id = struct.unpack_from("<H", payload, OFFSET_ID)[0]
            num_pts_packet = struct.unpack_from("<H", payload, OFFSET_COUNT)[0]

            points = []
            for i in range(num_pts_packet):
                idx = OFFSET_DATA + (i * BYTES_PER_POINT)
                if idx + BYTES_PER_POINT <= len(payload):
                    points.append(struct.unpack_from("<3f", payload, idx))

            if cluster_id not in storage:
                storage[cluster_id] = []
            storage[cluster_id].extend(points)

            if len(storage[cluster_id]) >= num_points:
                coord = np.array(storage[cluster_id], dtype=np.float32)
                data = prepare_input(coord, infer_tf, device, num_points)
                pred, conf = infer(model, data)
                print_pred(pred, conf, cluster_id=cluster_id)
                storage[cluster_id] = []
    finally:
        sock.close()


def main():
    parser = build_parser()
    args = parser.parse_args()

    cfg_path = resolve_path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    if args.ckpt is None:
        ckpt_path = default_ckpt_path()
        if not ckpt_path.exists():
            print("Checkpoint not found and --ckpt not provided.")
            parser.print_help()
            return
    else:
        ckpt_path = resolve_path(args.ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Use --device cpu.")

    device = torch.device(args.device)
    model = load_model(cfg_path, ckpt_path, device)
    infer_tf = get_infer_transform()

    if args.test_npz:
        run_test_npz(model, infer_tf, device, resolve_path(args.test_npz), args.num_points)
        return

    run_udp(model, infer_tf, device, args.ip, args.port, args.num_points)


if __name__ == "__main__":
    main()
