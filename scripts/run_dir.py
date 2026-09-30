import os
import sys
import argparse
import logging
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{code_dir}/../")

from core.utils.utils import InputPadder
from core.foundation_stereo import FoundationStereo
from Utils import set_logging_format, set_seed


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def read_rgb(path):
    image = imageio.imread(path)

    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[..., :3]

    return image


def predict_depth(model, left_path, right_path, fx, baseline, args):
    left = read_rgb(left_path)
    right = read_rgb(right_path)

    if left.shape[:2] != right.shape[:2]:
        raise ValueError(f"Image size mismatch: {left_path} {left.shape} vs {right_path} {right.shape}")

    scale = args.scale
    if scale != 1:
        left = cv2.resize(left, None, fx=scale, fy=scale)
        right = cv2.resize(right, None, fx=scale, fy=scale)

    height, width = left.shape[:2]

    left = torch.from_numpy(left).cuda().float().permute(2, 0, 1)[None]
    right = torch.from_numpy(right).cuda().float().permute(2, 0, 1)[None]

    padder = InputPadder(left.shape, divis_by=32, force_square=False)
    left, right = padder.pad(left, right)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        if args.hiera:
            disparity = model.run_hierachical(left, right, iters=args.valid_iters, test_mode=True, small_ratio=0.5)
        else:
            disparity = model(left, right, iters=args.valid_iters, test_mode=True)

    disparity = padder.unpad(disparity.float())
    disparity = disparity.cpu().numpy().reshape(height, width)

    scaled_fx = fx * scale
    depth = np.zeros_like(disparity, dtype=np.float32)

    valid = disparity > 0
    if args.remove_invisible:
        x = np.arange(width)[None, :]
        valid &= (x - disparity) >= 0

    depth[valid] = scaled_fx * baseline / disparity[valid]
    return depth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left_dir", required=True)
    parser.add_argument("--right_dir", required=True)
    parser.add_argument("--intrinsic_file", required=True) # npy file containing K and baseline
    parser.add_argument("--ckpt_dir", default=f"{code_dir}/../pretrained_models/23-51-11/model_best_bp2.pth")
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--hiera", type=int, default=0)
    parser.add_argument("--valid_iters", type=int, default=32)
    parser.add_argument("--remove_invisible", type=int, default=0)
    parser.add_argument("--to_png", action="store_true", help="Save depth as PNG instead of NPY")
    args = parser.parse_args()

    assert 0 < args.scale <= 1

    set_logging_format()
    set_seed(0)
    torch.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = OmegaConf.load(
        os.path.join(os.path.dirname(args.ckpt_dir), "cfg.yaml")
    )
    if "vit_size" not in cfg:
        cfg.vit_size = "vitl"

    for key, value in vars(args).items():
        cfg[key] = value

    model = FoundationStereo(cfg).cuda().eval()

    checkpoint = torch.load(args.ckpt_dir, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    with open(args.intrinsic_file, "rb") as f:
        K = np.load(f)
        baseline = np.load(f)

    fx = float(K[0, 0])

    left_dir = Path(args.left_dir)
    right_dir = Path(args.right_dir)

    left_files = sorted(left_dir.glob("*"))
    right_files = sorted(right_dir.glob("*"))

    assert len(left_files) == len(right_files), "Number of left and right images must be the same"

    for (left_path, right_path) in tqdm(zip(left_files, right_files)):
        depth = predict_depth(model, left_path, right_path, fx, baseline, cfg)

        if args.to_png:
            output_path = Path(args.out_dir) / f"{left_path.stem}.png"
            depth = depth.round().astype(np.uint8)
            Image.fromarray(depth).save(output_path)
        else:
            output_path = Path(args.out_dir) / f"{left_path.stem}.npy"
            np.save(output_path, depth)


if __name__ == "__main__":
    main()