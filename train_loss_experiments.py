"""
Phase 2: Loss function ablation experiments.
Place at: ~/EECE7370/train_loss_experiments.py

These experiments use the BASELINE model (yolov8s.pt) — no attention modules,
no weight transfer issues. Only the loss function changes.

Usage:
  cd ~/EECE7370

  # Quick smoke test:
  python train_loss_experiments.py --loss EIoU --epochs 5 --imgsz 640

  # Full experiments:
  python train_loss_experiments.py --loss EIoU
  python train_loss_experiments.py --loss WIoU
"""

import sys
import os
import argparse
from pathlib import Path

# Fix Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ultralytics"))

from ultralytics import YOLO


COMMON = dict(
    data="VisDrone.yaml",
    epochs=150,
    imgsz=1280,
    batch=-1,
    patience=50,
    cos_lr=True,
    amp=True,
    close_mosaic=10,
    max_det=1000,
    project="runs/detect",
    seed=42,
    device=0,
    save_json=True,
    workers=8,
)


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Loss function experiments")
    parser.add_argument("--loss", required=True, choices=["EIoU", "WIoU"],
                        help="Loss function type")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    args = parser.parse_args()

    # Set the IoU type via environment variable
    # This is read by BboxLoss.__init__ in loss.py
    os.environ["YOLO_IOU_TYPE"] = args.loss

    name = f"baseline_{args.loss.lower()}"
    settings = {**COMMON, "name": name}

    if args.epochs:
        settings["epochs"] = args.epochs
    if args.imgsz:
        settings["imgsz"] = args.imgsz

    print(f"\n{'='*60}")
    print(f"  Phase 2 Experiment: {name}")
    print(f"  Loss function: {args.loss}")
    print(f"  YOLO_IOU_TYPE env var: {os.environ['YOLO_IOU_TYPE']}")
    print(f"  Epochs: {settings['epochs']}  |  Image size: {settings['imgsz']}")
    print(f"{'='*60}\n")

    # Use the standard pretrained model — no YAML, no weight transfer issues
    model = YOLO("yolov8s.pt")

    results = model.train(**settings)

    print(f"\n{'='*60}")
    print(f"  Experiment {name} complete!")
    print(f"  Results: {settings['project']}/{name}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main() 