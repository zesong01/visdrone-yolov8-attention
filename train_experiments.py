"""
Training script for VisDrone ablation experiments.
Place at: ~/EECE7370/train_experiments.py

Usage:
  cd ~/EECE7370

  # Quick smoke test (5 epochs, 640px):
  python train_experiments.py --exp cbam --epochs 5 --imgsz 640

  # Full attention experiments:
  python train_experiments.py --exp cbam
  python train_experiments.py --exp simam
  python train_experiments.py --exp coordatt
"""

import sys
import os
import argparse
from pathlib import Path

# Fix Python path: the ultralytics PACKAGE lives inside the repo folder
# Directory structure: EECE7370/ultralytics/ultralytics/ (package is the inner one)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ultralytics"))

from ultralytics import YOLO


# =============================================================================
# Experiment configurations
# =============================================================================

# Path to model YAML configs
CFG_DIR = Path(__file__).parent / "ultralytics" / "ultralytics" /  "cfg" / "models" / "v8"

EXPERIMENTS = {
    # Phase 1: Attention mechanisms (all with default CIoU loss)
    "cbam": {
        "model": str(CFG_DIR / "yolov8s-cbam.yaml"),
        "name": "cbam_ciou",
    },
    "simam": {
        "model": str(CFG_DIR / "yolov8s-simam.yaml"),
        "name": "simam_ciou",
    },
    "coordatt": {
        "model": str(CFG_DIR / "yolov8s-coordatt.yaml"),
        "name": "coordatt_ciou",
    },
}

# Common training settings
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
    parser = argparse.ArgumentParser(description="VisDrone ablation experiments")
    parser.add_argument("--exp", required=True, choices=EXPERIMENTS.keys(),
                        help="Which experiment to run")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epoch count (e.g., 5 for quick test)")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Override image size (e.g., 640 for quick test)")
    args = parser.parse_args()

    exp = EXPERIMENTS[args.exp]
    settings = {**COMMON, "name": exp["name"]}

    if args.epochs:
        settings["epochs"] = args.epochs
    if args.imgsz:
        settings["imgsz"] = args.imgsz

    print(f"\n{'='*60}")
    print(f"  Experiment: {exp['name']}")
    print(f"  Model config: {exp['model']}")
    print(f"  Epochs: {settings['epochs']}  |  Image size: {settings['imgsz']}")
    print(f"{'='*60}\n")

    # Load custom model architecture from YAML
    model = YOLO(exp["model"])

    # Transfer pretrained YOLOv8s COCO weights where layer shapes match
    pretrained = YOLO("yolov8s.pt")
    model.model.load_state_dict(pretrained.model.state_dict(), strict=False)
    print("  Loaded pretrained YOLOv8s backbone weights (strict=False)")
    print(f"  New attention layers initialized randomly\n")

    # Train
    results = model.train(**settings)

    print(f"\n{'='*60}")
    print(f"  Experiment {exp['name']} complete!")
    print(f"  Results: {settings['project']}/{exp['name']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()