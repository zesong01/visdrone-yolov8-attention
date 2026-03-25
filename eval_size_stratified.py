"""
Size-stratified evaluation using pycocotools.
Reports APsmall, APmedium, APlarge for each experiment.

Place at: ~/EECE7370/eval_size_stratified.py

Usage:
  cd ~/EECE7370
  rm visdrone_val_coco_gt.json   # delete old GT if it exists
  pip install pycocotools Pillow
  python eval_size_stratified.py
"""

import json
import sys
import os
from pathlib import Path
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ultralytics"))

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent

CANDIDATE_PATHS = [
    PROJECT_ROOT / "datasets" / "VisDrone",
    PROJECT_ROOT / "ultralytics" / "datasets" / "VisDrone",
    Path.home() / "datasets" / "VisDrone",
]

VISDRONE_ROOT = None
for p in CANDIDATE_PATHS:
    if (p / "images" / "val").exists():
        VISDRONE_ROOT = p
        break

RESULTS_BASE = PROJECT_ROOT / "runs" / "detect" / "runs" / "detect"

EXPERIMENTS = {
    "Baseline":  RESULTS_BASE / "baseline_ciou",
    "CBAM":      RESULTS_BASE / "cbam_ciou",
    "SimAM":     RESULTS_BASE / "simam_ciou",
    "CoordAtt":  RESULTS_BASE / "coordatt_ciou",
    "EIoU":      RESULTS_BASE / "baseline_eiou",
    "WIoU":      RESULTS_BASE / "baseline_wiou",
}

CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]

GT_JSON_PATH = PROJECT_ROOT / "visdrone_val_coco_gt.json"


# =============================================================================
# Step 1: Create GT JSON
# =============================================================================

def create_coco_gt(images_dir, labels_dir, output_json):
    print(f"  Creating COCO ground truth JSON...")
    print(f"    Images: {images_dir}")
    print(f"    Labels: {labels_dir}")

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": n} for i, n in enumerate(CLASS_NAMES)]
    }

    img_files = sorted(images_dir.glob("*.jpg"))
    if len(img_files) == 0:
        print(f"\n  ERROR: No .jpg files found in {images_dir}")
        sys.exit(1)

    print(f"  Found {len(img_files)} validation images")

    stem_to_id = {}
    for idx, img_path in enumerate(img_files):
        stem_to_id[img_path.stem] = idx

    ann_id = 0
    for img_path in img_files:
        stem = img_path.stem
        img_id = stem_to_id[stem]
        im = Image.open(img_path)
        w, h = im.size

        coco["images"].append({
            "id": img_id,
            "file_name": img_path.name,
            "stem": stem,
            "width": w,
            "height": h
        })

        label_path = labels_dir / img_path.with_suffix(".txt").name
        if label_path.exists():
            for line in label_path.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls = int(parts[0])
                xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                abs_w = bw * w
                abs_h = bh * h
                abs_x = (xc * w) - abs_w / 2
                abs_y = (yc * h) - abs_h / 2

                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls,  # 0-indexed (0-9)
                    "bbox": [abs_x, abs_y, abs_w, abs_h],
                    "area": abs_w * abs_h,
                    "iscrowd": 0
                })
                ann_id += 1

    mapping_path = output_json.parent / "visdrone_stem_to_id.json"
    with open(mapping_path, "w") as f:
        json.dump(stem_to_id, f)

    with open(output_json, "w") as f:
        json.dump(coco, f)

    print(f"  Total annotations: {ann_id}")
    print(f"  GT saved to: {output_json}\n")
    return stem_to_id


# =============================================================================
# Step 2: Remap predictions (fix image_id AND category_id)
# =============================================================================

def load_and_remap_predictions(pred_json_path, stem_to_id):
    """
    Fix TWO issues in Ultralytics predictions.json:
      1. image_id: filename stem (string) -> integer matching GT
      2. category_id: 1-indexed (1-10) -> 0-indexed (0-9)
    """
    with open(pred_json_path) as f:
        preds = json.load(f)

    if len(preds) == 0:
        return []

    remapped = []
    unmatched = 0
    for p in preds:
        stem = p["image_id"]
        if stem in stem_to_id:
            p_copy = p.copy()
            p_copy["image_id"] = stem_to_id[stem]
            p_copy["category_id"] = p["category_id"] - 1  # FIX: 1-indexed -> 0-indexed
            remapped.append(p_copy)
        else:
            unmatched += 1

    if unmatched > 0:
        print(f"  WARNING: {unmatched} predictions had unmatched image_ids")

    print(f"  Remapped {len(remapped)} / {len(preds)} predictions")
    return remapped


# =============================================================================
# Step 3: Run COCO evaluation
# =============================================================================

def evaluate_experiment(name, exp_dir, coco_gt, stem_to_id):
    pred_json = exp_dir / "predictions.json"

    if not pred_json.exists():
        print(f"  {name}: predictions.json not found — skipping\n")
        return None

    remapped_preds = load_and_remap_predictions(pred_json, stem_to_id)
    if len(remapped_preds) == 0:
        print(f"  {name}: No valid predictions — skipping\n")
        return None

    temp_path = exp_dir / "predictions_remapped.json"
    with open(temp_path, "w") as f:
        json.dump(remapped_preds, f)

    coco_dt = coco_gt.loadRes(str(temp_path))
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()

    print(f"\n  {'='*50}")
    print(f"  {name}")
    print(f"  {'='*50}")
    coco_eval.summarize()

    stats = coco_eval.stats
    return {
        "AP":       stats[0],
        "AP50":     stats[1],
        "AP75":     stats[2],
        "APsmall":  stats[3],
        "APmedium": stats[4],
        "APlarge":  stats[5],
        "AR1":      stats[6],
        "AR10":     stats[7],
        "AR100":    stats[8],
        "ARsmall":  stats[9],
        "ARmedium": stats[10],
        "ARlarge":  stats[11],
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("  VisDrone Size-Stratified Evaluation (pycocotools)")
    print("=" * 60 + "\n")

    if VISDRONE_ROOT is None:
        print("ERROR: Cannot find VisDrone dataset.")
        print("Tried:", [str(p) for p in CANDIDATE_PATHS])
        print("\nRun: find ~ -path '*/VisDrone/images/val' 2>/dev/null")
        sys.exit(1)

    images_dir = VISDRONE_ROOT / "images" / "val"
    labels_dir = VISDRONE_ROOT / "labels" / "val"
    print(f"Dataset found: {VISDRONE_ROOT}\n")

    # Always regenerate GT to ensure correct format
    stem_to_id = create_coco_gt(images_dir, labels_dir, GT_JSON_PATH)

    print("Loading ground truth...")
    coco_gt = COCO(str(GT_JSON_PATH))
    print(f"  {len(coco_gt.dataset['images'])} images, "
          f"{len(coco_gt.dataset['annotations'])} annotations\n")

    # Evaluate each experiment
    all_results = {}
    for name, exp_dir in EXPERIMENTS.items():
        print(f"\n{'─'*60}")
        print(f"Evaluating: {name}")
        print(f"  Directory: {exp_dir}")

        if not exp_dir.exists():
            print(f"  Not found — skipping")
            continue

        results = evaluate_experiment(name, exp_dir, coco_gt, stem_to_id)
        if results:
            all_results[name] = results

    # Print comparison tables
    if all_results:
        print("\n" + "=" * 70)
        print("  COMPARISON TABLE — Size-Stratified AP")
        print("=" * 70)
        print(f"\n{'Model':<12} {'AP':>7} {'AP50':>7} {'AP75':>7} {'APsm':>7} {'APmd':>7} {'APlg':>7}")
        print("-" * 60)
        for name, r in all_results.items():
            print(f"{name:<12} {r['AP']:7.4f} {r['AP50']:7.4f} {r['AP75']:7.4f} "
                  f"{r['APsmall']:7.4f} {r['APmedium']:7.4f} {r['APlarge']:7.4f}")

        print(f"\n{'Model':<12} {'ARsm':>7} {'ARmd':>7} {'ARlg':>7}")
        print("-" * 38)
        for name, r in all_results.items():
            print(f"{name:<12} {r['ARsmall']:7.4f} {r['ARmedium']:7.4f} {r['ARlarge']:7.4f}")

        if "Baseline" in all_results:
            bl = all_results["Baseline"]
            print(f"\n{'Model':<12} {'APsmall':>8} {'Δ':>8}  {'APmed':>8} {'Δ':>8}  {'APlg':>8} {'Δ':>8}")
            print("-" * 72)
            for name, r in all_results.items():
                ds = r["APsmall"] - bl["APsmall"]
                dm = r["APmedium"] - bl["APmedium"]
                dl = r["APlarge"] - bl["APlarge"]
                ss = f"+{ds:.4f}" if ds >= 0 else f"{ds:.4f}"
                sm = f"+{dm:.4f}" if dm >= 0 else f"{dm:.4f}"
                sl = f"+{dl:.4f}" if dl >= 0 else f"{dl:.4f}"
                print(f"{name:<12} {r['APsmall']:8.4f} {ss:>8}  "
                      f"{r['APmedium']:8.4f} {sm:>8}  "
                      f"{r['APlarge']:8.4f} {sl:>8}")

        print()


if __name__ == "__main__":
    main()