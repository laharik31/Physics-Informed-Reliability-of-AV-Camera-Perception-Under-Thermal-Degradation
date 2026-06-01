"""
BDD100K Dataset Setup Script
=============================
Converts the official BDD100K detection labels to YOLO format
and creates the dataset YAML configuration for Ultralytics validation.

Supports TWO label formats (auto-detected):
  Format A — Per-image JSON (Scalabel format):
      labels/100k/val/*.json   (one JSON per image)
  Format B — Consolidated JSON (det_20 format):
      labels/det_20/det_val.json   (single JSON with all frames)

Prerequisites:
  1. Register at https://bdd-data.berkeley.edu/ and download:
     - Images:  bdd100k_images_100k.zip
     - Labels:  bdd100k_labels.zip  (or bdd100k_labels_release.zip)
  2. Extract BOTH zips into data/bdd100k/ so you get:
       data/bdd100k/
         100k/
           val/        ← 10,000 validation JPEGs
         100k/
           val/        ← per-image JSON labels (or det_20/ folder)

  NOTE: The images zip extracts to 100k/val/ and the labels zip also
  extracts to 100k/val/. Extract IMAGES first, then LABELS into the
  same data/bdd100k/ directory — they will merge into the same folder
  since images are .jpg and labels are .json.

  Alternatively, extract them separately and this script will find them.

Usage:
  python3 python/setup_bdd100k.py
  python3 python/setup_bdd100k.py --data-root data/bdd100k
"""

import json
import os
import yaml
import argparse
from pathlib import Path

# ── BDD100K detection categories → YOLO class IDs ──────────────────────
BDD_CLASSES = [
    "pedestrian",       # 0
    "rider",            # 1
    "car",              # 2
    "truck",            # 3
    "bus",              # 4
    "train",            # 5
    "motorcycle",       # 6
    "bicycle",          # 7
    "traffic light",    # 8
    "traffic sign",     # 9
]

# Also handle alternate category names used in some BDD100K versions
CATEGORY_ALIASES = {
    "person": "pedestrian",
    "motor": "motorcycle",
    "bike": "bicycle",
}

BDD_CLASS_TO_ID = {name: idx for idx, name in enumerate(BDD_CLASSES)}

# BDD100K image dimensions (all images are 1280×720)
IMG_W = 1280
IMG_H = 720


def bbox_to_yolo(x1, y1, x2, y2):
    """Convert [x1, y1, x2, y2] pixel coords → YOLO [cx, cy, w, h] normalized."""
    cx = max(0.0, min(1.0, ((x1 + x2) / 2.0) / IMG_W))
    cy = max(0.0, min(1.0, ((y1 + y2) / 2.0) / IMG_H))
    w  = max(0.0, min(1.0, (x2 - x1) / IMG_W))
    h  = max(0.0, min(1.0, (y2 - y1) / IMG_H))
    return cx, cy, w, h


def resolve_category(category: str) -> int:
    """Map a BDD100K category string to a YOLO class ID, or -1 if unknown."""
    cat = category.lower().strip()
    # Direct match
    if cat in BDD_CLASS_TO_ID:
        return BDD_CLASS_TO_ID[cat]
    # Alias match
    if cat in CATEGORY_ALIASES:
        return BDD_CLASS_TO_ID[CATEGORY_ALIASES[cat]]
    return -1


def convert_per_image_jsons(json_dir: str, output_label_dir: str) -> dict:
    """
    Convert per-image Scalabel JSON files → YOLO .txt label files.
    Each JSON has: {"name": ..., "frames": [{"objects": [{"category":..., "box2d":...}]}]}
    """
    os.makedirs(output_label_dir, exist_ok=True)

    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])
    stats = {
        "total_images": len(json_files),
        "images_with_labels": 0,
        "total_boxes": 0,
        "skipped_categories": set(),
    }

    for jf in json_files:
        json_path = os.path.join(json_dir, jf)
        label_name = Path(jf).stem + ".txt"
        label_path = os.path.join(output_label_dir, label_name)

        with open(json_path, 'r') as f:
            data = json.load(f)

        lines = []
        # Scalabel format: data["frames"][0]["objects"]
        frames = data.get("frames", [])
        for frame in frames:
            objects = frame.get("objects", [])
            for obj in objects:
                category = obj.get("category", "")
                class_id = resolve_category(category)
                if class_id < 0:
                    stats["skipped_categories"].add(category)
                    continue

                box2d = obj.get("box2d")
                if box2d is None:
                    continue

                cx, cy, w, h = bbox_to_yolo(box2d["x1"], box2d["y1"],
                                             box2d["x2"], box2d["y2"])
                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                stats["total_boxes"] += 1

        with open(label_path, 'w') as f:
            f.write("\n".join(lines))

        if lines:
            stats["images_with_labels"] += 1

    return stats


def convert_consolidated_json(json_path: str, output_label_dir: str) -> dict:
    """
    Convert consolidated det_val.json → YOLO .txt label files.
    Each entry has: {"name": "xxx.jpg", "labels": [{"category":..., "box2d":...}]}
    """
    os.makedirs(output_label_dir, exist_ok=True)

    with open(json_path, 'r') as f:
        data = json.load(f)

    stats = {
        "total_images": len(data),
        "images_with_labels": 0,
        "total_boxes": 0,
        "skipped_categories": set(),
    }

    for frame in data:
        img_name = frame["name"]
        label_name = Path(img_name).stem + ".txt"
        label_path = os.path.join(output_label_dir, label_name)

        lines = []
        if "labels" in frame and frame["labels"] is not None:
            for label in frame["labels"]:
                category = label.get("category", "")
                class_id = resolve_category(category)
                if class_id < 0:
                    stats["skipped_categories"].add(category)
                    continue

                box2d = label.get("box2d")
                if box2d is None:
                    continue

                cx, cy, w, h = bbox_to_yolo(box2d["x1"], box2d["y1"],
                                             box2d["x2"], box2d["y2"])
                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                stats["total_boxes"] += 1

        with open(label_path, 'w') as f:
            f.write("\n".join(lines))

        if lines:
            stats["images_with_labels"] += 1

    return stats


def create_yaml(dataset_root: str, yaml_path: str):
    """Generate the Ultralytics dataset YAML configuration."""
    config = {
        "path": os.path.abspath(dataset_root),
        "val": "images/val",
        "nc": len(BDD_CLASSES),
        "names": BDD_CLASSES,
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"  Dataset YAML saved to: {yaml_path}")


def find_data_paths(data_root: str):
    """
    Auto-detect where images and labels are, handling various extraction layouts.
    Returns: (image_dir, label_source, label_format)
    """
    candidates_images = [
        os.path.join(data_root, "images", "100k", "val"),   # already organized
        os.path.join(data_root, "100k", "val"),              # direct extraction
    ]

    image_dir = None
    for d in candidates_images:
        if os.path.isdir(d):
            jpgs = [f for f in os.listdir(d) if f.endswith('.jpg')]
            if jpgs:
                image_dir = d
                break

    # Check for per-image JSONs in the same val directory (or a separate labels dir)
    candidates_labels_per_image = [
        os.path.join(data_root, "labels", "100k", "val"),
        os.path.join(data_root, "100k", "val"),    # JSONs co-located with images
        os.path.join(data_root, "labels", "val"),
    ]
    candidates_labels_consolidated = [
        os.path.join(data_root, "labels", "det_20", "det_val.json"),
        os.path.join(data_root, "det_val.json"),
    ]

    # Check consolidated first
    for p in candidates_labels_consolidated:
        if os.path.isfile(p):
            return image_dir, p, "consolidated"

    # Check per-image JSONs
    for d in candidates_labels_per_image:
        if os.path.isdir(d):
            jsons = [f for f in os.listdir(d) if f.endswith('.json')]
            if jsons:
                return image_dir, d, "per_image"

    return image_dir, None, None


def setup(data_root: str = "data/bdd100k"):
    """Main setup routine."""
    data_root = os.path.abspath(data_root)

    print(f"Scanning {data_root} for BDD100K data...")
    image_dir, label_source, label_format = find_data_paths(data_root)

    # ── Validate images ─────────────────────────────────────────────
    if image_dir is None:
        print(f"\nERROR: No validation images found!")
        print(f"Please extract bdd100k_images_100k.zip into {data_root}/")
        print(f"Expected JPEGs in one of:")
        print(f"  {data_root}/100k/val/")
        print(f"  {data_root}/images/100k/val/")
        return False

    num_images = len([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    print(f"✓ Found {num_images} validation images in:\n  {image_dir}")

    # ── Validate labels ─────────────────────────────────────────────
    if label_source is None:
        print(f"\nERROR: No label files found!")
        print(f"Please extract bdd100k_labels.zip into {data_root}/")
        return False

    print(f"✓ Found labels ({label_format} format):\n  {label_source}")

    # ── Create YOLO-compatible directory layout ─────────────────────
    # Ultralytics expects:  <root>/images/val/  and  <root>/labels/val/
    yolo_image_dir = os.path.join(data_root, "images", "val")
    yolo_label_dir = os.path.join(data_root, "labels", "val")

    # Symlink or copy images into the expected location
    if not os.path.exists(yolo_image_dir):
        os.makedirs(os.path.dirname(yolo_image_dir), exist_ok=True)
        os.symlink(image_dir, yolo_image_dir)
        print(f"\n  Symlinked images/val → {os.path.relpath(image_dir, data_root)}")
    else:
        # Check if it's already pointing to the right place
        if os.path.islink(yolo_image_dir):
            print(f"\n  images/val symlink already exists")
        else:
            print(f"\n  images/val directory already exists")

    # ── Convert labels → YOLO .txt ──────────────────────────────────
    print(f"\nConverting BDD100K labels to YOLO format...")
    if label_format == "per_image":
        stats = convert_per_image_jsons(label_source, yolo_label_dir)
    else:
        stats = convert_consolidated_json(label_source, yolo_label_dir)

    print(f"  Total images:         {stats['total_images']}")
    print(f"  Images with labels:   {stats['images_with_labels']}")
    print(f"  Total bounding boxes: {stats['total_boxes']}")
    if stats['skipped_categories']:
        print(f"  Skipped categories:   {stats['skipped_categories']}")

    # ── Generate dataset YAML ───────────────────────────────────────
    yaml_path = os.path.join(data_root, "bdd100k.yaml")
    create_yaml(data_root, yaml_path)

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"✅ BDD100K setup complete!")
    print(f"   Images:  {yolo_image_dir} ({num_images} files)")
    print(f"   Labels:  {yolo_label_dir} ({stats['images_with_labels']} with boxes)")
    print(f"   Config:  {yaml_path}")
    print(f"{'='*60}")
    print(f"\n   Next step: python3 python/experiment_bdd.py")
    print(f"   Quick test: python3 python/experiment_bdd.py --max-images 100")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup BDD100K for YOLO evaluation")
    parser.add_argument("--data-root", default="data/bdd100k",
                        help="Path to BDD100K data root (default: data/bdd100k)")
    args = parser.parse_args()
    setup(args.data_root)
