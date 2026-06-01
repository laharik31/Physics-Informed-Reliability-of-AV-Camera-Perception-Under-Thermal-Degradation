"""
BDD100K True mAP Experiment
============================
Evaluates YOLOv8 detection performance under physics-informed thermal
degradation using the BDD100K validation set with ground-truth annotations.

For each time snapshot:
  1. Corrupts all BDD100K val images using OpticalCorruptor
  2. Runs Ultralytics model.val() to compute true mAP@50 and mAP@50-95
  3. Records metrics and generates an availability curve

Results are saved to:
  results/bdd_availability_curve.png   — mAP vs time plot
  results/bdd_results.csv              — numerical results table

Usage:
  python3 python/experiment_bdd.py
  python3 python/experiment_bdd.py --max-images 500   # quick test with subset
"""

import cv2
import os
import sys
import csv
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from corruptor import OpticalCorruptor
from lookup import PhysicsLookup


def corrupt_dataset(image_dir: str, output_dir: str, corruptor: OpticalCorruptor,
                    t_s: float, delta_t_c: float, rh: float, surface: str,
                    max_images: int = None) -> int:
    """
    Apply physics-informed corruption to all images in a directory.
    
    Args:
        image_dir:  Path to clean validation images
        output_dir: Path to write corrupted images (same filenames)
        corruptor:  OpticalCorruptor instance
        t_s:        Simulation time in seconds
        delta_t_c:  Subcooling ΔT in °C
        rh:         Relative humidity (0-1)
        surface:    Surface treatment type
        max_images: If set, only process this many images (for quick testing)
    
    Returns:
        Number of images processed
    """
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    if max_images is not None:
        image_files = image_files[:max_images]

    for i, filename in enumerate(image_files):
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue

        corrupted = corruptor.corrupt_image(img, t_s=t_s, delta_t_c=delta_t_c,
                                            rh=rh, surface=surface)
        cv2.imwrite(os.path.join(output_dir, filename), corrupted)

        if (i + 1) % 500 == 0:
            print(f"    Corrupted {i+1}/{len(image_files)} images...")

    return len(image_files)


def run_experiment(data_root: str = "data/bdd100k", max_images: int = None):
    """
    Main experiment loop: corrupt → validate → record → plot.
    """
    data_root = os.path.abspath(data_root)
    yaml_path = os.path.join(data_root, "bdd100k.yaml")

    # ── Validate setup ──────────────────────────────────────────────
    if not os.path.isfile(yaml_path):
        print("ERROR: bdd100k.yaml not found. Run setup_bdd100k.py first.")
        print("  python3 python/setup_bdd100k.py")
        return

    clean_image_dir = os.path.join(data_root, "images", "val")
    label_dir = os.path.join(data_root, "labels", "val")

    if not os.path.isdir(clean_image_dir):
        print(f"ERROR: Validation images not found at {clean_image_dir}")
        return
    if not os.path.isdir(label_dir):
        print(f"ERROR: YOLO labels not found at {label_dir}")
        return

    num_images = len([f for f in os.listdir(clean_image_dir) if f.endswith('.jpg')])
    effective_count = min(num_images, max_images) if max_images else num_images
    print(f"BDD100K Experiment: {effective_count} images"
          f"{' (subset)' if max_images else ''}")

    # ── Setup ───────────────────────────────────────────────────────
    print("Loading YOLOv8n model...")
    model = YOLO('yolov8n.pt')

    lookup = PhysicsLookup()
    corruptor = OpticalCorruptor(lookup)

    # Time snapshots (seconds)
    t_snapshots = [0, 60, 120, 180, 300, 450, 600]

    # Environmental conditions (matches Kim/Hendrycks fog calibration)
    delta_t = 5
    rh = 0.80
    surface = "Untreated glass"

    # Directory for temporarily storing corrupted images
    # We use "images/val_corrupted" and create a matching "labels/val_corrupted"
    # symlink because Ultralytics auto-discovers labels by replacing "images"
    # with "labels" in the path (e.g., images/val_corrupted → labels/val_corrupted).
    corrupted_base = os.path.join(data_root, "images", "val_corrupted")

    # Results storage
    results = []

    # ── Evaluation loop ─────────────────────────────────────────────
    import yaml

    for t in t_snapshots:
        tau, sigma, coverage = lookup.get_optical_params(t, delta_t, rh, surface)
        print(f"\n{'='*60}")
        print(f"Snapshot t={t}s  |  τ={tau:.4f}  σ={sigma:.2f}px  C={coverage:.4f}")
        print(f"{'='*60}")

        # Clean up any previous iteration's corrupted images
        if os.path.isdir(corrupted_base):
            shutil.rmtree(corrupted_base)

        if t == 0 and coverage == 0.0:
            # At t=0, images are clean — copy originals into val_corrupted
            # (We can't use the symlink at images/val directly because
            #  Ultralytics resolves symlinks and breaks label auto-discovery)
            print("  Copying clean images (no corruption needed)...")
            os.makedirs(corrupted_base, exist_ok=True)
            image_files = sorted([f for f in os.listdir(clean_image_dir) if f.endswith('.jpg')])
            if max_images is not None:
                image_files = image_files[:max_images]
            for f in image_files:
                shutil.copy2(os.path.join(clean_image_dir, f),
                             os.path.join(corrupted_base, f))
            print(f"  Copied {len(image_files)} images → {corrupted_base}")
        else:
            # Corrupt images and save to temporary directory
            print(f"  Corrupting images...")
            n = corrupt_dataset(clean_image_dir, corrupted_base, corruptor,
                                t_s=t, delta_t_c=delta_t, rh=rh, surface=surface,
                                max_images=max_images)
            print(f"  Corrupted {n} images → {corrupted_base}")

        # ── Set up labels for val_corrupted ─────────────────────────
        # Ultralytics finds labels by replacing "images" → "labels" in the
        # val path: images/val_corrupted → labels/val_corrupted
        corrupted_label_link = os.path.join(data_root, "labels", "val_corrupted")

        # Remove stale cache files that Ultralytics may have written
        cache_file = corrupted_label_link + ".cache"
        if os.path.exists(cache_file):
            os.remove(cache_file)

        if max_images is not None:
            # Subset mode: copy only the matching label files
            if os.path.islink(corrupted_label_link):
                os.unlink(corrupted_label_link)
            if os.path.isdir(corrupted_label_link):
                shutil.rmtree(corrupted_label_link)
            os.makedirs(corrupted_label_link, exist_ok=True)

            for f in os.listdir(corrupted_base):
                if f.endswith('.jpg'):
                    lbl_name = Path(f).stem + ".txt"
                    src = os.path.join(label_dir, lbl_name)
                    dst = os.path.join(corrupted_label_link, lbl_name)
                    if os.path.exists(src):
                        shutil.copy2(src, dst)
        else:
            # Full dataset: symlink labels/val_corrupted → labels/val
            if os.path.islink(corrupted_label_link):
                os.unlink(corrupted_label_link)
            elif os.path.isdir(corrupted_label_link):
                shutil.rmtree(corrupted_label_link)
            os.symlink(os.path.abspath(label_dir), corrupted_label_link)

        # ── Build eval YAML ─────────────────────────────────────────
        temp_yaml_path = os.path.join(data_root, "bdd100k_eval.yaml")
        eval_config = {
            "path": data_root,
            "train": "images/val_corrupted",  # Required by Ultralytics (not used)
            "val": "images/val_corrupted",
            "nc": 10,
            "names": [
                "pedestrian", "rider", "car", "truck", "bus",
                "train", "motorcycle", "bicycle", "traffic light", "traffic sign"
            ],
        }
        with open(temp_yaml_path, 'w') as f:
            yaml.dump(eval_config, f, default_flow_style=False, sort_keys=False)

        print(f"  Running YOLOv8 validation...")
        try:
            metrics = model.val(
                data=temp_yaml_path,
                imgsz=640,
                batch=16,
                verbose=False,
                plots=False,
            )
            map50 = metrics.box.map50
            map50_95 = metrics.box.map
            print(f"  ✅ mAP@50 = {map50:.4f}  |  mAP@50-95 = {map50_95:.4f}")
        except Exception as e:
            print(f"  ❌ Validation failed: {e}")
            map50 = 0.0
            map50_95 = 0.0

        results.append({
            "t_s": t,
            "tau": tau,
            "sigma": sigma,
            "coverage": coverage,
            "mAP50": map50,
            "mAP50_95": map50_95,
        })

        # Clean up corrupted images to save disk space
        if os.path.isdir(corrupted_base):
            shutil.rmtree(corrupted_base)

    # Clean up temp files
    temp_yaml = os.path.join(data_root, "bdd100k_eval.yaml")
    if os.path.exists(temp_yaml):
        os.remove(temp_yaml)
    corrupted_label_link = os.path.join(data_root, "labels", "val_corrupted")
    if os.path.islink(corrupted_label_link):
        os.unlink(corrupted_label_link)
    elif os.path.isdir(corrupted_label_link):
        shutil.rmtree(corrupted_label_link)

    # ── Save numerical results ──────────────────────────────────────
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "bdd_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["t_s", "tau", "sigma", "coverage",
                                                "mAP50", "mAP50_95"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nNumerical results saved to {csv_path}")

    # ── Print results table ─────────────────────────────────────────
    baseline_map50 = results[0]["mAP50"] if results[0]["mAP50"] > 0 else 1.0
    print(f"\n{'='*75}")
    print(f"{'Time':>6s}  {'τ':>6s}  {'C':>6s}  {'mAP@50':>8s}  {'mAP@50-95':>10s}  {'% Drop':>8s}")
    print(f"{'-'*75}")
    for r in results:
        drop = ((r["mAP50"] - baseline_map50) / baseline_map50) * 100
        print(f"{r['t_s']:>5.0f}s  {r['tau']:>6.4f}  {r['coverage']:>6.4f}  "
              f"{r['mAP50']:>8.4f}  {r['mAP50_95']:>10.4f}  {drop:>+7.1f}%")
    print(f"{'='*75}")

    # ── Plot availability curve ─────────────────────────────────────
    times = [r["t_s"] for r in results]
    map50_vals = [r["mAP50"] for r in results]
    map50_95_vals = [r["mAP50_95"] for r in results]

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(times, map50_vals, marker='o', linestyle='-', color='#2196F3',
            linewidth=2.5, markersize=8, label='mAP@50', zorder=5)
    ax.plot(times, map50_95_vals, marker='s', linestyle='--', color='#FF5722',
            linewidth=2, markersize=7, label='mAP@50-95', zorder=5)

    # Shade the blackout window (condensation active)
    ax.axvspan(0, 240, alpha=0.08, color='red', label='Condensation window')

    # Mark heater activation
    ax.axvline(x=180, color='green', linestyle=':', linewidth=1.5,
               label='Heater ON (t=180s)')

    ax.set_title(f'YOLOv8 Detection on BDD100K under Thermal Degradation\n'
                 f'(RH={rh*100:.0f}%, ΔT={delta_t}°C, {surface})',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('mAP', fontsize=12)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-10, 620)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "bdd_availability_curve.png")
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Availability curve saved to {plot_path}")

    print(f"\n✅ BDD100K experiment complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BDD100K True mAP Experiment")
    parser.add_argument("--data-root", default="data/bdd100k",
                        help="Path to BDD100K data root (default: data/bdd100k)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit number of images for quick testing (default: all)")
    args = parser.parse_args()
    run_experiment(data_root=args.data_root, max_images=args.max_images)
