"""
BDD100K True mAP Experiment
============================
Evaluates YOLOv8 detection performance under physics-informed thermal
degradation using the BDD100K validation set with ground-truth annotations.

Supports multi-surface evaluation: loops through all glass coatings and
generates both per-surface and combined availability curves.

Results are saved to:
  results/bdd_results.csv                    — numerical results (all surfaces)
  results/bdd_availability_curve.png         — single-surface plot (backward compat)
  results/bdd_availability_curve_combined.png — combined multi-surface comparison

Usage:
  python3 python/experiment_bdd.py
  python3 python/experiment_bdd.py --max-images 500       # quick test
  python3 python/experiment_bdd.py --surface "Untreated glass"  # single surface
  python3 python/experiment_bdd.py --all-surfaces          # all 4 coatings
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
import yaml


# All available glass coatings in the physics lookup
ALL_SURFACES = [
    "Untreated glass",
    "Hydrophilic coat",
    "Hydrophobic coat",
    "Superhydrophobic",
]

# Colors and markers for combined plot
SURFACE_STYLES = {
    "Untreated glass":   {"color": "#E53935", "marker": "o", "label": "Untreated Glass"},
    "Hydrophilic coat":  {"color": "#1E88E5", "marker": "s", "label": "Hydrophilic"},
    "Hydrophobic coat":  {"color": "#43A047", "marker": "D", "label": "Hydrophobic"},
    "Superhydrophobic":  {"color": "#8E24AA", "marker": "^", "label": "Superhydrophobic"},
}


import multiprocessing
from functools import partial

def _process_single_image(filename, image_dir, output_dir, corruptor, t_s, delta_t_c, rh, surface, mode):
    img = cv2.imread(os.path.join(image_dir, filename))
    if img is not None:
        corrupted = corruptor.corrupt_image(img, t_s=t_s, delta_t_c=delta_t_c, rh=rh, surface=surface, mode=mode)
        cv2.imwrite(os.path.join(output_dir, filename), corrupted)

def corrupt_dataset(image_dir: str, output_dir: str, corruptor: OpticalCorruptor,
                    t_s: float, delta_t_c: float, rh: float, surface: str, mode: str,
                    max_images: int = None) -> int:
    """Apply physics-informed corruption to all images in a directory using multiprocessing."""
    os.makedirs(output_dir, exist_ok=True)
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    if max_images is not None:
        image_files = image_files[:max_images]

    # Pre-generate and cache the spatial mask once so workers don't duplicate effort
    if image_files:
        sample_img = cv2.imread(os.path.join(image_dir, image_files[0]))
        if sample_img is not None:
            corruptor.corrupt_image(sample_img, t_s=t_s, delta_t_c=delta_t_c, rh=rh, surface=surface, mode=mode)

    func = partial(_process_single_image, image_dir=image_dir, output_dir=output_dir,
                   corruptor=corruptor, t_s=t_s, delta_t_c=delta_t_c, rh=rh, surface=surface, mode=mode)
                   
    with multiprocessing.Pool(processes=64) as pool:
        for i, _ in enumerate(pool.imap_unordered(func, image_files)):
            if (i + 1) % 1000 == 0:
                print(f"    Corrupted {i+1}/{len(image_files)} images...")
                
    return len(image_files)


def run_single_surface(model, lookup, corruptor, surface, t_snapshots,
                       delta_t, rh, mode, data_root, clean_image_dir, label_dir,
                       corrupted_base, max_images):
    """Run the evaluation loop for a single surface coating."""
    results = []

    for t in t_snapshots:
        tau, sigma, coverage = lookup.get_optical_params(t, delta_t, rh, surface)
        print(f"\n{'='*60}")
        print(f"[{surface}] t={t}s  |  τ={tau:.4f}  σ={sigma:.2f}px  C={coverage:.4f}")
        print(f"{'='*60}")

        # Clean up previous iteration
        if os.path.isdir(corrupted_base):
            shutil.rmtree(corrupted_base)

        if t == 0 and coverage == 0.0:
            # Clean images — copy originals into val_corrupted
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
            print(f"  Corrupting images...")
            n = corrupt_dataset(clean_image_dir, corrupted_base, corruptor,
                                t_s=t, delta_t_c=delta_t, rh=rh, surface=surface, mode=mode,
                                max_images=max_images)
            print(f"  Corrupted {n} images → {corrupted_base}")

        # Set up labels for val_corrupted
        corrupted_label_link = os.path.join(data_root, "labels", "val_corrupted")
        cache_file = corrupted_label_link + ".cache"
        if os.path.exists(cache_file):
            os.remove(cache_file)

        if max_images is not None:
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
            if os.path.islink(corrupted_label_link):
                os.unlink(corrupted_label_link)
            elif os.path.isdir(corrupted_label_link):
                shutil.rmtree(corrupted_label_link)
            os.symlink(os.path.abspath(label_dir), corrupted_label_link)

        # Build eval YAML
        temp_yaml_path = os.path.join(data_root, "bdd100k_eval.yaml")
        eval_config = {
            "path": data_root,
            "train": "images/val_corrupted",
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
            "surface": surface,
            "t_s": t,
            "tau": tau,
            "sigma": sigma,
            "coverage": coverage,
            "mAP50": map50,
            "mAP50_95": map50_95,
        })

        # Clean up corrupted images
        if os.path.isdir(corrupted_base):
            shutil.rmtree(corrupted_base)

    # Clean up temp files
    for tmp in [os.path.join(data_root, "bdd100k_eval.yaml")]:
        if os.path.exists(tmp):
            os.remove(tmp)
    corrupted_label_link = os.path.join(data_root, "labels", "val_corrupted")
    if os.path.islink(corrupted_label_link):
        os.unlink(corrupted_label_link)
    elif os.path.isdir(corrupted_label_link):
        shutil.rmtree(corrupted_label_link)

    return results


def plot_single_surface(results, surface, rh, delta_t, out_dir):
    """Generate availability curve for a single surface (backward compatible)."""
    times = [r["t_s"] for r in results]
    map50_vals = [r["mAP50"] for r in results]
    map50_95_vals = [r["mAP50_95"] for r in results]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(times, map50_vals, marker='o', linestyle='-', color='#2196F3',
            linewidth=2.5, markersize=8, label='mAP@50', zorder=5)
    ax.plot(times, map50_95_vals, marker='s', linestyle='--', color='#FF5722',
            linewidth=2, markersize=7, label='mAP@50-95', zorder=5)
    ax.axvspan(0, 240, alpha=0.08, color='red', label='Condensation window')
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

    safe_name = surface.replace(" ", "_").lower()
    plot_path = os.path.join(out_dir, f"bdd_availability_curve_{safe_name}.png")
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Availability curve saved to {plot_path}")

    # Also save as the default name if it's untreated glass (backward compat)
    if surface == "Untreated glass":
        default_path = os.path.join(out_dir, "bdd_availability_curve.png")
        fig2, ax2 = plt.subplots(figsize=(12, 7))
        ax2.plot(times, map50_vals, marker='o', linestyle='-', color='#2196F3',
                linewidth=2.5, markersize=8, label='mAP@50', zorder=5)
        ax2.plot(times, map50_95_vals, marker='s', linestyle='--', color='#FF5722',
                linewidth=2, markersize=7, label='mAP@50-95', zorder=5)
        ax2.axvspan(0, 240, alpha=0.08, color='red', label='Condensation window')
        ax2.axvline(x=180, color='green', linestyle=':', linewidth=1.5,
                   label='Heater ON (t=180s)')
        ax2.set_title(f'YOLOv8 Detection on BDD100K under Thermal Degradation\n'
                     f'(RH={rh*100:.0f}%, ΔT={delta_t}°C, {surface})',
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('mAP', fontsize=12)
        ax2.legend(loc='lower right', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-10, 620)
        plt.tight_layout()
        fig2.savefig(default_path, dpi=300, bbox_inches='tight')
        plt.close()


def plot_combined(all_results, rh, delta_t, out_dir):
    """Generate a combined availability curve comparing all surfaces."""
    fig, ax = plt.subplots(figsize=(14, 8))

    for surface, results in all_results.items():
        style = SURFACE_STYLES[surface]
        times = [r["t_s"] for r in results]
        map50_vals = [r["mAP50"] for r in results]
        ax.plot(times, map50_vals, marker=style["marker"], linestyle='-',
                color=style["color"], linewidth=2.5, markersize=9,
                label=style["label"], zorder=5)

    # Shade condensation window and mark heater activation
    ax.axvspan(0, 240, alpha=0.06, color='red', label='Condensation window')
    ax.axvline(x=180, color='gray', linestyle=':', linewidth=1.5,
               label='Heater ON (t=180s)')

    ax.set_title(f'Glass Coating Comparison: mAP@50 under Thermal Degradation\n'
                 f'(BDD100K val, RH={rh*100:.0f}%, ΔT={delta_t}°C)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=13)
    ax.set_ylabel('mAP@50', fontsize=13)
    ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-10, 620)
    plt.tight_layout()

    plot_path = os.path.join(out_dir, "bdd_availability_curve_combined.png")
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nCombined availability curve saved to {plot_path}")


def run_experiment(data_root: str = "data/bdd100k", max_images: int = None,
                   surfaces: list = None, rh: float = 0.90, mode: str = "patchy"):
    """Main experiment: evaluate one or more surface coatings."""
    data_root = os.path.abspath(data_root)
    yaml_path = os.path.join(data_root, "bdd100k.yaml")

    if not os.path.isfile(yaml_path):
        print("ERROR: bdd100k.yaml not found. Run setup_bdd100k.py first.")
        return

    clean_image_dir = os.path.join(data_root, "images", "val")
    label_dir = os.path.join(data_root, "labels", "val")

    if not os.path.isdir(clean_image_dir):
        print(f"ERROR: Validation images not found at {clean_image_dir}")
        return
    if not os.path.isdir(label_dir):
        print(f"ERROR: YOLO labels not found at {label_dir}")
        return

    if surfaces is None:
        surfaces = ["Untreated glass"]

    num_images = len([f for f in os.listdir(clean_image_dir) if f.endswith('.jpg')])
    effective_count = min(num_images, max_images) if max_images else num_images

    print(f"BDD100K Multi-Surface Experiment")
    print(f"  Images: {effective_count}{' (subset)' if max_images else ''}")
    print(f"  Surfaces: {surfaces}")

    # Setup
    print("Loading YOLOv8n model...")
    model = YOLO('yolov8n.pt')
    lookup = PhysicsLookup()
    corruptor = OpticalCorruptor(lookup)

    t_snapshots = [0, 60, 120, 180, 300, 450, 600]
    delta_t = 5
    corrupted_base = os.path.join(data_root, "images", "val_corrupted")
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    # Run all surfaces
    all_results = {}
    all_flat = []

    for surface in surfaces:
        print(f"\n{'#'*70}")
        print(f"# SURFACE: {surface}")
        print(f"{'#'*70}")

        results = run_single_surface(
            model, lookup, corruptor, surface, t_snapshots,
            delta_t, rh, data_root, clean_image_dir, label_dir,
            corrupted_base, max_images
        )
        all_results[surface] = results
        all_flat.extend(results)

        # Print per-surface results table
        baseline_map50 = results[0]["mAP50"] if results[0]["mAP50"] > 0 else 1.0
        print(f"\n{'='*75}")
        print(f"  [{surface}] Results")
        print(f"{'Time':>6s}  {'τ':>6s}  {'C':>6s}  {'mAP@50':>8s}  {'mAP@50-95':>10s}  {'% Drop':>8s}")
        print(f"{'-'*75}")
        for r in results:
            drop = ((r["mAP50"] - baseline_map50) / baseline_map50) * 100
            print(f"{r['t_s']:>5.0f}s  {r['tau']:>6.4f}  {r['coverage']:>6.4f}  "
                  f"{r['mAP50']:>8.4f}  {r['mAP50_95']:>10.4f}  {drop:>+7.1f}%")
        print(f"{'='*75}")

        # Generate per-surface availability curve
        plot_single_surface(results, surface, rh, delta_t, out_dir)

    # Save all results to CSV
    csv_path = os.path.join(out_dir, "bdd_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["surface", "t_s", "tau", "sigma",
                                                "coverage", "mAP50", "mAP50_95"])
        writer.writeheader()
        writer.writerows(all_flat)
    print(f"\nNumerical results saved to {csv_path}")

    # Generate combined plot if multiple surfaces
    if len(surfaces) > 1:
        plot_combined(all_results, rh, delta_t, out_dir)

    print(f"\n✅ BDD100K experiment complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BDD100K True mAP Experiment")
    parser.add_argument("--data-root", default="data/bdd100k",
                        help="Path to BDD100K data root (default: data/bdd100k)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit number of images for quick testing")
    parser.add_argument("--surface", type=str, default=None,
                        help="Single surface to evaluate (default: Untreated glass)")
    parser.add_argument("--all-surfaces", action="store_true",
                        help="Evaluate all 4 glass coatings")
    parser.add_argument("--mode", type=str, choices=["uniform", "patchy"], default="patchy",
                        help="Condensation mode (default: patchy)")
    parser.add_argument("--rh", type=float, default=0.90,
                        help="Relative humidity (default: 0.90)")
    args = parser.parse_args()

    if args.all_surfaces:
        surfaces = ALL_SURFACES
    elif args.surface:
        surfaces = [args.surface]
    else:
        surfaces = ["Untreated glass"]

    run_experiment(data_root=args.data_root, max_images=args.max_images,
                   surfaces=surfaces, rh=args.rh, mode=args.mode)
