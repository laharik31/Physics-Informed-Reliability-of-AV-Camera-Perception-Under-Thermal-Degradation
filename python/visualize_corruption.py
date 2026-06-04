"""
Generates a visual grid showing how each sample image degrades
across the time snapshots used in our experiment.
"""
import cv2
import glob
import os
import numpy as np
import argparse
from lookup import PhysicsLookup
from corruptor import OpticalCorruptor

ALL_SURFACES = [
    "Untreated glass",
    "Hydrophilic coat",
    "Hydrophobic coat",
    "Superhydrophobic",
]

def visualize(surface, rh, mode, out_dir):
    lookup = PhysicsLookup()
    corruptor = OpticalCorruptor(lookup)

    image_paths = sorted(glob.glob("data/sample_images/*.jpg"))
    if not image_paths:
        print("No sample images found! Run download_sample.py first.")
        return

    t_snapshots = [0, 60, 120, 180, 300, 450, 600]
    delta_t = 5

    os.makedirs(out_dir, exist_ok=True)

    # For each image, save the corrupted version at every time snapshot
    for img_idx, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read {img_path}, skipping.")
            continue

        # Resize for consistent display
        img = cv2.resize(img, (640, 480))

        row_images = []
        for t in t_snapshots:
            corrupted = corruptor.corrupt_image(img, t_s=t, delta_t_c=delta_t, rh=rh, surface=surface, mode=mode)

            # Add label to the image
            label = f"t={t}s"
            tau, sigma, cov = lookup.get_optical_params(t, delta_t, rh, surface)
            sublabel = f"tau={tau:.2f}, C={cov:.2f}"
            cv2.putText(corrupted, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(corrupted, sublabel, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            row_images.append(corrupted)

            # Also save individual corrupted images
            safe_surface = surface.replace(" ", "_").lower()
            individual_path = os.path.join(out_dir, f"sample_{img_idx}_{safe_surface}_{mode}_t{t}s.jpg")
            cv2.imwrite(individual_path, corrupted)

        # Build a grid: top row = first 4, bottom row = last 3 + blank
        top_row = np.hstack(row_images[:4])
        # Pad the bottom row to match width
        bottom_images = row_images[4:]
        while len(bottom_images) < 4:
            bottom_images.append(np.zeros_like(row_images[0]))
        bottom_row = np.hstack(bottom_images)

        grid = np.vstack([top_row, bottom_row])
        safe_surface = surface.replace(" ", "_").lower()
        grid_path = os.path.join(out_dir, f"sample_{img_idx}_grid_{safe_surface}_{mode}.jpg")
        cv2.imwrite(grid_path, grid)
        print(f"Saved grid for sample_{img_idx} ({surface}, {mode}) -> {grid_path}")

    print(f"\nAll corrupted images saved to {out_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Corruption Grids")
    parser.add_argument("--surface", type=str, default=None,
                        help="Single surface to evaluate")
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

    out_dir = f"results/corrupted_images_{args.mode}"
    for surf in surfaces:
        visualize(surf, args.rh, args.mode, out_dir)
