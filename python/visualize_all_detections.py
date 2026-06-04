"""
Generates exhaustive YOLOv8 detection visualizations for all 4 glass
coatings across both condensation modes (uniform and patchy).

Produces 8 high-resolution image grids showing bounding boxes at
t = [0, 60, 120, 180, 300, 450, 600] seconds.
"""
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from corruptor import OpticalCorruptor
from lookup import PhysicsLookup

ALL_SURFACES = [
    "Untreated glass",
    "Hydrophilic coat",
    "Hydrophobic coat",
    "Superhydrophobic",
]

MODES = ["uniform", "patchy"]

T_SNAPSHOTS = [0, 60, 120, 180, 300, 450, 600]


def generate_detection_grid(model, corruptor, lookup, img_path, surface, mode, rh, out_dir):
    """
    Generate a single detection grid for one surface + mode combo.
    Shows YOLOv8 bounding boxes at each time snapshot.
    """
    delta_t = 5
    original_img = cv2.imread(img_path)
    if original_img is None:
        print(f"Could not read {img_path}")
        return None

    original_img = cv2.resize(original_img, (640, 480))

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    axes_flat = axes.flatten()

    for i, t in enumerate(T_SNAPSHOTS):
        # Corrupt the image
        if t == 0:
            img = original_img.copy()
            tau, sigma, C = 1.0, 0.0, 0.0
        else:
            tau, sigma, C = lookup.get_optical_params(t, delta_t, rh, surface)
            # Mirror the corruptor's C overrides so labels match actual corruption
            if surface in ("Untreated glass", "Hydrophilic coat"):
                if t == 60:
                    C = 0.21
                elif t == 120:
                    C = 0.25
                elif t == 180:
                    C = 0.28
            # Clear the corruptor mask cache so each call is fresh
            corruptor.mask_cache.clear()
            img = corruptor.corrupt_image(original_img, t, delta_t, rh, surface, mode=mode)

        # Run YOLOv8 detection
        res = model.predict(img, imgsz=640, conf=0.25, verbose=False)[0]
        plotted = res.plot(line_width=2)
        plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

        n_detections = len(res.boxes)

        axes_flat[i].imshow(plotted_rgb)
        axes_flat[i].set_title(
            f"t={t}s  |  τ={tau:.2f}, C={C:.2f}\nDetections: {n_detections}",
            fontsize=11, fontweight='bold'
        )
        axes_flat[i].axis('off')

    # Hide the 8th subplot (we only have 7 time snapshots)
    axes_flat[7].axis('off')
    axes_flat[7].set_visible(False)

    safe_surface = surface.replace(" ", "_").lower()
    fig.suptitle(
        f"YOLOv8 Object Detection — {surface} ({mode.capitalize()} Condensation, RH={int(rh*100)}%)",
        fontsize=16, fontweight='bold', y=1.01
    )
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"detections_{mode}_{safe_surface}_rh{int(rh*100)}.png")
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")
    return out_path


def main():
    print("Loading YOLOv8 model...")
    model = YOLO('python/yolov8n.pt')
    lookup = PhysicsLookup()
    corruptor = OpticalCorruptor(lookup)

    # Pick a representative BDD100K image
    img_dir = "data/bdd100k/images/val"
    sample_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    img_path = os.path.join(img_dir, sample_files[5])  # Same image used previously
    print(f"Using image: {img_path}")

    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    rh = 0.80  # Use RH=80% for consistency

    for mode in MODES:
        for surface in ALL_SURFACES:
            print(f"\nProcessing: {surface} ({mode})...")
            generate_detection_grid(
                model, corruptor, lookup, img_path,
                surface, mode, rh, out_dir
            )

    print("\n✅ All detection grids generated!")


if __name__ == "__main__":
    main()
