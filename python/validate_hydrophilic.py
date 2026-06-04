import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from corruptor import OpticalCorruptor
from lookup import PhysicsLookup

def main():
    print("Loading YOLOv8 model...")
    model = YOLO('python/yolov8n.pt')
    lookup = PhysicsLookup()
    corruptor = OpticalCorruptor(lookup)

    val_dir = "data/bdd100k/images/val/"
    out_dir = "results/hydrophilic_validation"
    os.makedirs(out_dir, exist_ok=True)

    # Get all jpgs and pick 4 random ones
    all_imgs = [f for f in os.listdir(val_dir) if f.endswith('.jpg')]
    random.seed(42) # For reproducible random selection
    chosen_imgs = random.sample(all_imgs, 4)

    # We will test these specific times
    T_SNAPSHOTS = [0, 60, 120, 180]
    surface = "Hydrophilic coat"
    rh = 0.80

    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    
    for row, img_name in enumerate(chosen_imgs):
        img_path = os.path.join(val_dir, img_name)
        original_img = cv2.imread(img_path)
        original_img = cv2.resize(original_img, (640, 480))

        for col, t in enumerate(T_SNAPSHOTS):
            if t == 0:
                img = original_img.copy()
                tau, sigma, C = 1.0, 0.0, 0.0
            else:
                tau, sigma, C = lookup.get_optical_params(t, 5, rh, surface)
                # Overrides for accurate labeling
                if t == 60: C = 0.21
                elif t == 120: C = 0.25
                elif t == 180: C = 0.28
                
                corruptor.mask_cache.clear()
                # Uniform mode because Hydrophilic is always uniform
                img = corruptor.corrupt_image(original_img, t, 5, rh, surface, mode="uniform")

            # YOLO inference
            res = model.predict(img, imgsz=640, conf=0.25, verbose=False)[0]
            plotted = res.plot(line_width=2)
            plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
            n_detections = len(res.boxes)

            ax = axes[row, col]
            ax.imshow(plotted_rgb)
            ax.axis('off')

            if row == 0:
                ax.set_title(f"t={t}s  |  τ={tau:.2f}\nDetections: {n_detections}", fontsize=14, fontweight='bold', pad=15)
            else:
                ax.set_title(f"Detections: {n_detections}", fontsize=12, pad=5)

    fig.suptitle("Hydrophilic Coat YOLOv8 Erratic Validation (t=60s to 180s)", fontsize=20, fontweight='bold')
    plt.subplots_adjust(top=0.90, hspace=0.15, wspace=0.05)
    
    out_path = os.path.join(out_dir, "hydrophilic_erratic_validation.png")
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Validation grid saved to {out_path}")

if __name__ == "__main__":
    main()
