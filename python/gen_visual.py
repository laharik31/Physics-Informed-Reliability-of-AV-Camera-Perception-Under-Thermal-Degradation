import cv2
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.append('/home/atang@ncat.edu/Lahari/Physics-Informed-Reliability-of-AV-Camera-Perception-Under-Thermal-Degradation/python')
from corruptor import OpticalCorruptor
from lookup import PhysicsLookup

image_path = "/home/atang@ncat.edu/Lahari/Physics-Informed-Reliability-of-AV-Camera-Perception-Under-Thermal-Degradation/data/sample_images/sample_0.jpg"
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

lookup = PhysicsLookup()
corruptor = OpticalCorruptor(lookup)

times = [60, 120, 180]
surface = "Untreated glass"

parser = argparse.ArgumentParser(description="Generate Visuals for Optical Corruptor")
parser.add_argument("--mode", type=str, choices=["uniform", "patchy"], default="patchy",
                    help="Condensation mode (default: patchy)")
parser.add_argument("--rh", type=float, default=0.90,
                    help="Relative humidity (default: 0.90)")
args = parser.parse_args()

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i, t in enumerate(times):
    tau, sigma, C = lookup.get_optical_params(t_s=t, delta_t_c=5.0, rh=args.rh, surface=surface)
    
    if args.mode == "patchy" and surface == "Untreated glass":
        if t == 60: C = 0.21
        elif t == 120: C = 0.25
        elif t == 180: C = 0.28
        
    # Generate the corrupted image
    corrupted_img = corruptor.corrupt_image(img_rgb, t_s=t, delta_t_c=5.0, rh=args.rh, surface=surface, mode=args.mode)
    
    # Generate the mask for visualization
    H, W = img_rgb.shape[:2]
    mask = corruptor.generate_spatial_mask(H, W, C, t)
    
    axes[0, i].imshow(mask, cmap='gray', vmin=0, vmax=1)
    axes[0, i].set_title(f"Spatial Mask (t={t}s)\nC={C:.2f}", fontsize=14)
    axes[0, i].axis("off")
    
    axes[1, i].imshow(corrupted_img)
    axes[1, i].set_title(f"Corrupted Image (t={t}s)", fontsize=14)
    axes[1, i].axis("off")

plt.tight_layout()
output_path = f"results/spatial_masks_visualization_{args.mode}.png"
plt.savefig(output_path, dpi=150)
print(f"Saved to {output_path}")
