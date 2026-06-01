import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from corruptor import OpticalCorruptor
from lookup import PhysicsLookup

def create_visual_results():
    print("Loading model and physics engine...")
    model = YOLO('yolov8n.pt')
    lookup = PhysicsLookup()
    corruptor = OpticalCorruptor(lookup)
    
    # Environmental conditions
    delta_t = 5
    rh = 0.80
    surface = "Untreated glass"
    
    # Pick a good sample image from BDD100K val set
    img_dir = "../data/bdd100k/images/val"
    out_dir = "../results"
    
    # Let's just pick the first image, or a specific one
    # bcc942dc-d64dc48f.jpg is a good daytime driving scene
    sample_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    sample_img_path = os.path.join(img_dir, sample_files[5]) # 6th image
    
    print(f"Processing image: {sample_img_path}")
    original_img = cv2.imread(sample_img_path)
    
    snapshots = [
        (0, "Baseline (t=0s)"),
        (60, "Peak Condensation (t=60s)"),
        (300, "Heater Recovered (t=300s)")
    ]
    
    results_imgs = []
    
    for t, title in snapshots:
        # 1. Corrupt
        if t == 0:
            img = original_img.copy()
            tau, sigma, C = 1.0, 0.0, 0.0
        else:
            tau, sigma, C = lookup.get_optical_params(t, delta_t, rh, surface)
            img = corruptor.corrupt_image(original_img, t, delta_t, rh, surface)
            
        # 2. Run YOLO predict
        # We run predict and get the plotted image back
        res = model.predict(img, imgsz=640, conf=0.25, verbose=False)[0]
        plotted_img = res.plot(line_width=2)
        
        # Convert BGR to RGB for matplotlib
        plotted_img_rgb = cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB)
        
        results_imgs.append({
            "img": plotted_img_rgb,
            "title": f"{title}\n$\\tau={tau:.2f}, C={C:.2f}$",
            "detections": len(res.boxes)
        })
        
    # 3. Plot side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, data in enumerate(results_imgs):
        axes[i].imshow(data["img"])
        axes[i].set_title(data["title"], fontsize=14, pad=10)
        axes[i].axis('off')
        axes[i].text(0.5, -0.1, f"Detections: {data['detections']}", 
                     transform=axes[i].transAxes, fontsize=12, 
                     ha='center', va='top', 
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
    plt.tight_layout()
    out_path = os.path.join(out_dir, "visual_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Visual results saved to {out_path}")

if __name__ == "__main__":
    create_visual_results()
