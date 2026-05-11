import cv2
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from corruptor import OpticalCorruptor
from lookup import PhysicsLookup

def run_experiment():
    # 1. Setup
    print("Loading YOLOv8n model...")
    model = YOLO('yolov8n.pt') # Will auto-download the weights if not present
    
    lookup = PhysicsLookup()
    corruptor = OpticalCorruptor(lookup)
    
    image_paths = glob.glob("data/sample_images/*.jpg")
    if not image_paths:
        print("No sample images found! Run download_sample.py first.")
        return
        
    print(f"Found {len(image_paths)} images.")
    
    # 2. Define snapshots (times in seconds)
    t_snapshots = [0, 60, 120, 180, 300, 450, 600]
    
    # Environmental variables (matches Kim/Hendrycks fog calibration)
    delta_t = 5
    rh = 0.80
    surface = "Untreated glass"
    
    results_map = []
    
    # 3. Evaluation Loop
    for t in t_snapshots:
        print(f"--- Evaluating Snapshot t = {t}s ---")
        
        # To simulate mAP drop, we run YOLO and look at average confidence of detections.
        # Since we don't have Ground Truth for the random sample images to calculate pure mAP,
        # we track the sum of confidence scores or average confidence as a proxy for detection robustness.
        total_conf = 0.0
        total_boxes = 0
        
        for img_path in image_paths:
            img = cv2.imread(img_path)
            
            # Corrupt image based on time snapshot
            corrupted_img = corruptor.corrupt_image(img, t_s=t, delta_t_c=delta_t, rh=rh, surface=surface)
            
            # Run YOLOv8
            results = model(corrupted_img, verbose=False)
            
            # Extract confidences for boxes
            for r in results:
                boxes = r.boxes
                if len(boxes) > 0:
                    conf = boxes.conf.cpu().numpy()
                    total_conf += np.sum(conf)
                    total_boxes += len(conf)
                    
        # Calculate proxy mAP (Average Confidence across all detected boxes in the dataset)
        avg_conf = (total_conf / total_boxes) if total_boxes > 0 else 0
        results_map.append(avg_conf)
        print(f"  Proxy mAP (Avg Conf) = {avg_conf:.4f}")
        
    # 4. Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(t_snapshots, results_map, marker='o', linestyle='-', color='b', linewidth=2)
    
    # Highlight the SOTIF threshold if it exists
    # If starting proxy mAP is, say, 0.65, we draw a line where it drops.
    plt.title(f'YOLOv8 Detection Robustness under Thermal Degradation\n(RH={rh*100}%, DeltaT={delta_t}°C, {surface})')
    plt.xlabel('Time (s)')
    plt.ylabel('Proxy mAP (Average Confidence)')
    plt.grid(True)
    
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "availability_curve.png")
    plt.savefig(out_path, dpi=300)
    print(f"\nSaved availability curve plot to {out_path}")

if __name__ == "__main__":
    run_experiment()
