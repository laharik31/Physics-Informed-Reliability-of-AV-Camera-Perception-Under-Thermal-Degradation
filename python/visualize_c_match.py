"""
Generates side-by-side comparison grids showing how the SAME
coverage fraction C looks under Uniform vs Patchy condensation
for Untreated Glass and Hydrophilic Coat.

This script bypasses the time-based physics lookup and directly
forces specific C values so the comparison is perfectly controlled.
"""
import cv2
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from lookup import PhysicsLookup
from corruptor import OpticalCorruptor


def corrupt_with_forced_c(corruptor, img, tau, sigma, coverage, mode):
    """
    Apply corruption with explicitly forced tau, sigma, and C values,
    bypassing the physics lookup entirely.
    """
    H, W = img.shape[:2]

    # Generate spatial mask based on mode
    if mode == "uniform":
        mask = np.ones((H, W), dtype=np.float32)
    else:
        # Patchy: use radial-biased noise mask
        if coverage <= 0.0:
            mask = np.zeros((H, W), dtype=np.float32)
        elif coverage >= 1.0:
            mask = np.ones((H, W), dtype=np.float32)
        else:
            # Reproduce the corruptor's spatial mask logic
            np.random.seed(42)  # Deterministic for reproducibility
            noise = np.random.rand(32, 32).astype(np.float32)
            noise_up = cv2.resize(noise, (W, H), interpolation=cv2.INTER_CUBIC)

            y, x = np.ogrid[:H, :W]
            center_y, center_x = H / 2, W / 2
            max_dist = np.sqrt(center_y**2 + center_x**2)
            dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2) / max_dist

            combined_field = noise_up + 0.6 * dist_from_center
            C_norm = min(1.0, coverage / 0.30)

            threshold = np.percentile(combined_field, (1.0 - C_norm) * 100)
            mask = 1.0 / (1.0 + np.exp(-15.0 * (combined_field - threshold)))
            mask = mask.astype(np.float32)

    mask_3c = np.expand_dims(mask, axis=-1)

    # Apply Mie scattering blur
    effective_sigma = sigma * np.sqrt(coverage) if coverage > 0 else 0
    if effective_sigma > 0.5:
        ksize = int(2 * np.ceil(3 * effective_sigma) + 1)
        if ksize % 2 == 0:
            ksize += 1
        img_blurred_full = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=effective_sigma)
    else:
        img_blurred_full = img.astype(np.float32)

    img_float = img.astype(np.float32)
    img_blurred_local = img_float * (1.0 - mask_3c) + img_blurred_full.astype(np.float32) * mask_3c

    # Beer-Lambert transmittance
    tau_local = 1.0 - mask_3c * (1.0 - tau)
    img_dark = img_blurred_local * tau_local

    # Veiling glare
    glare_fraction = 0.35
    glare_intensity = (1.0 - tau_local) * glare_fraction * 255.0
    img_corrupted = img_dark + glare_intensity

    return np.clip(img_corrupted, 0, 255).astype(np.uint8)


def generate_c_comparison(surface_name, c_values, tau_values, sigma=8.0):
    """
    Generate a comparison grid for a given surface type.

    For each C value, produce two images side-by-side:
      Left = Uniform condensation at that C
      Right = Patchy condensation at that C

    Args:
        surface_name: Name for labeling (e.g., "Untreated Glass")
        c_values: List of coverage fractions to compare
        tau_values: Corresponding transmittance values for each C
        sigma: Mie scattering blur width (constant)
    """
    lookup = PhysicsLookup()
    corruptor = OpticalCorruptor(lookup)

    # Find a sample image
    image_paths = sorted(glob.glob("data/bdd100k/images/val/*.jpg"))
    if not image_paths:
        image_paths = sorted(glob.glob("data/sample_images/*.jpg"))
    if not image_paths:
        print("No images found!")
        return

    img = cv2.imread(image_paths[5])
    img = cv2.resize(img, (640, 480))

    n_c = len(c_values)
    fig, axes = plt.subplots(n_c, 2, figsize=(14, 5 * n_c))

    if n_c == 1:
        axes = axes.reshape(1, -1)

    for i, (c_val, tau_val) in enumerate(zip(c_values, tau_values)):
        # Uniform
        img_uniform = corrupt_with_forced_c(corruptor, img, tau_val, sigma, c_val, mode="uniform")
        img_uniform_rgb = cv2.cvtColor(img_uniform, cv2.COLOR_BGR2RGB)

        # Patchy
        img_patchy = corrupt_with_forced_c(corruptor, img, tau_val, sigma, c_val, mode="patchy")
        img_patchy_rgb = cv2.cvtColor(img_patchy, cv2.COLOR_BGR2RGB)

        axes[i, 0].imshow(img_uniform_rgb)
        axes[i, 0].set_title(f"Uniform  |  C={c_val:.2f}, τ={tau_val:.2f}", fontsize=13, fontweight='bold')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(img_patchy_rgb)
        axes[i, 1].set_title(f"Patchy  |  C={c_val:.2f}, τ={tau_val:.2f}", fontsize=13, fontweight='bold')
        axes[i, 1].axis('off')

    fig.suptitle(f"{surface_name}: Uniform vs Patchy at Matched Coverage (C)",
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()

    safe_name = surface_name.replace(" ", "_").lower()
    out_path = f"results/c_value_comparison_{safe_name}.png"
    os.makedirs("results", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    # --- Untreated Glass ---
    # Using the patchy override C values: 0.21, 0.25, 0.28
    # With representative tau values from the physics lookup at RH=80%
    print("Generating C-value comparison for Untreated Glass...")
    generate_c_comparison(
        surface_name="Untreated Glass",
        c_values=[0.21, 0.25, 0.28],
        tau_values=[0.38, 0.44, 0.48],  # approximate tau at t=60,120,180 for RH=80%
        sigma=8.0
    )

    # --- Hydrophilic Coat ---
    # Using the same C values for a controlled comparison
    # Hydrophilic has higher tau (less light loss) at same coverage
    print("Generating C-value comparison for Hydrophilic Coat...")
    generate_c_comparison(
        surface_name="Hydrophilic Coat",
        c_values=[0.21, 0.25, 0.28],
        tau_values=[0.66, 0.70, 0.73],  # approximate tau at t=60,120,180 for RH=80%
        sigma=8.0
    )

    print("\nDone! C-value comparison grids generated.")
