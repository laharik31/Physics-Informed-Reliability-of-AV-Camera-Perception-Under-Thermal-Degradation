import cv2
import numpy as np
from pathlib import Path

class OpticalCorruptor:
    def __init__(self, lookup):
        """
        Args:
            lookup (PhysicsLookup): Initialized PhysicsLookup instance.
        """
        self.lookup = lookup
        self.mask_cache = {}
        
    def generate_spatial_mask(self, H, W, coverage, t_s):
        """
        Generates a 2D spatial mask for patchy condensation with a radial bias.
        Values range [0, 1] representing localized condensation intensity.
        """
        if coverage <= 0.0:
            return np.zeros((H, W), dtype=np.float32)
        if coverage >= 1.0:
            return np.ones((H, W), dtype=np.float32)
            
        # 1. Low-frequency noise (32x32)
        noise = np.random.rand(32, 32).astype(np.float32)
        noise_up = cv2.resize(noise, (W, H), interpolation=cv2.INTER_CUBIC)
        
        # 2. Radial Bias
        # Create coordinate grids
        y, x = np.ogrid[:H, :W]
        center_y, center_x = H / 2, W / 2
        # Calculate distance from center, normalized to [0, 1] at the corners
        max_dist = np.sqrt(center_y**2 + center_x**2)
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2) / max_dist
        
        # Combine noise with radial bias.
        # Higher distance = more likely to fog.
        # Weight of radial bias is 0.6 to keep noise relevant but strongly biased.
        combined_field = noise_up + 0.6 * dist_from_center
        
        # Normalize coverage against a typical peak equilibrium coverage (~0.30 for Untreated glass)
        C_norm = min(1.0, coverage / 0.30)
        
        # Artificial time-delay for the spatial spread to match visual and physical expectations
        # (The center of the lens takes longer to cool down, so fog creeps from the edges inward)
        if t_s <= 180:
            time_factor = t_s / 180.0
            C_norm = C_norm * time_factor
        
        # 3. Thresholding to match normalized coverage C_norm
        threshold = np.percentile(combined_field, (1.0 - C_norm) * 100)
        
        # Create a soft mask using a sigmoid transition
        mask = 1.0 / (1.0 + np.exp(-15.0 * (combined_field - threshold)))
        
        return mask.astype(np.float32)

    def corrupt_image(self, img, t_s, delta_t_c=5, rh=0.80, surface="Untreated glass", mode="patchy"):
        """
        Applies physics-informed optical degradation to an image.
        """
        # 1. Fetch parameters (tau, sigma, AND coverage C)
        tau, sigma, coverage = self.lookup.get_optical_params(t_s, delta_t_c, rh, surface)
        
        # Override coverage for smooth progression as requested
        # Apply C = 0.21, 0.25, 0.28 for both Untreated glass and Hydrophilic coat
        if surface in ("Untreated glass", "Hydrophilic coat"):
            if t_s == 60:
                coverage = 0.21
            elif t_s == 120:
                coverage = 0.25
            elif t_s == 180:
                coverage = 0.28
                
        H, W = img.shape[:2]
        
        # Generate spatial mask based on surface type and mode
        cache_key = (surface, t_s, H, W, mode)
        if cache_key in self.mask_cache:
            mask = self.mask_cache[cache_key]
        else:
            if mode == "uniform" or surface == "Hydrophilic coat":
                mask = np.ones((H, W), dtype=np.float32)
            elif coverage == 0.0:
                mask = np.zeros((H, W), dtype=np.float32)
            else:
                mask = self.generate_spatial_mask(H, W, coverage, t_s)
            self.mask_cache[cache_key] = mask
            
        mask_3c = np.expand_dims(mask, axis=-1)
        
        # 2. Apply Mie scattering blur
        effective_sigma = sigma * np.sqrt(coverage) if coverage > 0 else 0
        if effective_sigma > 0.5:
            ksize = int(2 * np.ceil(3 * effective_sigma) + 1)
            if ksize % 2 == 0:
                ksize += 1
            img_blurred_full = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=effective_sigma)
        else:
            img_blurred_full = img.astype(np.float32)
            
        # Blend sharp and blurred images based on spatial mask
        img_float = img.astype(np.float32)
        img_blurred_local = img_float * (1.0 - mask_3c) + img_blurred_full.astype(np.float32) * mask_3c
        
        # 3. Apply Beer-Lambert transmittance (attenuation) spatially
        tau_local = 1.0 - mask_3c * (1.0 - tau)
        img_dark = img_blurred_local * tau_local
        
        # 4. Add veiling glare (scattered light haze) spatially
        glare_fraction = 0.35  # 35% of scattered light becomes haze
        glare_intensity = (1.0 - tau_local) * glare_fraction * 255.0
        img_corrupted = img_dark + glare_intensity
        
        img_corrupted = np.clip(img_corrupted, 0, 255).astype(np.uint8)
        
        return img_corrupted

if __name__ == "__main__":
    from lookup import PhysicsLookup
    
    lookup = PhysicsLookup()
    corruptor = OpticalCorruptor(lookup)
    
    # Create a dummy white image to test
    dummy_img = np.ones((512, 512, 3), dtype=np.uint8) * 255
    
    # Apply degradation at peak fog (t=180s)
    corrupted_img = corruptor.corrupt_image(dummy_img, t_s=180, delta_t_c=5, rh=0.60)
    
    mean_val = np.mean(corrupted_img)
    print(f"Test on dummy image at t=180s.")
    print(f"Original mean: 255.0")
    print(f"Corrupted mean: {mean_val:.2f}")
