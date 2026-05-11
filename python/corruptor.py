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
        
    def corrupt_image(self, img, t_s, delta_t_c=5, rh=0.80, surface="Untreated glass"):
        """
        Applies physics-informed optical degradation to an image.
        
        The two degradation channels are:
          1. Mie scattering blur (sigma) — weighted by coverage C(t).
             When C=0 (no droplets), no blur is applied even if sigma>0.
          2. Beer-Lambert transmittance (tau) — darkens the image.
        
        Args:
            img (np.ndarray): BGR image loaded via cv2.imread
            t_s (float): Time in seconds
            delta_t_c (float): Delta T
            rh (float): Relative humidity
            surface (str): Surface treatment
            
        Returns:
            np.ndarray: Corrupted image
        """
        # 1. Fetch parameters (tau, sigma, AND coverage C)
        tau, sigma, coverage = self.lookup.get_optical_params(t_s, delta_t_c, rh, surface)
        
        # 2. Apply Mie scattering blur, weighted by coverage C(t)
        #    The blur kernel represents per-droplet scattering, but only the
        #    fraction C of the lens surface is covered by droplets. We blend:
        #      corrupted = C * blurred_img + (1-C) * original_img
        #    This way, when C=0 (no condensation), the image is perfectly sharp.
        if sigma > 0.1 and coverage > 1e-4:
            ksize = int(2 * np.ceil(2 * sigma) + 1)
            img_blurred = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma)
            # Blend based on coverage fraction
            img_blended = (coverage * img_blurred.astype(np.float32) +
                           (1.0 - coverage) * img.astype(np.float32))
            img_blended = np.clip(img_blended, 0, 255).astype(np.uint8)
        else:
            img_blended = img.copy()
            
        # 3. Apply Beer-Lambert transmittance (attenuation)
        img_corrupted = img_blended.astype(np.float32) * tau
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
    
