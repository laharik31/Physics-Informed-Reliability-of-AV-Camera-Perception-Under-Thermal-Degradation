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
        
        Three degradation channels, matching real foggy-lens optics:
          1. Mie scattering blur — applied globally via PSF convolution.
             Even partial droplet coverage scatters light across the full
             image plane. Effective sigma scales with sqrt(C) since
             scattering power grows with the number of scatterers.
          2. Beer-Lambert transmittance (tau) — darkens the image.
          3. Veiling glare — scattered light adds a uniform bright haze
             on top of the darkened image, simulating the milky/foggy
             appearance of real condensation on glass.
        
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
        
        # 2. Apply Mie scattering blur (global PSF convolution)
        #    Forward-scattered light from droplets spreads across the
        #    entire image plane — even 25% coverage fogs the whole view.
        #    Effective sigma scales with sqrt(C): more droplets = stronger blur.
        #    When C=0, effective_sigma=0 → no blur applied.
        effective_sigma = sigma * np.sqrt(coverage)
        
        if effective_sigma > 0.5:
            ksize = int(2 * np.ceil(3 * effective_sigma) + 1)
            if ksize % 2 == 0:
                ksize += 1
            img_blurred = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=effective_sigma)
        else:
            img_blurred = img.copy()
        
        # 3. Apply Beer-Lambert transmittance (attenuation)
        img_dark = img_blurred.astype(np.float32) * tau
        
        # 4. Add veiling glare (scattered light haze)
        #    When light is scattered by droplets, some of it becomes a
        #    uniform bright "veil" overlaid on the image. This creates
        #    the washed-out, milky look of real fogged glass.
        #    Glare intensity = fraction of lost light that becomes haze.
        glare_fraction = 0.35  # 35% of scattered light becomes haze
        glare_intensity = (1.0 - tau) * glare_fraction * 255.0
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
    
