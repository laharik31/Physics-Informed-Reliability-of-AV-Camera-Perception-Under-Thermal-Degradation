import pandas as pd
import numpy as np
import os

class PhysicsLookup:
    def __init__(self, matlab_dir=None):
        if matlab_dir is None:
            matlab_dir = os.path.join(os.path.dirname(__file__), "..", "matlab")
        kernel_path = os.path.join(matlab_dir, "kernel_lookup_W3.csv")
        tau_path = os.path.join(matlab_dir, "tau_lookup_W3.csv")
        
        # Load datasets
        if not os.path.exists(kernel_path) or not os.path.exists(tau_path):
            raise FileNotFoundError(f"Could not find lookup CSVs in {matlab_dir}. Ensure they are in the correct directory.")
            
        self.df_kernel = pd.read_csv(kernel_path)
        self.df_tau = pd.read_csv(tau_path)
        
    def get_optical_params(self, t_s, delta_t_c=5, rh=0.80, surface="Untreated glass"):
        """
        Interpolates optical degradation parameters for a given time and environment.
        
        Args:
            t_s (float): Simulation time in seconds.
            delta_t_c (float): Delta T (subcooling) in Celsius.
            rh (float): Relative humidity (0.0 to 1.0).
            surface (str): Surface treatment type (e.g., 'Untreated glass').
            
        Returns:
            tuple: (tau, sigma_Mie_px)
                - tau: Optical transmittance (0 to 1)
                - sigma_Mie_px: Mie scattering PSF width in pixels
        """
        # 1. Get sigma_Mie_px (only dependent on t_s in this specific kernel lookup)
        # We linearly interpolate based on t_s
        sigma = np.interp(t_s, self.df_kernel['t_s'], self.df_kernel['sigma_Mie_px'])
        
        # 2. Get tau
        # Filter tau DataFrame by environmental parameters
        # Floating point comparisons can be tricky, so we find the closest match
        df_filtered = self.df_tau[
            (self.df_tau['surface'] == surface) & 
            (np.isclose(self.df_tau['DeltaT_C'], delta_t_c, atol=0.1)) & 
            (np.isclose(self.df_tau['RH'], rh, atol=0.01))
        ]
        
        if df_filtered.empty:
            raise ValueError(f"No tau data found for DeltaT_C={delta_t_c}, RH={rh}, surface='{surface}'.")
            
        # Linearly interpolate tau based on t_s
        tau = np.interp(t_s, df_filtered['t_s'], df_filtered['tau'])
        
        return float(tau), float(sigma)

if __name__ == "__main__":
    # Quick test
    lookup = PhysicsLookup()
    tau, sigma = lookup.get_optical_params(180, delta_t_c=5, rh=0.60, surface="Untreated glass")
    print(f"Test at t=180s: tau={tau:.4f}, sigma={sigma:.4f}px")
