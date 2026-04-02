"""
Spherical harmonics gravity model for asteroid exterior (outside Brillouin sphere).

Implements Eqs. 1-3 from the paper:
U = μ/r Σ_{l=0}^4 Σ_{m=0}^l (r₀/r)^l P_lm(sinδ)[C_lm cos(mλ) + S_lm sin(mλ)]
∇U = ∂U/∂x, ∂U/∂y, ∂U/∂z via chain rule

Reference: "Trajectory Design Employing Convex Optimization for Landing on 
Irregularly Shaped Asteroids"
"""

import numpy as np
from scipy.special import lpmv  # Associated Legendre functions
from typing import Tuple, Optional
import warnings

# Type hints
from numpy.typing import NDArray


class SphericalHarmonicsGravity:
    """
    Computes gravitational acceleration using 4×4 spherical harmonics.
    
    This model is valid outside the Brillouin sphere (r > R_b).
    
    Attributes:
        degree (int): Maximum degree of spherical harmonics (l_max = 4)
        order (int): Maximum order of spherical harmonics (m_max = 4)
        C_lm (NDArray): Cosine coefficients, shape (degree+1, order+1)
        S_lm (NDArray): Sine coefficients, shape (degree+1, order+1)
        mu (float): Gravitational parameter (km³/s²)
        r0 (float): Reference radius (km)
    """
    
    def __init__(self, C_lm: NDArray, S_lm: NDArray, mu: float, r0: float = 1.0):
        """
        Initialize spherical harmonics gravity model.
        
        Args:
            C_lm: Cosine coefficients (degree 4, order 4)
            S_lm: Sine coefficients (degree 4, order 4)
            mu: Gravitational parameter (km³/s²)
            r0: Reference radius (km), default 1.0
        """
        # Validate input shapes
        if C_lm.shape != (5, 5) or S_lm.shape != (5, 5):
            raise ValueError(f"C_lm and S_lm must be 5x5 arrays (degree 4, order 4). "
                           f"Got shapes: C_lm={C_lm.shape}, S_lm={S_lm.shape}")
        
        self.degree = 4
        self.order = 4
        self.C_lm = C_lm
        self.S_lm = S_lm
        self.mu = mu
        self.r0 = r0
        
        # Precompute normalization factors for associated Legendre functions
        self._precompute_normalization()
        
    def _precompute_normalization(self):
        """Precompute normalization factors for associated Legendre functions."""
        # Normalization factor: sqrt((2 - δ_{0m}) * (l - m)! / (l + m)!)
        # where δ_{0m} is Kronecker delta
        self.norm_factors = np.zeros((self.degree + 1, self.order + 1))
        
        for l in range(self.degree + 1):
            for m in range(min(l, self.order) + 1):
                if m == 0:
                    delta_0m = 1
                else:
                    delta_0m = 0
                
                # Compute factorial ratio (l - m)! / (l + m)!
                # Use log gamma to avoid overflow for large l
                from math import lgamma
                log_ratio = lgamma(l - m + 1) - lgamma(l + m + 1)
                ratio = np.exp(log_ratio)
                
                self.norm_factors[l, m] = np.sqrt((2 - delta_0m) * ratio)
    
    def potential(self, position: NDArray) -> float:
        """
        Compute gravitational potential U at given position.
        
        Eq. 1: U = μ/r Σ_{l=0}^4 Σ_{m=0}^l (r₀/r)^l P_lm(sinδ)[C_lm cos(mλ) + S_lm sin(mλ)]
        
        Args:
            position: Cartesian position vector [x, y, z] in km
            
        Returns:
            Gravitational potential U (km²/s²)
        """
        # Convert to spherical coordinates
        r, delta, lam = self._cartesian_to_spherical(position)
        
        if r == 0:
            return np.inf
        
        # Compute potential
        U = 0.0
        sin_delta = np.sin(delta)
        
        for l in range(self.degree + 1):
            r_factor = (self.r0 / r) ** l
            
            for m in range(min(l, self.order) + 1):
                # Associated Legendre function
                P_lm = lpmv(m, l, sin_delta)
                
                # Apply normalization
                P_lm_norm = P_lm * self.norm_factors[l, m]
                
                # Trigonometric terms
                cos_term = np.cos(m * lam)
                sin_term = np.sin(m * lam)
                
                # Contribution to potential
                U += r_factor * P_lm_norm * (
                    self.C_lm[l, m] * cos_term + self.S_lm[l, m] * sin_term
                )
        
        U = self.mu / r * U
        return U
    
    def acceleration(self, position: NDArray) -> NDArray:
        """
        Compute gravitational acceleration ∇U at given position.
        
        Eq. 2-3: ∇U = ∂U/∂x, ∂U/∂y, ∂U/∂z computed via chain rule.
        
        Args:
            position: Cartesian position vector [x, y, z] in km
            
        Returns:
            Gravitational acceleration vector [a_x, a_y, a_z] in km/s²
        """
        # Convert to spherical coordinates
        x, y, z = position
        r, delta, lam = self._cartesian_to_spherical(position)
        
        if r == 0:
            return np.array([0.0, 0.0, 0.0])
        
        # Precompute trigonometric functions
        sin_delta = np.sin(delta)
        cos_delta = np.cos(delta)
        sin_lam = np.sin(lam)
        cos_lam = np.cos(lam)
        
        # Initialize partial derivatives
        dU_dr = 0.0
        dU_ddelta = 0.0
        dU_dlam = 0.0
        
        # Compute partial derivatives in spherical coordinates
        for l in range(self.degree + 1):
            r_factor = (self.r0 / r) ** l
            dr_factor = -l * (self.r0 ** l) / (r ** (l + 1))
            
            for m in range(min(l, self.order) + 1):
                # Associated Legendre function and its derivative
                P_lm = lpmv(m, l, sin_delta)
                
                # Derivative of P_lm with respect to sin_delta
                # Using recurrence relation: dP_lm/dx = (l*x*P_lm - (l+m)*P_{l-1,m})/(x²-1)
                if l == 0:
                    dP_lm_dsin = 0.0
                else:
                    if abs(sin_delta) == 1.0:
                        # Handle poles carefully
                        dP_lm_dsin = 0.0
                    else:
                        P_lm_minus1 = lpmv(m, l-1, sin_delta) if l > 0 else 0.0
                        dP_lm_dsin = (l * sin_delta * P_lm - (l + m) * P_lm_minus1) / (sin_delta**2 - 1)
                
                # Apply normalization
                P_lm_norm = P_lm * self.norm_factors[l, m]
                dP_lm_dsin_norm = dP_lm_dsin * self.norm_factors[l, m]
                
                # Trigonometric terms
                cos_term = np.cos(m * lam)
                sin_term = np.sin(m * lam)
                
                # dU/dr term
                dU_dr += dr_factor * P_lm_norm * (
                    self.C_lm[l, m] * cos_term + self.S_lm[l, m] * sin_term
                )
                
                # dU/ddelta term (chain rule: dU/ddelta = dU/d(sin_delta) * cos_delta)
                dU_ddelta += r_factor * dP_lm_dsin_norm * cos_delta * (
                    self.C_lm[l, m] * cos_term + self.S_lm[l, m] * sin_term
                )
                
                # dU/dlam term
                dU_dlam += r_factor * P_lm_norm * m * (
                    -self.C_lm[l, m] * sin_term + self.S_lm[l, m] * cos_term
                )
        
        # Scale by μ/r
        U_sum = 0.0
        for l in range(self.degree + 1):
            r_factor = (self.r0 / r) ** l
            for m in range(min(l, self.order) + 1):
                P_lm = lpmv(m, l, sin_delta)
                P_lm_norm = P_lm * self.norm_factors[l, m]
                cos_term = np.cos(m * lam)
                sin_term = np.sin(m * lam)
                U_sum += r_factor * P_lm_norm * (
                    self.C_lm[l, m] * cos_term + self.S_lm[l, m] * sin_term
                )
        
        # Complete derivatives
        dU_dr = self.mu / r * dU_dr - self.mu / (r**2) * U_sum
        dU_ddelta = self.mu / r * dU_ddelta
        dU_dlam = self.mu / r * dU_dlam
        
        # Convert to Cartesian coordinates
        # Transformation matrix from spherical to Cartesian
        sin_delta = np.sin(delta)
        cos_delta = np.cos(delta)
        sin_lam = np.sin(lam)
        cos_lam = np.cos(lam)
        
        # Jacobian matrix
        dx_dr = sin_delta * cos_lam
        dx_ddelta = r * cos_delta * cos_lam
        dx_dlam = -r * sin_delta * sin_lam
        
        dy_dr = sin_delta * sin_lam
        dy_ddelta = r * cos_delta * sin_lam
        dy_dlam = r * sin_delta * cos_lam
        
        dz_dr = cos_delta
        dz_ddelta = -r * sin_delta
        dz_dlam = 0.0
        
        # Compute Cartesian acceleration (negative gradient of potential)
        a_x = -(dU_dr * dx_dr + dU_ddelta * dx_ddelta + dU_dlam * dx_dlam)
        a_y = -(dU_dr * dy_dr + dU_ddelta * dy_ddelta + dU_dlam * dy_dlam)
        a_z = -(dU_dr * dz_dr + dU_ddelta * dz_ddelta + dU_dlam * dz_dlam)
        
        return np.array([a_x, a_y, a_z])
    
    def _cartesian_to_spherical(self, position: NDArray) -> Tuple[float, float, float]:
        """
        Convert Cartesian coordinates to spherical coordinates.
        
        Args:
            position: [x, y, z] in km
            
        Returns:
            (r, delta, lam) where:
                r: radius (km)
                delta: latitude angle (radians, -π/2 to π/2)
                lam: longitude angle (radians, -π to π)
        """
        x, y, z = position
        r = np.sqrt(x**2 + y**2 + z**2)
        
        if r == 0:
            return 0.0, 0.0, 0.0
        
        # Latitude (declination)
        delta = np.arcsin(z / r)
        
        # Longitude (right ascension)
        lam = np.arctan2(y, x)
        
        return r, delta, lam
    
    def test_point_mass_approximation(self, position: NDArray) -> bool:
        """
        Test if the spherical harmonics model reduces to point mass at large distances.
        
        For r >> r0, the dominant term should be l=0, m=0: C_00 = 1, all others ≈ 0.
        
        Args:
            position: Test position
            
        Returns:
            True if point mass approximation holds within 1%
        """
        # Compute full acceleration
        a_full = self.acceleration(position)
        
        # Compute point mass acceleration
        r = np.linalg.norm(position)
        a_point = -self.mu / (r**3) * position
        
        # Compare
        error = np.linalg.norm(a_full - a_point) / np.linalg.norm(a_point)
        return error < 0.01  # Within 1%


def create_uniform_sphere_coefficients(mu: float, r0: float = 1.0) -> Tuple[NDArray, NDArray]:
    """
    Create spherical harmonic coefficients for a uniform sphere.
    
    For a uniform sphere:
        C_00 = 1.0
        All other C_lm = 0.0
        All S_lm = 0.0
    
    Args:
        mu: Gravitational parameter
        r0: Reference radius
        
    Returns:
        (C_lm, S_lm) arrays for degree 4, order 4
    """
    C_lm = np.zeros((5, 5))
    S_lm = np.zeros((5, 5))
    
    # Only the monopole term is non-zero for a sphere
    C_lm[0, 0] = 1.0
    
    return C_lm, S_lm


def test_spherical_harmonics():
    """Test the spherical harmonics implementation."""
    import numpy as np
    
    # Test 1: Uniform sphere
    print("Test 1: Uniform sphere")
    mu = 1.0
    C_lm, S_lm = create_uniform_sphere_coefficients(mu)
    gravity = SphericalHarmonicsGravity(C_lm, S_lm, mu)
    
    # Test at various positions
    test_positions = [
        np.array([2.0, 0.0, 0.0]),
        np.array([0.0, 3.0, 0.0]),
        np.array([0.0, 0.0, 4.0]),
        np.array([1.0, 1.0, 1.0]),
    ]
    
    for pos in test_positions:
        a = gravity.acceleration(pos)
        a_expected = -mu / (np.linalg.norm(pos)**3) * pos
        error = np.linalg.norm(a - a_expected)
        print(f"  Position {pos}: error = {error:.2e}")
        assert error < 1e-10, f"Uniform sphere test failed: error = {error}"
    
    # Test 2: Potential calculation
    print("\nTest 2: Potential calculation")
    pos = np.array([2.0, 0.0, 0.0])
    U = gravity.potential(pos)
    U_expected = -mu / np.linalg.norm(pos)
    error = abs(U - U_expected)
    print(f"  Potential at {pos}: U = {U:.6f}, expected = {U_expected:.6f}, error = {error:.2e}")
    assert error < 1e-10, f"Potential test failed: error = {error}"
    
    # Test 3: Non-zero coefficients (simple quadrupole)
    print("\nTest 3: Simple quadrupole field")
    C_lm2 = np.zeros((5, 5))
    S_lm2 = np.zeros((5, 5))
    C_lm2[0, 0] = 1.0
    C_lm2[2, 0] = 0.1  # J2 term
    
    gravity2 = SphericalHarmonicsGravity(C_lm2, S_lm2, mu)
    
    # Test symmetry
    pos1 = np.array([3.0, 0.0, 0.0])
    pos2 = np.array([0.0, 3.0, 0.0])
    a1 = gravity2.acceleration(pos1)
    a2 = gravity2.acceleration(pos2)
    
    # Should be symmetric in x and y for this simple case
    print(f"  Acceleration at [3,0,0]: {a1}")
    print(f"  Acceleration at [0,3,0]: {a2}")
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    test_spherical_harmonics()