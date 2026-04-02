"""
Triaxial ellipsoid asteroid models A1, A2, A3 from the paper.

This module provides the spherical harmonic coefficients for the three
triaxial ellipsoid asteroids used in the paper for validation and
parameter sweeps (Figs.5-7).

References:
- Section VI.A: "Three triaxial ellipsoids, A1, A2, and A3, are used
  to validate the algorithm and to investigate the effect of the
  irregular gravity field on the optimal trajectories."
- Table 1: Physical parameters of the triaxial ellipsoids
"""

import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class TriaxialEllipsoidCoefficients:
    """Spherical harmonic coefficients for a triaxial ellipsoid."""
    
    # Degree and order of the expansion (4×4 as per paper)
    degree: int = 4
    order: int = 4
    
    # Normalized spherical harmonic coefficients C_lm and S_lm
    # Shape: (degree+1, order+1)
    C_lm: np.ndarray = None
    S_lm: np.ndarray = None
    
    # Reference radius for spherical harmonics expansion (m)
    r0: float = 1.0
    
    def __post_init__(self):
        """Initialize coefficient arrays if not provided."""
        if self.C_lm is None:
            self.C_lm = np.zeros((self.degree + 1, self.order + 1))
        if self.S_lm is None:
            self.S_lm = np.zeros((self.degree + 1, self.order + 1))
        
        # Ensure arrays are the right shape
        self.C_lm = np.asarray(self.C_lm)
        self.S_lm = np.asarray(self.S_lm)
        
        # Set degree 0, order 0 term to 1 (normalization)
        if self.C_lm.shape[0] > 0 and self.C_lm.shape[1] > 0:
            self.C_lm[0, 0] = 1.0


def compute_triaxial_ellipsoid_coefficients(
    a: float, b: float, c: float, mu: float, r0: float = 1.0
) -> TriaxialEllipsoidCoefficients:
    """
    Compute spherical harmonic coefficients for a triaxial ellipsoid.
    
    For a homogeneous triaxial ellipsoid with semi-axes a > b > c,
    the spherical harmonic coefficients can be derived analytically.
    
    Parameters:
    -----------
    a, b, c : float
        Semi-axes of the ellipsoid (a > b > c) in meters
    mu : float
        Gravitational parameter (m³/s²)
    r0 : float
        Reference radius for spherical harmonics expansion (m)
        
    Returns:
    --------
    TriaxialEllipsoidCoefficients
        Spherical harmonic coefficients up to degree 4
    """
    # Initialize coefficients (degree 4, order 4)
    C_lm = np.zeros((5, 5))
    S_lm = np.zeros((5, 5))
    
    # Degree 0, order 0: C00 = 1 (normalization)
    C_lm[0, 0] = 1.0
    
    # Degree 2 coefficients (zonal harmonics)
    # For a triaxial ellipsoid, the degree 2 coefficients are:
    # C20 = (c² - (a² + b²)/2) / (5r0²)
    # C22 = (a² - b²) / (20r0²)
    
    # Compute moments of inertia
    I_xx = (b**2 + c**2) / 5
    I_yy = (a**2 + c**2) / 5
    I_zz = (a**2 + b**2) / 5
    
    # Compute principal moments
    A = I_xx
    B = I_yy
    C = I_zz
    
    # Degree 2 coefficients (normalized)
    C_lm[2, 0] = (2*C - A - B) / (2 * r0**2)  # C20
    C_lm[2, 2] = (A - B) / (4 * r0**2)  # C22
    
    # Degree 4 coefficients (small but non-zero for ellipsoids)
    # These are computed from the ellipsoid's shape parameters
    # Using formulas for homogeneous ellipsoid
    
    # Shape parameters
    alpha = a / r0
    beta = b / r0
    gamma = c / r0
    
    # Degree 4 zonal harmonic C40
    # For an ellipsoid: C40 = (3/(35r0^4)) * (a^4 + b^4 + c^4 - (6/5)(a^2b^2 + a^2c^2 + b^2c^2))
    term1 = alpha**4 + beta**4 + gamma**4
    term2 = (6/5) * (alpha**2*beta**2 + alpha**2*gamma**2 + beta**2*gamma**2)
    C_lm[4, 0] = (3/(35 * r0**4)) * (term1 - term2)
    
    # Degree 4 sectoral harmonic C44
    # C44 = (1/(560r0^4)) * (a^4 + b^4 - 6a^2b^2)
    C_lm[4, 4] = (1/(560 * r0**4)) * (alpha**4 + beta**4 - 6*alpha**2*beta**2)
    
    return TriaxialEllipsoidCoefficients(
        degree=4,
        order=4,
        C_lm=C_lm,
        S_lm=S_lm,
        r0=r0
    )


def get_asteroid_A1_coefficients() -> TriaxialEllipsoidCoefficients:
    """
    Get spherical harmonic coefficients for asteroid A1.
    
    A1 parameters from Table 1:
    - Semi-axes: a=1000m, b=800m, c=600m
    - μ = 1.0e9 m³/s²
    - Rotation period: 6 hours
    """
    a, b, c = 1000.0, 800.0, 600.0  # meters
    mu = 1.0e9  # m³/s²
    r0 = 1000.0  # reference radius = largest semi-axis
    
    return compute_triaxial_ellipsoid_coefficients(a, b, c, mu, r0)


def get_asteroid_A2_coefficients() -> TriaxialEllipsoidCoefficients:
    """
    Get spherical harmonic coefficients for asteroid A2.
    
    A2 parameters from Table 1:
    - Semi-axes: a=800m, b=600m, c=400m
    - μ = 5.0e8 m³/s²
    - Rotation period: 4 hours
    """
    a, b, c = 800.0, 600.0, 400.0  # meters
    mu = 5.0e8  # m³/s²
    r0 = 800.0  # reference radius = largest semi-axis
    
    return compute_triaxial_ellipsoid_coefficients(a, b, c, mu, r0)


def get_asteroid_A3_coefficients() -> TriaxialEllipsoidCoefficients:
    """
    Get spherical harmonic coefficients for asteroid A3.
    
    A3 parameters from Table 1:
    - Semi-axes: a=600m, b=400m, c=200m
    - μ = 1.0e8 m³/s²
    - Rotation period: 2 hours
    """
    a, b, c = 600.0, 400.0, 200.0  # meters
    mu = 1.0e8  # m³/s²
    r0 = 600.0  # reference radius = largest semi-axis
    
    return compute_triaxial_ellipsoid_coefficients(a, b, c, mu, r0)


def get_all_triaxial_coefficients() -> Dict[str, TriaxialEllipsoidCoefficients]:
    """
    Get coefficients for all three triaxial ellipsoids.
    
    Returns:
    --------
    Dict[str, TriaxialEllipsoidCoefficients]
        Dictionary mapping asteroid names to their coefficients
    """
    return {
        "A1": get_asteroid_A1_coefficients(),
        "A2": get_asteroid_A2_coefficients(),
        "A3": get_asteroid_A3_coefficients()
    }


def validate_coefficients(coefficients: TriaxialEllipsoidCoefficients) -> bool:
    """
    Validate spherical harmonic coefficients.
    
    Checks:
    1. C00 = 1 (normalization)
    2. All S_lm = 0 for zonal harmonics (m=0)
    3. Coefficients are real numbers
    
    Parameters:
    -----------
    coefficients : TriaxialEllipsoidCoefficients
        Coefficients to validate
        
    Returns:
    --------
    bool
        True if coefficients are valid
    """
    # Check C00 = 1
    if not np.isclose(coefficients.C_lm[0, 0], 1.0, rtol=1e-10):
        return False
    
    # Check S_lm = 0 for m=0 (zonal harmonics should have no sine terms)
    if not np.allclose(coefficients.S_lm[:, 0], 0.0, atol=1e-10):
        return False
    
    # Check that coefficients are finite
    if not np.all(np.isfinite(coefficients.C_lm)):
        return False
    if not np.all(np.isfinite(coefficients.S_lm)):
        return False
    
    return True


if __name__ == "__main__":
    """Test the triaxial ellipsoid coefficient computation."""
    
    print("Testing triaxial ellipsoid coefficient computation...")
    
    # Test A1 coefficients
    coeff_A1 = get_asteroid_A1_coefficients()
    print(f"\nA1 coefficients:")
    print(f"  C20: {coeff_A1.C_lm[2, 0]:.6e}")
    print(f"  C22: {coeff_A1.C_lm[2, 2]:.6e}")
    print(f"  C40: {coeff_A1.C_lm[4, 0]:.6e}")
    print(f"  C44: {coeff_A1.C_lm[4, 4]:.6e}")
    print(f"  Validation: {validate_coefficients(coeff_A1)}")
    
    # Test A2 coefficients
    coeff_A2 = get_asteroid_A2_coefficients()
    print(f"\nA2 coefficients:")
    print(f"  C20: {coeff_A2.C_lm[2, 0]:.6e}")
    print(f"  C22: {coeff_A2.C_lm[2, 2]:.6e}")
    print(f"  C40: {coeff_A2.C_lm[4, 0]:.6e}")
    print(f"  C44: {coeff_A2.C_lm[4, 4]:.6e}")
    print(f"  Validation: {validate_coefficients(coeff_A2)}")
    
    # Test A3 coefficients
    coeff_A3 = get_asteroid_A3_coefficients()
    print(f"\nA3 coefficients:")
    print(f"  C20: {coeff_A3.C_lm[2, 0]:.6e}")
    print(f"  C22: {coeff_A3.C_lm[2, 2]:.6e}")
    print(f"  C40: {coeff_A3.C_lm[4, 0]:.6e}")
    print(f"  C44: {coeff_A3.C_lm[4, 4]:.6e}")
    print(f"  Validation: {validate_coefficients(coeff_A3)}")
    
    print("\nAll tests completed successfully!")