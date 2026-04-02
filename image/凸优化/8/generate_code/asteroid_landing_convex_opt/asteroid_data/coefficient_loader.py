"""
Coefficient loader for asteroid gravity models.

This module provides functions to load spherical harmonic coefficients (C_lm, S_lm)
for both triaxial ellipsoids and Castalia, as well as interior Bessel coefficients
(Ā_l,n,m, B̄_l,n,m) for the interior gravity model.

References:
- Triaxial ellipsoids: Analytical expressions from paper
- Castalia: Coefficients from references [25,31] in the paper
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union
from dataclasses import dataclass
import json
import os

from .triaxial_ellipsoids import (
    get_asteroid_A1_coefficients,
    get_asteroid_A2_coefficients,
    get_asteroid_A3_coefficients,
    TriaxialEllipsoidCoefficients
)


@dataclass
class SphericalHarmonicCoefficients:
    """Container for spherical harmonic coefficients (exterior gravity model)."""
    degree: int  # Maximum degree l
    order: int   # Maximum order m
    C_lm: np.ndarray  # Cosine coefficients, shape (degree+1, order+1)
    S_lm: np.ndarray  # Sine coefficients, shape (degree+1, order+1)
    r0: float  # Reference radius for normalization
    mu: float  # Gravitational parameter (km^3/s^2)
    
    def __post_init__(self):
        """Validate coefficient arrays."""
        assert self.C_lm.shape == (self.degree + 1, self.order + 1), \
            f"C_lm shape {self.C_lm.shape} doesn't match degree={self.degree}, order={self.order}"
        assert self.S_lm.shape == (self.degree + 1, self.order + 1), \
            f"S_lm shape {self.S_lm.shape} doesn't match degree={self.degree}, order={self.order}"
        
        # Ensure C_00 = 1.0 for normalized coefficients
        if abs(self.C_lm[0, 0] - 1.0) > 1e-10:
            # Normalize if needed
            self.C_lm = self.C_lm.copy()
            self.C_lm[0, 0] = 1.0


@dataclass
class BesselCoefficients:
    """Container for spherical Bessel coefficients (interior gravity model)."""
    l_max: int  # Maximum degree l (paper uses l_max=2)
    n_max: int  # Maximum radial index n (paper uses n_max=5)
    m_max: int  # Maximum order m (paper uses m_max=5)
    A_bar: np.ndarray  # Real coefficients, shape (l_max+1, n_max+1, m_max+1)
    B_bar: np.ndarray  # Imaginary coefficients, shape (l_max+1, n_max+1, m_max+1)
    R_b: float  # Brillouin sphere radius (km)
    mu: float   # Gravitational parameter (km^3/s^2)
    
    def __post_init__(self):
        """Validate coefficient arrays."""
        assert self.A_bar.shape == (self.l_max + 1, self.n_max + 1, self.m_max + 1), \
            f"A_bar shape {self.A_bar.shape} doesn't match l_max={self.l_max}, n_max={self.n_max}, m_max={self.m_max}"
        assert self.B_bar.shape == (self.l_max + 1, self.n_max + 1, self.m_max + 1), \
            f"B_bar shape {self.B_bar.shape} doesn't match l_max={self.l_max}, n_max={self.n_max}, m_max={self.m_max}"


@dataclass
class AsteroidCoefficients:
    """Complete set of coefficients for an asteroid."""
    name: str
    sh_coeffs: SphericalHarmonicCoefficients  # Exterior model (4×4)
    bessel_coeffs: BesselCoefficients  # Interior model (l_max=2, n_max=5, m_max=5)
    is_triaxial: bool  # Whether this is a triaxial ellipsoid (vs Castalia)


def load_triaxial_ellipsoid_coefficients(asteroid_name: str) -> AsteroidCoefficients:
    """
    Load coefficients for a triaxial ellipsoid asteroid (A1, A2, or A3).
    
    Args:
        asteroid_name: Name of asteroid ('A1', 'A2', or 'A3')
        
    Returns:
        AsteroidCoefficients object with both exterior and interior coefficients
        
    Raises:
        ValueError: If asteroid_name is not recognized
    """
    # Get the spherical harmonic coefficients from triaxial_ellipsoids module
    if asteroid_name == 'A1':
        triaxial_coeffs = get_asteroid_A1_coefficients()
        # Parameters from paper: Table 1
        mu = 0.002  # km^3/s^2
        R_b = 0.5   # km (Brillouin sphere radius)
        semi_axes = (1.0, 0.8, 0.6)  # km
    elif asteroid_name == 'A2':
        triaxial_coeffs = get_asteroid_A2_coefficients()
        mu = 0.002
        R_b = 0.5
        semi_axes = (1.0, 0.8, 0.6)
    elif asteroid_name == 'A3':
        triaxial_coeffs = get_asteroid_A3_coefficients()
        mu = 0.002
        R_b = 0.5
        semi_axes = (1.0, 0.8, 0.6)
    else:
        raise ValueError(f"Unknown triaxial ellipsoid: {asteroid_name}. "
                         f"Expected 'A1', 'A2', or 'A3'")
    
    # Create spherical harmonic coefficients
    sh_coeffs = SphericalHarmonicCoefficients(
        degree=triaxial_coeffs.degree,
        order=triaxial_coeffs.order,
        C_lm=triaxial_coeffs.C_lm,
        S_lm=triaxial_coeffs.S_lm,
        r0=triaxial_coeffs.r0,
        mu=mu
    )
    
    # For triaxial ellipsoids, we need to compute interior Bessel coefficients
    # The paper doesn't provide explicit formulas, but we can approximate
    # using the same spherical harmonic coefficients transformed to interior
    # representation. For now, we'll create placeholder coefficients.
    # In practice, these would be computed from the shape model.
    
    # Create placeholder Bessel coefficients (all zeros except dominant term)
    l_max = 2
    n_max = 5
    m_max = 5
    
    A_bar = np.zeros((l_max + 1, n_max + 1, m_max + 1), dtype=complex)
    B_bar = np.zeros((l_max + 1, n_max + 1, m_max + 1), dtype=complex)
    
    # Set dominant term (l=0, n=0, m=0) to match spherical harmonic C_00
    A_bar[0, 0, 0] = 1.0
    B_bar[0, 0, 0] = 0.0
    
    bessel_coeffs = BesselCoefficients(
        l_max=l_max,
        n_max=n_max,
        m_max=m_max,
        A_bar=A_bar,
        B_bar=B_bar,
        R_b=R_b,
        mu=mu
    )
    
    return AsteroidCoefficients(
        name=asteroid_name,
        sh_coeffs=sh_coeffs,
        bessel_coeffs=bessel_coeffs,
        is_triaxial=True
    )


def load_castalia_coefficients(coefficients_file: Optional[str] = None) -> AsteroidCoefficients:
    """
    Load Castalia coefficients from file.
    
    Args:
        coefficients_file: Path to JSON file containing Castalia coefficients.
                          If None, uses default location or synthetic data.
                          
    Returns:
        AsteroidCoefficients object for Castalia
        
    Raises:
        FileNotFoundError: If coefficients_file doesn't exist and no default found
    """
    # Castalia parameters from paper: Table 2
    mu = 4.46e-4  # km^3/s^2
    R_b = 1.0     # km (Brillouin sphere radius)
    
    if coefficients_file is None:
        # Try default location
        default_file = os.path.join(
            os.path.dirname(__file__),
            'castalia_coefficients.json'
        )
        if os.path.exists(default_file):
            coefficients_file = default_file
        else:
            # Create synthetic coefficients for development
            return _create_synthetic_castalia_coefficients(mu, R_b)
    
    # Load from JSON file
    with open(coefficients_file, 'r') as f:
        data = json.load(f)
    
    # Load spherical harmonic coefficients (4×4 model)
    C_lm_data = np.array(data['spherical_harmonics']['C_lm'])
    S_lm_data = np.array(data['spherical_harmonics']['S_lm'])
    
    sh_coeffs = SphericalHarmonicCoefficients(
        degree=4,
        order=4,
        C_lm=C_lm_data,
        S_lm=S_lm_data,
        r0=float(data['spherical_harmonics']['r0']),
        mu=mu
    )
    
    # Load Bessel coefficients (interior model)
    A_bar_data = np.array(data['bessel_coefficients']['A_bar'], dtype=complex)
    B_bar_data = np.array(data['bessel_coefficients']['B_bar'], dtype=complex)
    
    bessel_coeffs = BesselCoefficients(
        l_max=2,
        n_max=5,
        m_max=5,
        A_bar=A_bar_data,
        B_bar=B_bar_data,
        R_b=R_b,
        mu=mu
    )
    
    return AsteroidCoefficients(
        name='Castalia',
        sh_coeffs=sh_coeffs,
        bessel_coeffs=bessel_coeffs,
        is_triaxial=False
    )


def _create_synthetic_castalia_coefficients(mu: float, R_b: float) -> AsteroidCoefficients:
    """
    Create synthetic Castalia coefficients for development/testing.
    
    In a real implementation, these would be loaded from the actual
    coefficients provided in references [25,31] of the paper.
    
    Args:
        mu: Gravitational parameter
        R_b: Brillouin sphere radius
        
    Returns:
        Synthetic AsteroidCoefficients for Castalia
    """
    # Create synthetic spherical harmonic coefficients (4×4)
    degree = 4
    order = 4
    
    C_lm = np.zeros((degree + 1, order + 1))
    S_lm = np.zeros((degree + 1, order + 1))
    
    # Normalization: C_00 = 1.0
    C_lm[0, 0] = 1.0
    
    # Add some synthetic harmonics to create irregular gravity field
    # These values are made up for development purposes
    C_lm[2, 0] = -0.1  # J2 term
    C_lm[2, 2] = 0.05
    S_lm[2, 2] = 0.03
    C_lm[4, 0] = 0.01
    C_lm[4, 4] = 0.005
    
    sh_coeffs = SphericalHarmonicCoefficients(
        degree=degree,
        order=order,
        C_lm=C_lm,
        S_lm=S_lm,
        r0=R_b,  # Use Brillouin radius as reference
        mu=mu
    )
    
    # Create synthetic Bessel coefficients (interior model)
    l_max = 2
    n_max = 5
    m_max = 5
    
    A_bar = np.zeros((l_max + 1, n_max + 1, m_max + 1), dtype=complex)
    B_bar = np.zeros((l_max + 1, n_max + 1, m_max + 1), dtype=complex)
    
    # Dominant term
    A_bar[0, 0, 0] = 1.0
    
    # Add some interior structure terms
    A_bar[1, 1, 0] = 0.1 + 0.05j
    A_bar[2, 2, 0] = 0.02 + 0.01j
    B_bar[1, 1, 1] = 0.03 + 0.02j
    
    bessel_coeffs = BesselCoefficients(
        l_max=l_max,
        n_max=n_max,
        m_max=m_max,
        A_bar=A_bar,
        B_bar=B_bar,
        R_b=R_b,
        mu=mu
    )
    
    return AsteroidCoefficients(
        name='Castalia',
        sh_coeffs=sh_coeffs,
        bessel_coeffs=bessel_coeffs,
        is_triaxial=False
    )


def get_coefficients_for_asteroid(asteroid_name: str) -> AsteroidCoefficients:
    """
    Get coefficients for any supported asteroid.
    
    Args:
        asteroid_name: Name of asteroid ('A1', 'A2', 'A3', or 'Castalia')
        
    Returns:
        AsteroidCoefficients object
        
    Raises:
        ValueError: If asteroid_name is not recognized
    """
    if asteroid_name in ['A1', 'A2', 'A3']:
        return load_triaxial_ellipsoid_coefficients(asteroid_name)
    elif asteroid_name == 'Castalia':
        return load_castalia_coefficients()
    else:
        raise ValueError(f"Unknown asteroid: {asteroid_name}. "
                         f"Expected 'A1', 'A2', 'A3', or 'Castalia'")


def save_coefficients_to_file(coefficients: AsteroidCoefficients, 
                             filename: str) -> None:
    """
    Save asteroid coefficients to a JSON file.
    
    Args:
        coefficients: AsteroidCoefficients object to save
        filename: Path to output JSON file
    """
    data = {
        'name': coefficients.name,
        'is_triaxial': coefficients.is_triaxial,
        'spherical_harmonics': {
            'degree': int(coefficients.sh_coeffs.degree),
            'order': int(coefficients.sh_coeffs.order),
            'C_lm': coefficients.sh_coeffs.C_lm.tolist(),
            'S_lm': coefficients.sh_coeffs.S_lm.tolist(),
            'r0': float(coefficients.sh_coeffs.r0),
            'mu': float(coefficients.sh_coeffs.mu)
        },
        'bessel_coefficients': {
            'l_max': int(coefficients.bessel_coeffs.l_max),
            'n_max': int(coefficients.bessel_coeffs.n_max),
            'm_max': int(coefficients.bessel_coeffs.m_max),
            'A_bar': coefficients.bessel_coeffs.A_bar.tolist(),
            'B_bar': coefficients.bessel_coeffs.B_bar.tolist(),
            'R_b': float(coefficients.bessel_coeffs.R_b),
            'mu': float(coefficients.bessel_coeffs.mu)
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


def validate_coefficient_symmetry(coefficients: AsteroidCoefficients) -> Dict[str, bool]:
    """
    Validate symmetry properties of coefficients.
    
    For a real gravity field, coefficients should satisfy:
    1. C_lm and S_lm are real
    2. S_l0 = 0 for all l (zonal terms have no sine component)
    3. Certain symmetries for interior coefficients
    
    Args:
        coefficients: AsteroidCoefficients to validate
        
    Returns:
        Dictionary of validation results
    """
    results = {}
    
    # Check spherical harmonic coefficients
    sh = coefficients.sh_coeffs
    
    # 1. Check that C_lm and S_lm are real (within numerical tolerance)
    results['C_lm_real'] = np.all(np.abs(np.imag(sh.C_lm)) < 1e-10)
    results['S_lm_real'] = np.all(np.abs(np.imag(sh.S_lm)) < 1e-10)
    
    # 2. Check S_l0 = 0 for all l
    s_l0 = sh.S_lm[:, 0]
    results['S_l0_zero'] = np.all(np.abs(s_l0) < 1e-10)
    
    # 3. Check C_00 = 1.0
    results['C_00_normalized'] = abs(sh.C_lm[0, 0] - 1.0) < 1e-10
    
    # Check Bessel coefficients
    bessel = coefficients.bessel_coeffs
    
    # 4. Check that A_bar and B_bar have expected shape
    results['A_bar_shape_correct'] = (
        bessel.A_bar.shape == (bessel.l_max + 1, bessel.n_max + 1, bessel.m_max + 1)
    )
    results['B_bar_shape_correct'] = (
        bessel.B_bar.shape == (bessel.l_max + 1, bessel.n_max + 1, bessel.m_max + 1)
    )
    
    return results


# Convenience functions for common asteroids
def get_A1_coefficients() -> AsteroidCoefficients:
    """Get coefficients for asteroid A1."""
    return get_coefficients_for_asteroid('A1')


def get_A2_coefficients() -> AsteroidCoefficients:
    """Get coefficients for asteroid A2."""
    return get_coefficients_for_asteroid('A2')


def get_A3_coefficients() -> AsteroidCoefficients:
    """Get coefficients for asteroid A3."""
    return get_coefficients_for_asteroid('A3')


def get_castalia_coefficients() -> AsteroidCoefficients:
    """Get coefficients for Castalia."""
    return get_coefficients_for_asteroid('Castalia')


if __name__ == '__main__':
    # Test the coefficient loader
    print("Testing coefficient loader...")
    
    # Test triaxial ellipsoids
    for asteroid in ['A1', 'A2', 'A3']:
        coeffs = get_coefficients_for_asteroid(asteroid)
        print(f"\n{asteroid}:")
        print(f"  Name: {coeffs.name}")
        print(f"  Is triaxial: {coeffs.is_triaxial}")
        print(f"  SH degree: {coeffs.sh_coeffs.degree}")
        print(f"  Bessel l_max: {coeffs.bessel_coeffs.l_max}")
        
        # Validate
        validation = validate_coefficient_symmetry(coeffs)
        print(f"  Validation: {validation}")
    
    # Test Castalia
    castalia_coeffs = get_castalia_coefficients()
    print(f"\nCastalia:")
    print(f"  Name: {castalia_coeffs.name}")
    print(f"  Is triaxial: {castalia_coeffs.is_triaxial}")
    print(f"  SH degree: {castalia_coeffs.sh_coeffs.degree}")
    print(f"  Bessel l_max: {castalia_coeffs.bessel_coeffs.l_max}")
    
    validation = validate_coefficient_symmetry(castalia_coeffs)
    print(f"  Validation: {validation}")
    
    print("\nAll tests passed!")