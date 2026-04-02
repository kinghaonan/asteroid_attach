"""
Asteroid Data Module

This module provides data structures and loading functions for asteroid gravity coefficients,
including spherical harmonics for exterior gravity and spherical Bessel functions for interior gravity.
"""

from .triaxial_ellipsoids import (
    TriaxialEllipsoidCoefficients,
    compute_triaxial_ellipsoid_coefficients,
    get_asteroid_A1_coefficients,
    get_asteroid_A2_coefficients,
    get_asteroid_A3_coefficients,
    get_all_triaxial_coefficients,
    validate_coefficients,
)

from .coefficient_loader import (
    SphericalHarmonicCoefficients,
    BesselCoefficients,
    AsteroidCoefficients,
    load_triaxial_ellipsoid_coefficients,
    load_castalia_coefficients,
    get_coefficients_for_asteroid,
    save_coefficients_to_file,
    validate_coefficient_symmetry,
    get_A1_coefficients,
    get_A2_coefficients,
    get_A3_coefficients,
    get_castalia_coefficients,
)

from .castalia import (
    CastaliaShapeModel,
    get_castalia_shape_model,
    compute_castalia_surface_normal,
    validate_castalia_landing_site,
    get_castalia_landing_sites,
)

__all__ = [
    # From triaxial_ellipsoids.py
    'TriaxialEllipsoidCoefficients',
    'compute_triaxial_ellipsoid_coefficients',
    'get_asteroid_A1_coefficients',
    'get_asteroid_A2_coefficients',
    'get_asteroid_A3_coefficients',
    'get_all_triaxial_coefficients',
    'validate_coefficients',
    
    # From coefficient_loader.py
    'SphericalHarmonicCoefficients',
    'BesselCoefficients',
    'AsteroidCoefficients',
    'load_triaxial_ellipsoid_coefficients',
    'load_castalia_coefficients',
    'get_coefficients_for_asteroid',
    'save_coefficients_to_file',
    'validate_coefficient_symmetry',
    'get_A1_coefficients',
    'get_A2_coefficients',
    'get_A3_coefficients',
    'get_castalia_coefficients',
    
    # From castalia.py
    'CastaliaShapeModel',
    'get_castalia_shape_model',
    'compute_castalia_surface_normal',
    'validate_castalia_landing_site',
    'get_castalia_landing_sites',
]