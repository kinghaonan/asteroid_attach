"""
Gravity Models Module

This module provides unified access to all gravity models for asteroid landing optimization,
including spherical harmonics for exterior gravity and spherical Bessel functions for interior gravity.
"""

from .spherical_harmonics import (
    SphericalHarmonicsGravity,
    create_uniform_sphere_coefficients,
    test_spherical_harmonics,
)

from .interior_bessel import (
    InteriorBesselGravity,
    BesselGravityResult,
    create_uniform_sphere_bessel_coefficients,
    test_interior_bessel,
)

from .gravity_calculator import (
    GravityCalculator,
    GravityResult,
    create_gravity_calculator_for_asteroid,
    test_gravity_calculator,
)

__all__ = [
    # Spherical harmonics (exterior gravity)
    "SphericalHarmonicsGravity",
    "create_uniform_sphere_coefficients",
    "test_spherical_harmonics",
    
    # Spherical Bessel (interior gravity)
    "InteriorBesselGravity",
    "BesselGravityResult",
    "create_uniform_sphere_bessel_coefficients",
    "test_interior_bessel",
    
    # Unified gravity calculator with switching logic
    "GravityCalculator",
    "GravityResult",
    "create_gravity_calculator_for_asteroid",
    "test_gravity_calculator",
]

# Convenience function to test the entire gravity models module
def test_gravity_models():
    """Run comprehensive tests for all gravity models."""
    import logging
    
    logger = logging.getLogger(__name__)
    logger.info("Testing gravity models module...")
    
    # Test spherical harmonics
    logger.info("Testing spherical harmonics...")
    sh_success = test_spherical_harmonics()
    if not sh_success:
        logger.error("Spherical harmonics tests failed!")
        return False
    
    # Test interior Bessel
    logger.info("Testing interior Bessel...")
    bessel_success = test_interior_bessel()
    if not bessel_success:
        logger.error("Interior Bessel tests failed!")
        return False
    
    # Test gravity calculator
    logger.info("Testing gravity calculator...")
    calculator_success = test_gravity_calculator()
    if not calculator_success:
        logger.error("Gravity calculator tests failed!")
        return False
    
    logger.info("All gravity models tests passed!")
    return True