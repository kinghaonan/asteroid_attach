"""
Unit tests for gravity models module.

This module tests the spherical harmonics, interior Bessel, and gravity calculator
implementations to ensure they produce correct gravitational accelerations and
satisfy expected properties.
"""

import numpy as np
import pytest
from pathlib import Path
import sys
import os

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gravity_models import (
    SphericalHarmonicsGravity,
    InteriorBesselGravity,
    GravityCalculator,
    create_uniform_sphere_coefficients,
    create_uniform_sphere_bessel_coefficients,
    create_gravity_calculator_for_asteroid,
    test_gravity_models
)
from asteroid_data import (
    get_A1_coefficients,
    get_A2_coefficients,
    get_A3_coefficients,
    get_castalia_coefficients
)
from config import (
    ASTEROID_A1,
    ASTEROID_A2,
    ASTEROID_A3,
    CASTALIA,
    G
)


class TestSphericalHarmonics:
    """Test spherical harmonics gravity model."""
    
    def test_uniform_sphere_coefficients(self):
        """Test creation of uniform sphere coefficients."""
        mu = 1.0
        r0 = 1.0
        C_lm, S_lm = create_uniform_sphere_coefficients(mu, r0)
        
        # Check shape
        assert C_lm.shape == (5, 5)
        assert S_lm.shape == (5, 5)
        
        # Check C_00 = 1 (normalized)
        assert np.abs(C_lm[0, 0] - 1.0) < 1e-10
        
        # Check all other coefficients are zero
        for l in range(5):
            for m in range(l + 1):
                if l == 0 and m == 0:
                    continue
                assert np.abs(C_lm[l, m]) < 1e-10
                assert np.abs(S_lm[l, m]) < 1e-10
    
    def test_uniform_sphere_gravity(self):
        """Test spherical harmonics for uniform sphere (should give point mass)."""
        mu = 1.0
        r0 = 1.0
        C_lm, S_lm = create_uniform_sphere_coefficients(mu, r0)
        
        gravity_model = SphericalHarmonicsGravity(C_lm, S_lm, mu, r0)
        
        # Test at various positions
        test_positions = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 2.0, 0.0]),
            np.array([0.0, 0.0, 3.0]),
            np.array([1.0, 1.0, 1.0]) / np.sqrt(3),
        ]
        
        for r_vec in test_positions:
            r = np.linalg.norm(r_vec)
            
            # Compute potential and acceleration
            potential = gravity_model.potential(r_vec)
            acceleration = gravity_model.acceleration(r_vec)
            
            # Point mass formulas
            expected_potential = -mu / r
            expected_acceleration = -mu * r_vec / r**3
            
            # Check accuracy
            assert np.abs(potential - expected_potential) < 1e-10
            assert np.linalg.norm(acceleration - expected_acceleration) < 1e-10
    
    def test_spherical_harmonics_symmetry(self):
        """Test symmetry properties of spherical harmonics."""
        mu = 1.0
        r0 = 1.0
        
        # Create non-uniform coefficients (simple quadrupole)
        C_lm = np.zeros((5, 5))
        S_lm = np.zeros((5, 5))
        C_lm[0, 0] = 1.0
        C_lm[2, 0] = 0.1  # J2 term
        C_lm[2, 2] = 0.05  # C22 term
        S_lm[2, 2] = 0.03  # S22 term
        
        gravity_model = SphericalHarmonicsGravity(C_lm, S_lm, mu, r0)
        
        # Test symmetry: acceleration at (x,y,z) should be consistent
        r_vec1 = np.array([1.0, 2.0, 3.0])
        r_vec2 = np.array([-1.0, -2.0, -3.0])
        
        acc1 = gravity_model.acceleration(r_vec1)
        acc2 = gravity_model.acceleration(r_vec2)
        
        # Acceleration should point in opposite direction for opposite position
        assert np.linalg.norm(acc1 + acc2) < 1e-10
    
    def test_spherical_harmonics_derivatives(self):
        """Test that acceleration is gradient of potential."""
        mu = 1.0
        r0 = 1.0
        
        # Create test coefficients
        C_lm = np.zeros((5, 5))
        S_lm = np.zeros((5, 5))
        C_lm[0, 0] = 1.0
        C_lm[2, 0] = 0.1
        C_lm[2, 2] = 0.05
        S_lm[2, 2] = 0.03
        
        gravity_model = SphericalHarmonicsGravity(C_lm, S_lm, mu, r0)
        
        # Test position
        r_vec = np.array([1.5, 0.8, 0.3])
        
        # Compute potential and acceleration
        potential = gravity_model.potential(r_vec)
        acceleration = gravity_model.acceleration(r_vec)
        
        # Finite difference gradient check
        h = 1e-6
        for i in range(3):
            r_plus = r_vec.copy()
            r_minus = r_vec.copy()
            r_plus[i] += h
            r_minus[i] -= h
            
            potential_plus = gravity_model.potential(r_plus)
            potential_minus = gravity_model.potential(r_minus)
            
            grad_i = (potential_plus - potential_minus) / (2 * h)
            
            assert np.abs(grad_i - acceleration[i]) < 1e-6


class TestInteriorBessel:
    """Test interior Bessel gravity model."""
    
    def test_uniform_sphere_bessel_coefficients(self):
        """Test creation of uniform sphere Bessel coefficients."""
        mu = 1.0
        R_b = 1.0
        l_max = 2
        n_max = 5
        m_max = 5
        
        A_bar, B_bar = create_uniform_sphere_bessel_coefficients(mu, R_b, l_max, n_max, m_max)
        
        # Check shapes
        assert A_bar.shape == (l_max + 1, n_max + 1, m_max + 1)
        assert B_bar.shape == (l_max + 1, n_max + 1, m_max + 1)
        
        # For uniform sphere, only A_bar[0,0,0] should be non-zero
        assert np.abs(A_bar[0, 0, 0] - 1.0) < 1e-10
        
        # All other coefficients should be zero
        for l in range(l_max + 1):
            for n in range(n_max + 1):
                for m in range(m_max + 1):
                    if l == 0 and n == 0 and m == 0:
                        continue
                    assert np.abs(A_bar[l, n, m]) < 1e-10
                    assert np.abs(B_bar[l, n, m]) < 1e-10
    
    def test_uniform_sphere_interior_gravity(self):
        """Test interior Bessel model for uniform sphere."""
        mu = 1.0
        R_b = 1.0
        l_max = 2
        n_max = 5
        m_max = 5
        
        A_bar, B_bar = create_uniform_sphere_bessel_coefficients(mu, R_b, l_max, n_max, m_max)
        
        gravity_model = InteriorBesselGravity(A_bar, B_bar, mu, R_b, l_max, n_max, m_max)
        
        # Test positions inside sphere
        test_positions = [
            np.array([0.1, 0.0, 0.0]),
            np.array([0.0, 0.2, 0.0]),
            np.array([0.0, 0.0, 0.3]),
            np.array([0.4, 0.4, 0.4]) / np.sqrt(3),
        ]
        
        for r_vec in test_positions:
            r = np.linalg.norm(r_vec)
            
            # Compute result
            result = gravity_model.compute(r_vec)
            
            # For uniform sphere, interior gravity is linear: g = -mu * r / R_b^3
            expected_acceleration = -mu * r_vec / R_b**3
            expected_potential = -mu * (3 * R_b**2 - r**2) / (2 * R_b**3)
            
            # Check accuracy
            assert np.abs(result.potential - expected_potential) < 1e-10
            assert np.linalg.norm(result.acceleration - expected_acceleration) < 1e-10
            assert result.inside_brillouin == True
    
    def test_interior_bessel_at_origin(self):
        """Test interior Bessel model at origin (r=0)."""
        mu = 1.0
        R_b = 1.0
        l_max = 2
        n_max = 5
        m_max = 5
        
        A_bar, B_bar = create_uniform_sphere_bessel_coefficients(mu, R_b, l_max, n_max, m_max)
        
        gravity_model = InteriorBesselGravity(A_bar, B_bar, mu, R_b, l_max, n_max, m_max)
        
        # Test at origin
        r_vec = np.array([0.0, 0.0, 0.0])
        result = gravity_model.compute(r_vec)
        
        # Acceleration should be zero at origin for symmetric body
        assert np.linalg.norm(result.acceleration) < 1e-10
        assert result.inside_brillouin == True
    
    def test_interior_bessel_derivatives(self):
        """Test that acceleration is gradient of potential for interior model."""
        mu = 1.0
        R_b = 1.0
        l_max = 2
        n_max = 5
        m_max = 5
        
        # Create non-uniform coefficients
        A_bar = np.zeros((l_max + 1, n_max + 1, m_max + 1), dtype=complex)
        B_bar = np.zeros((l_max + 1, n_max + 1, m_max + 1), dtype=complex)
        
        # Set some non-zero coefficients
        A_bar[0, 0, 0] = 1.0
        A_bar[1, 1, 0] = 0.1 + 0.0j  # Real
        A_bar[2, 2, 1] = 0.05 + 0.02j  # Complex
        B_bar[1, 1, 1] = 0.03 + 0.01j  # Complex
        
        gravity_model = InteriorBesselGravity(A_bar, B_bar, mu, R_b, l_max, n_max, m_max)
        
        # Test position inside sphere
        r_vec = np.array([0.3, 0.2, 0.1])
        
        # Compute potential and acceleration
        result = gravity_model.compute(r_vec)
        potential = result.potential
        acceleration = result.acceleration
        
        # Finite difference gradient check
        h = 1e-6
        for i in range(3):
            r_plus = r_vec.copy()
            r_minus = r_vec.copy()
            r_plus[i] += h
            r_minus[i] -= h
            
            result_plus = gravity_model.compute(r_plus)
            result_minus = gravity_model.compute(r_minus)
            
            grad_i = (result_plus.potential - result_minus.potential) / (2 * h)
            
            assert np.abs(grad_i - acceleration[i]) < 1e-6


class TestGravityCalculator:
    """Test unified gravity calculator with switching logic."""
    
    def test_gravity_calculator_creation(self):
        """Test creation of gravity calculator for different asteroids."""
        # Test for A1
        calculator_A1 = create_gravity_calculator_for_asteroid("A1")
        assert calculator_A1 is not None
        assert calculator_A1.mu == ASTEROID_A1.mu
        assert calculator_A1.R_b == ASTEROID_A1.R_b
        
        # Test for Castalia
        calculator_castalia = create_gravity_calculator_for_asteroid("Castalia")
        assert calculator_castalia is not None
        assert calculator_castalia.mu == CASTALIA.mu
        assert calculator_castalia.R_b == CASTALIA.R_b
    
    def test_switching_logic(self):
        """Test that calculator switches between models at Brillouin sphere."""
        # Use A1 for testing
        calculator = create_gravity_calculator_for_asteroid("A1")
        R_b = ASTEROID_A1.R_b
        
        # Test positions outside Brillouin sphere
        outside_positions = [
            np.array([R_b * 1.1, 0.0, 0.0]),
            np.array([0.0, R_b * 1.2, 0.0]),
            np.array([0.0, 0.0, R_b * 1.3]),
        ]
        
        for r_vec in outside_positions:
            result = calculator.compute(r_vec)
            assert result.model_used == "spherical_harmonics"
            assert result.inside_brillouin == False
        
        # Test positions inside Brillouin sphere
        inside_positions = [
            np.array([R_b * 0.9, 0.0, 0.0]),
            np.array([0.0, R_b * 0.8, 0.0]),
            np.array([0.0, 0.0, R_b * 0.7]),
        ]
        
        for r_vec in inside_positions:
            result = calculator.compute(r_vec)
            assert result.model_used == "interior_bessel"
            assert result.inside_brillouin == True
        
        # Test position exactly at Brillouin sphere (should use interior)
        r_vec = np.array([R_b, 0.0, 0.0])
        result = calculator.compute(r_vec)
        assert result.model_used == "interior_bessel"
        assert result.inside_brillouin == True
    
    def test_continuity_at_brillouin_sphere(self):
        """Test that gravity is continuous at the Brillouin sphere boundary."""
        calculator = create_gravity_calculator_for_asteroid("A1")
        R_b = ASTEROID_A1.R_b
        
        # Test along radial direction
        directions = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 1.0, 1.0]) / np.sqrt(3),
        ]
        
        for direction in directions:
            # Normalize direction
            direction = direction / np.linalg.norm(direction)
            
            # Position just inside Brillouin sphere
            r_inside = direction * (R_b - 1e-6)
            result_inside = calculator.compute(r_inside)
            
            # Position just outside Brillouin sphere
            r_outside = direction * (R_b + 1e-6)
            result_outside = calculator.compute(r_outside)
            
            # Check continuity (allow small tolerance due to numerical differences)
            potential_diff = np.abs(result_inside.potential - result_outside.potential)
            acc_diff = np.linalg.norm(result_inside.acceleration - result_outside.acceleration)
            
            # Continuity should be good to about 1e-6
            assert potential_diff < 1e-5
            assert acc_diff < 1e-5
    
    def test_linearization(self):
        """Test gravity linearization for successive solution method."""
        calculator = create_gravity_calculator_for_asteroid("A1")
        
        # Reference position
        r_ref = np.array([ASTEROID_A1.R_b * 1.5, 0.0, 0.0])
        
        # Linearize gravity
        A, c = calculator.linearize(r_ref)
        
        # A should be 3x3 matrix
        assert A.shape == (3, 3)
        # c should be 3-vector
        assert c.shape == (3,)
        
        # Test linear approximation
        r_test = r_ref + np.array([0.1, 0.05, -0.02])
        
        # Exact gravity
        exact_result = calculator.compute(r_test)
        exact_acc = exact_result.acceleration
        
        # Linear approximation
        linear_acc = A @ r_test + c
        
        # Check approximation accuracy (should be good for small displacements)
        error = np.linalg.norm(exact_acc - linear_acc)
        assert error < 1e-3
    
    def test_point_mass_approximation(self):
        """Test point mass approximation for large distances."""
        calculator = create_gravity_calculator_for_asteroid("A1")
        mu = ASTEROID_A1.mu
        
        # Test at large distance (10x Brillouin radius)
        r_vec = np.array([ASTEROID_A1.R_b * 10.0, 0.0, 0.0])
        
        # Compute exact gravity
        result = calculator.compute(r_vec)
        exact_acc = result.acceleration
        
        # Point mass approximation
        r = np.linalg.norm(r_vec)
        point_mass_acc = -mu * r_vec / r**3
        
        # Error should be small for large distances
        error = np.linalg.norm(exact_acc - point_mass_acc) / np.linalg.norm(point_mass_acc)
        assert error < 0.01  # Less than 1% error


class TestGravityModelIntegration:
    """Integration tests for gravity models with asteroid data."""
    
    def test_A1_gravity_properties(self):
        """Test gravity properties for asteroid A1."""
        calculator = create_gravity_calculator_for_asteroid("A1")
        
        # Test at various positions
        positions = [
            np.array([ASTEROID_A1.R_b * 2.0, 0.0, 0.0]),  # Far outside
            np.array([ASTEROID_A1.R_b * 1.1, 0.0, 0.0]),  # Just outside
            np.array([ASTEROID_A1.R_b * 0.9, 0.0, 0.0]),  # Just inside
            np.array([ASTEROID_A1.R_b * 0.5, 0.0, 0.0]),  # Well inside
        ]
        
        for r_vec in positions:
            result = calculator.compute(r_vec)
            
            # Basic sanity checks
            assert result.acceleration.shape == (3,)
            assert isinstance(result.potential, float)
            assert result.radial_distance == np.linalg.norm(r_vec)
            
            # Acceleration should point toward origin (approximately)
            if np.linalg.norm(r_vec) > 1e-6:
                acc_dir = result.acceleration / np.linalg.norm(result.acceleration)
                r_dir = r_vec / np.linalg.norm(r_vec)
                
                # Acceleration should be opposite to position direction
                dot_product = np.dot(acc_dir, r_dir)
                assert dot_product < -0.9  # Strongly inward
    
    def test_A2_A3_comparison(self):
        """Compare gravity fields of A2 and A3 (should be different)."""
        calculator_A2 = create_gravity_calculator_for_asteroid("A2")
        calculator_A3 = create_gravity_calculator_for_asteroid("A3")
        
        # Test at same position
        r_vec = np.array([1000.0, 500.0, 200.0])
        
        result_A2 = calculator_A2.compute(r_vec)
        result_A3 = calculator_A3.compute(r_vec)
        
        # Accelerations should be different (different asteroid shapes)
        acc_diff = np.linalg.norm(result_A2.acceleration - result_A3.acceleration)
        assert acc_diff > 1e-6  # Should be measurably different
    
    def test_castalia_gravity(self):
        """Test Castalia gravity (using synthetic coefficients)."""
        calculator = create_gravity_calculator_for_asteroid("Castalia")
        
        # Test at a position
        r_vec = np.array([CASTALIA.R_b * 1.5, 0.0, 0.0])
        result = calculator.compute(r_vec)
        
        # Basic checks
        assert result.acceleration.shape == (3,)
        assert isinstance(result.potential, float)
        
        # Should use spherical harmonics (outside Brillouin sphere)
        assert result.model_used == "spherical_harmonics"
        assert result.inside_brillouin == False


def test_module_test_function():
    """Test the module's test function."""
    # This should run without errors
    success = test_gravity_models()
    assert success == True


if __name__ == "__main__":
    # Run tests
    print("Running gravity models tests...")
    
    # Create test instances
    test_sph = TestSphericalHarmonics()
    test_bessel = TestInteriorBessel()
    test_calc = TestGravityCalculator()
    test_integ = TestGravityModelIntegration()
    
    # Run test methods
    print("Testing spherical harmonics...")
    test_sph.test_uniform_sphere_coefficients()
    test_sph.test_uniform_sphere_gravity()
    test_sph.test_spherical_harmonics_symmetry()
    test_sph.test_spherical_harmonics_derivatives()
    
    print("Testing interior Bessel...")
    test_bessel.test_uniform_sphere_bessel_coefficients()
    test_bessel.test_uniform_sphere_interior_gravity()
    test_bessel.test_interior_bessel_at_origin()
    test_bessel.test_interior_bessel_derivatives()
    
    print("Testing gravity calculator...")
    test_calc.test_gravity_calculator_creation()
    test_calc.test_switching_logic()
    test_calc.test_continuity_at_brillouin_sphere()
    test_calc.test_linearization()
    test_calc.test_point_mass_approximation()
    
    print("Testing integration...")
    test_integ.test_A1_gravity_properties()
    test_integ.test_A2_A3_comparison()
    test_integ.test_castalia_gravity()
    
    print("Testing module test function...")
    test_module_test_function()
    
    print("\nAll gravity models tests passed!")