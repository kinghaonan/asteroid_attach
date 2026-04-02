"""
Gravity Calculator with Switching Logic

Implements the switching logic between exterior spherical harmonics and interior
spherical Bessel gravity models based on the Brillouin sphere radius.

Based on Eqs. 1-7 from the paper, with switching at the Brillouin sphere.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

from .spherical_harmonics import SphericalHarmonicsGravity
from .interior_bessel import InteriorBesselGravity
from ..asteroid_data.coefficient_loader import AsteroidCoefficients


@dataclass
class GravityResult:
    """Container for gravity calculation results."""
    acceleration: np.ndarray  # [m/s²] gravitational acceleration vector
    potential: float          # [m²/s²] gravitational potential
    model_used: str          # "spherical_harmonics" or "interior_bessel"
    radial_distance: float   # [m] distance from asteroid center
    inside_brillouin: bool   # True if inside Brillouin sphere


class GravityCalculator:
    """
    Unified gravity calculator that switches between exterior spherical harmonics
    and interior spherical Bessel models at the Brillouin sphere.
    
    This implements the switching logic described in the paper: use spherical
    harmonics outside the Brillouin sphere (r > R_b) and spherical Bessel
    functions inside (r ≤ R_b).
    
    Parameters
    ----------
    coefficients : AsteroidCoefficients
        Complete coefficient set for the asteroid
    mu : float
        Gravitational parameter [m³/s²]
    R_b : float
        Brillouin sphere radius [m]
    r0 : float, optional
        Reference radius for spherical harmonics [m], default=1.0
    """
    
    def __init__(self, coefficients: AsteroidCoefficients, mu: float, R_b: float, r0: float = 1.0):
        self.coefficients = coefficients
        self.mu = mu
        self.R_b = R_b
        self.r0 = r0
        
        # Initialize both gravity models
        self._init_models()
        
        # Cache for linearization matrices (used in successive solution)
        self._linearization_cache: Dict[str, Any] = {}
        
    def _init_models(self):
        """Initialize both gravity models from coefficients."""
        # Initialize spherical harmonics (exterior) model
        if self.coefficients.sh_coeffs is not None:
            self.sh_model = SphericalHarmonicsGravity(
                C_lm=self.coefficients.sh_coeffs.C_lm,
                S_lm=self.coefficients.sh_coeffs.S_lm,
                mu=self.mu,
                r0=self.r0
            )
        else:
            self.sh_model = None
            
        # Initialize spherical Bessel (interior) model
        if self.coefficients.bessel_coeffs is not None:
            self.bessel_model = InteriorBesselGravity(
                A_bar=self.coefficients.bessel_coeffs.A_bar,
                B_bar=self.coefficients.bessel_coeffs.B_bar,
                mu=self.mu,
                R_b=self.R_b,
                l_max=self.coefficients.bessel_coeffs.l_max,
                n_max=self.coefficients.bessel_coeffs.n_max,
                m_max=self.coefficients.bessel_coeffs.m_max
            )
        else:
            self.bessel_model = None
            
    def compute(self, r_vec: np.ndarray) -> GravityResult:
        """
        Compute gravitational acceleration and potential at position r_vec.
        
        Parameters
        ----------
        r_vec : np.ndarray
            Position vector in asteroid-fixed frame [m], shape (3,)
            
        Returns
        -------
        GravityResult
            Container with acceleration, potential, and model info
        """
        r_vec = np.asarray(r_vec, dtype=float)
        if r_vec.shape != (3,):
            raise ValueError(f"r_vec must have shape (3,), got {r_vec.shape}")
            
        r = np.linalg.norm(r_vec)
        
        # Determine which model to use based on Brillouin sphere
        if r > self.R_b:
            # Outside Brillouin sphere: use spherical harmonics
            if self.sh_model is None:
                raise ValueError("Spherical harmonics model not available")
                
            accel = self.sh_model.acceleration(r_vec)
            potential = self.sh_model.potential(r_vec)
            model_used = "spherical_harmonics"
            inside_brillouin = False
            
        else:
            # Inside or on Brillouin sphere: use spherical Bessel
            if self.bessel_model is None:
                raise ValueError("Spherical Bessel model not available")
                
            result = self.bessel_model.compute(r_vec)
            accel = result.acceleration
            potential = result.potential
            model_used = "interior_bessel"
            inside_brillouin = True
            
        return GravityResult(
            acceleration=accel,
            potential=potential,
            model_used=model_used,
            radial_distance=r,
            inside_brillouin=inside_brillouin
        )
    
    def acceleration(self, r_vec: np.ndarray) -> np.ndarray:
        """
        Compute only gravitational acceleration (convenience method).
        
        Parameters
        ----------
        r_vec : np.ndarray
            Position vector in asteroid-fixed frame [m], shape (3,)
            
        Returns
        -------
        np.ndarray
            Gravitational acceleration vector [m/s²], shape (3,)
        """
        return self.compute(r_vec).acceleration
    
    def potential(self, r_vec: np.ndarray) -> float:
        """
        Compute only gravitational potential (convenience method).
        
        Parameters
        ----------
        r_vec : np.ndarray
            Position vector in asteroid-fixed frame [m], shape (3,)
            
        Returns
        -------
        float
            Gravitational potential [m²/s²]
        """
        return self.compute(r_vec).potential
    
    def linearize(self, r_ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize gravity field around reference position r_ref.
        
        This computes the dominant term A(r_ref) and residual c(r_ref) for
        the linearized gravity approximation used in successive solution:
            g(r) ≈ A(r_ref) * r + c(r_ref)
            
        For spherical harmonics (exterior), the dominant term is the point-mass
        approximation: A = -μ/r³ * I, c = g(r_ref) - A * r_ref.
        
        For spherical Bessel (interior), the dominant term is computed from
        the Bessel function derivatives (Eq. 57 in paper).
        
        Parameters
        ----------
        r_ref : np.ndarray
            Reference position vector [m], shape (3,)
            
        Returns
        -------
        A : np.ndarray
            Linearization matrix (3×3) [1/s²]
        c : np.ndarray
            Residual vector (3,) [m/s²]
        """
        r_ref = np.asarray(r_ref, dtype=float)
        r_norm = np.linalg.norm(r_ref)
        
        # Check cache first
        cache_key = tuple(r_ref)
        if cache_key in self._linearization_cache:
            return self._linearization_cache[cache_key]
        
        if r_norm > self.R_b:
            # Spherical harmonics linearization (point-mass dominant term)
            if r_norm < 1e-6:
                # At origin, use zero matrix
                A = np.zeros((3, 3))
                c = np.zeros(3)
            else:
                # Point-mass approximation: A = -μ/r³ * I
                A = -self.mu / (r_norm**3) * np.eye(3)
                
                # Compute actual gravity at reference point
                g_ref = self.acceleration(r_ref)
                
                # Residual: c = g_ref - A * r_ref
                c = g_ref - A @ r_ref
                
        else:
            # Spherical Bessel linearization (Eq. 57)
            # For interior model, we need to compute the Jacobian of g(r)
            # We'll approximate it using finite differences for now
            # (In the paper, they derive analytical expressions)
            
            # Compute gravity at reference point
            g_ref = self.acceleration(r_ref)
            
            # Finite difference approximation of Jacobian
            eps = 1e-4  # Small perturbation
            A = np.zeros((3, 3))
            
            for i in range(3):
                # Perturb in each coordinate direction
                r_pert = r_ref.copy()
                r_pert[i] += eps
                g_pert = self.acceleration(r_pert)
                A[:, i] = (g_pert - g_ref) / eps
                
            # Residual: c = g_ref - A * r_ref
            c = g_ref - A @ r_ref
            
        # Cache the result
        self._linearization_cache[cache_key] = (A, c)
        
        return A, c
    
    def test_switching_continuity(self, num_points: int = 100) -> Dict[str, Any]:
        """
        Test continuity of gravity field across Brillouin sphere boundary.
        
        Parameters
        ----------
        num_points : int, optional
            Number of test points along radial direction, default=100
            
        Returns
        -------
        Dict[str, Any]
            Test results including max discontinuity and plots
        """
        # Generate points along radial direction crossing R_b
        r_min = 0.9 * self.R_b
        r_max = 1.1 * self.R_b
        radii = np.linspace(r_min, r_max, num_points)
        
        # Test along x-axis
        positions = np.zeros((num_points, 3))
        positions[:, 0] = radii
        
        # Compute accelerations
        accelerations = np.zeros((num_points, 3))
        potentials = np.zeros(num_points)
        models_used = []
        
        for i, r_vec in enumerate(positions):
            result = self.compute(r_vec)
            accelerations[i] = result.acceleration
            potentials[i] = result.potential
            models_used.append(result.model_used)
            
        # Find index where switching occurs
        switch_idx = None
        for i in range(1, num_points):
            if models_used[i] != models_used[i-1]:
                switch_idx = i
                break
                
        # Compute discontinuity at switching point
        discontinuity = None
        if switch_idx is not None:
            # Acceleration just inside and just outside
            accel_inside = accelerations[switch_idx-1]
            accel_outside = accelerations[switch_idx]
            discontinuity = np.linalg.norm(accel_outside - accel_inside)
            
        return {
            'radii': radii,
            'accelerations': accelerations,
            'potentials': potentials,
            'models_used': models_used,
            'switch_radius': self.R_b,
            'switch_idx': switch_idx,
            'discontinuity_magnitude': discontinuity,
            'max_acceleration': np.max(np.linalg.norm(accelerations, axis=1)),
            'min_acceleration': np.min(np.linalg.norm(accelerations, axis=1))
        }
    
    def test_point_mass_approximation(self, r_test: np.ndarray) -> Dict[str, float]:
        """
        Test how well point-mass approximation matches full model.
        
        Parameters
        ----------
        r_test : np.ndarray
            Test position vector [m], shape (3,)
            
        Returns
        -------
        Dict[str, float]
            Comparison metrics
        """
        # Compute actual gravity
        g_actual = self.acceleration(r_test)
        
        # Point-mass approximation
        r = np.linalg.norm(r_test)
        if r < 1e-6:
            g_point_mass = np.zeros(3)
        else:
            g_point_mass = -self.mu / (r**3) * r_test
            
        # Compute errors
        error_vector = g_actual - g_point_mass
        error_magnitude = np.linalg.norm(error_vector)
        error_relative = error_magnitude / np.linalg.norm(g_actual) if np.linalg.norm(g_actual) > 0 else 0
        
        return {
            'actual_acceleration': g_actual,
            'point_mass_approximation': g_point_mass,
            'error_vector': error_vector,
            'error_magnitude': error_magnitude,
            'error_relative': error_relative,
            'distance': r,
            'inside_brillouin': r <= self.R_b
        }


def create_gravity_calculator_for_asteroid(asteroid_name: str) -> GravityCalculator:
    """
    Factory function to create a gravity calculator for a given asteroid.
    
    Parameters
    ----------
    asteroid_name : str
        Name of asteroid ("A1", "A2", "A3", or "Castalia")
        
    Returns
    -------
    GravityCalculator
        Configured gravity calculator for the asteroid
    """
    from ..asteroid_data.coefficient_loader import get_coefficients_for_asteroid
    from ..config import get_asteroid_by_name
    
    # Get coefficients
    coefficients = get_coefficients_for_asteroid(asteroid_name)
    
    # Get asteroid parameters for mu and R_b
    asteroid_params = get_asteroid_by_name(asteroid_name)
    
    # Create calculator
    calculator = GravityCalculator(
        coefficients=coefficients,
        mu=asteroid_params.mu,
        R_b=asteroid_params.R_b,
        r0=1.0  # Default reference radius
    )
    
    return calculator


def test_gravity_calculator():
    """Run basic tests for the gravity calculator."""
    import matplotlib.pyplot as plt
    
    print("Testing Gravity Calculator...")
    
    # Test with asteroid A1 (triaxial ellipsoid)
    print("\n1. Testing with asteroid A1:")
    calculator = create_gravity_calculator_for_asteroid("A1")
    
    # Test at various positions
    test_positions = [
        np.array([2000.0, 0.0, 0.0]),  # Outside Brillouin
        np.array([500.0, 0.0, 0.0]),   # Inside Brillouin
        np.array([0.0, 0.0, 0.0]),     # At origin
    ]
    
    for i, pos in enumerate(test_positions):
        result = calculator.compute(pos)
        print(f"  Position {i}: {pos}")
        print(f"    Acceleration: {result.acceleration}")
        print(f"    Potential: {result.potential:.6f}")
        print(f"    Model used: {result.model_used}")
        print(f"    Inside Brillouin: {result.inside_brillouin}")
        
    # Test switching continuity
    print("\n2. Testing switching continuity:")
    continuity_test = calculator.test_switching_continuity(num_points=50)
    if continuity_test['switch_idx'] is not None:
        print(f"  Switching occurs at radius: {continuity_test['radii'][continuity_test['switch_idx']]:.2f} m")
        print(f"  Discontinuity magnitude: {continuity_test['discontinuity_magnitude']:.6e} m/s²")
    else:
        print("  No switching detected in test range")
        
    # Test point mass approximation
    print("\n3. Testing point mass approximation:")
    test_pos = np.array([1500.0, 500.0, 200.0])
    pm_test = calculator.test_point_mass_approximation(test_pos)
    print(f"  Test position: {test_pos}")
    print(f"  Distance: {pm_test['distance']:.2f} m")
    print(f"  Inside Brillouin: {pm_test['inside_brillouin']}")
    print(f"  Relative error: {pm_test['error_relative']:.6f}")
    
    # Test linearization
    print("\n4. Testing linearization:")
    A, c = calculator.linearize(test_pos)
    print(f"  Linearization matrix A (norm): {np.linalg.norm(A):.6e}")
    print(f"  Residual vector c (norm): {np.linalg.norm(c):.6e}")
    
    # Verify linearization approximation
    g_actual = calculator.acceleration(test_pos)
    g_linear = A @ test_pos + c
    lin_error = np.linalg.norm(g_actual - g_linear)
    print(f"  Linearization error: {lin_error:.6e} m/s²")
    
    print("\nGravity calculator tests completed!")
    
    return calculator


if __name__ == "__main__":
    # Run tests if executed directly
    test_gravity_calculator()