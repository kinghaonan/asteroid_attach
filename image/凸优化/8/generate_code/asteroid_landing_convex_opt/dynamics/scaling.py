"""
Scaling system for numerical conditioning of SOCP solver.

Implements the scaling factors from Eq. 59 of the paper:
- Distance scaling: R_sc = γ (smallest semi-major axis)
- Velocity scaling: v_sc = √(R_sc * g_sc)
- Acceleration scaling: g_sc = μ / R_sc²
- Time scaling: t_sc = √(R_sc / g_sc)
- Mass scaling: m_sc = 1 (use q = ln(m))

This module provides scaling and unscaling functions for all variables
to improve numerical stability of the convex optimization solver.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any
from ..config import AsteroidParameters, VehicleParameters


@dataclass
class ScalingFactors:
    """Container for all scaling factors."""
    # Primary scaling factors
    R_sc: float  # Distance scaling (m)
    g_sc: float  # Acceleration scaling (m/s²)
    v_sc: float  # Velocity scaling (m/s)
    t_sc: float  # Time scaling (s)
    m_sc: float  # Mass scaling (kg) - typically 1.0
    
    # Derived factors
    mu_sc: float  # Gravitational parameter scaling (m³/s²)
    force_sc: float  # Force scaling (N)
    energy_sc: float  # Energy scaling (J)
    
    def __post_init__(self):
        """Validate scaling factors."""
        assert self.R_sc > 0, "Distance scaling must be positive"
        assert self.g_sc > 0, "Acceleration scaling must be positive"
        assert self.v_sc > 0, "Velocity scaling must be positive"
        assert self.t_sc > 0, "Time scaling must be positive"
        assert self.m_sc > 0, "Mass scaling must be positive"
        
        # Verify relationships
        assert np.abs(self.v_sc - np.sqrt(self.R_sc * self.g_sc)) < 1e-9, \
            f"v_sc should equal sqrt(R_sc * g_sc): {self.v_sc} vs {np.sqrt(self.R_sc * self.g_sc)}"
        assert np.abs(self.t_sc - np.sqrt(self.R_sc / self.g_sc)) < 1e-9, \
            f"t_sc should equal sqrt(R_sc / g_sc): {self.t_sc} vs {np.sqrt(self.R_sc / self.g_sc)}"
        assert np.abs(self.mu_sc - self.g_sc * self.R_sc**2) < 1e-9, \
            f"mu_sc should equal g_sc * R_sc^2: {self.mu_sc} vs {self.g_sc * self.R_sc**2}"
        assert np.abs(self.force_sc - self.m_sc * self.g_sc) < 1e-9, \
            f"force_sc should equal m_sc * g_sc: {self.force_sc} vs {self.m_sc * self.g_sc}"
        assert np.abs(self.energy_sc - self.m_sc * self.v_sc**2) < 1e-9, \
            f"energy_sc should equal m_sc * v_sc^2: {self.energy_sc} vs {self.m_sc * self.v_sc**2}"


class ScalingSystem:
    """
    Scaling system for numerical conditioning.
    
    Implements scaling and unscaling of variables according to Eq. 59.
    All variables are scaled before being passed to the SOCP solver,
    and unscaled after the solution is obtained.
    """
    
    def __init__(self, asteroid: AsteroidParameters, vehicle: VehicleParameters):
        """
        Initialize scaling system for given asteroid and vehicle.
        
        Args:
            asteroid: Asteroid parameters
            vehicle: Vehicle parameters
        """
        self.asteroid = asteroid
        self.vehicle = vehicle
        
        # Compute scaling factors according to Eq. 59
        # R_sc = γ (smallest semi-major axis)
        if hasattr(asteroid, 'semi_axes') and asteroid.semi_axes is not None:
            R_sc = min(asteroid.semi_axes)  # Smallest semi-major axis
        else:
            # Default to Brillouin sphere radius if semi-axes not available
            R_sc = asteroid.R_b
        
        # g_sc = μ / R_sc²
        g_sc = asteroid.mu / (R_sc**2)
        
        # v_sc = √(R_sc * g_sc)
        v_sc = np.sqrt(R_sc * g_sc)
        
        # t_sc = √(R_sc / g_sc)
        t_sc = np.sqrt(R_sc / g_sc)
        
        # m_sc = 1 (use q = ln(m))
        m_sc = 1.0
        
        # Derived factors
        mu_sc = g_sc * R_sc**2
        force_sc = m_sc * g_sc
        energy_sc = m_sc * v_sc**2
        
        self.factors = ScalingFactors(
            R_sc=R_sc,
            g_sc=g_sc,
            v_sc=v_sc,
            t_sc=t_sc,
            m_sc=m_sc,
            mu_sc=mu_sc,
            force_sc=force_sc,
            energy_sc=energy_sc
        )
        
        # Precompute inverse factors for unscaling
        self.inv_R_sc = 1.0 / R_sc
        self.inv_g_sc = 1.0 / g_sc
        self.inv_v_sc = 1.0 / v_sc
        self.inv_t_sc = 1.0 / t_sc
        self.inv_m_sc = 1.0 / m_sc
        self.inv_force_sc = 1.0 / force_sc
        self.inv_energy_sc = 1.0 / energy_sc
        
    def scale_position(self, r: np.ndarray) -> np.ndarray:
        """
        Scale position vector.
        
        Args:
            r: Position vector in meters (3,)
            
        Returns:
            Scaled position vector (dimensionless)
        """
        return r * self.inv_R_sc
    
    def unscale_position(self, r_scaled: np.ndarray) -> np.ndarray:
        """
        Unscale position vector.
        
        Args:
            r_scaled: Scaled position vector (dimensionless)
            
        Returns:
            Unscaled position vector in meters (3,)
        """
        return r_scaled * self.factors.R_sc
    
    def scale_velocity(self, v: np.ndarray) -> np.ndarray:
        """
        Scale velocity vector.
        
        Args:
            v: Velocity vector in m/s (3,)
            
        Returns:
            Scaled velocity vector (dimensionless)
        """
        return v * self.inv_v_sc
    
    def unscale_velocity(self, v_scaled: np.ndarray) -> np.ndarray:
        """
        Unscale velocity vector.
        
        Args:
            v_scaled: Scaled velocity vector (dimensionless)
            
        Returns:
            Unscaled velocity vector in m/s (3,)
        """
        return v_scaled * self.factors.v_sc
    
    def scale_acceleration(self, a: np.ndarray) -> np.ndarray:
        """
        Scale acceleration vector.
        
        Args:
            a: Acceleration vector in m/s² (3,)
            
        Returns:
            Scaled acceleration vector (dimensionless)
        """
        return a * self.inv_g_sc
    
    def unscale_acceleration(self, a_scaled: np.ndarray) -> np.ndarray:
        """
        Unscale acceleration vector.
        
        Args:
            a_scaled: Scaled acceleration vector (dimensionless)
            
        Returns:
            Unscaled acceleration vector in m/s² (3,)
        """
        return a_scaled * self.factors.g_sc
    
    def scale_time(self, t: float) -> float:
        """
        Scale time.
        
        Args:
            t: Time in seconds
            
        Returns:
            Scaled time (dimensionless)
        """
        return t * self.inv_t_sc
    
    def unscale_time(self, t_scaled: float) -> float:
        """
        Unscale time.
        
        Args:
            t_scaled: Scaled time (dimensionless)
            
        Returns:
            Unscaled time in seconds
        """
        return t_scaled * self.factors.t_sc
    
    def scale_mass(self, m: float) -> float:
        """
        Scale mass.
        
        Note: For the SOCP formulation, we use q = ln(m) rather than
        direct mass scaling. This function provides linear scaling
        for completeness.
        
        Args:
            m: Mass in kg
            
        Returns:
            Scaled mass (dimensionless)
        """
        return m * self.inv_m_sc
    
    def unscale_mass(self, m_scaled: float) -> float:
        """
        Unscale mass.
        
        Args:
            m_scaled: Scaled mass (dimensionless)
            
        Returns:
            Unscaled mass in kg
        """
        return m_scaled * self.factors.m_sc
    
    def scale_force(self, F: np.ndarray) -> np.ndarray:
        """
        Scale force vector.
        
        Args:
            F: Force vector in Newtons (3,)
            
        Returns:
            Scaled force vector (dimensionless)
        """
        return F * self.inv_force_sc
    
    def unscale_force(self, F_scaled: np.ndarray) -> np.ndarray:
        """
        Unscale force vector.
        
        Args:
            F_scaled: Scaled force vector (dimensionless)
            
        Returns:
            Unscaled force vector in Newtons (3,)
        """
        return F_scaled * self.factors.force_sc
    
    def scale_gravitational_parameter(self, mu: float) -> float:
        """
        Scale gravitational parameter.
        
        Args:
            mu: Gravitational parameter in m³/s²
            
        Returns:
            Scaled gravitational parameter (dimensionless)
        """
        return mu / self.factors.mu_sc
    
    def unscale_gravitational_parameter(self, mu_scaled: float) -> float:
        """
        Unscale gravitational parameter.
        
        Args:
            mu_scaled: Scaled gravitational parameter (dimensionless)
            
        Returns:
            Unscaled gravitational parameter in m³/s²
        """
        return mu_scaled * self.factors.mu_sc
    
    def scale_state_vector(self, state: np.ndarray) -> np.ndarray:
        """
        Scale a full state vector [r, v, m].
        
        Args:
            state: State vector [r_x, r_y, r_z, v_x, v_y, v_z, m]
            
        Returns:
            Scaled state vector
        """
        scaled = np.zeros_like(state)
        scaled[0:3] = self.scale_position(state[0:3])
        scaled[3:6] = self.scale_velocity(state[3:6])
        scaled[6] = self.scale_mass(state[6])
        return scaled
    
    def unscale_state_vector(self, state_scaled: np.ndarray) -> np.ndarray:
        """
        Unscale a full state vector [r, v, m].
        
        Args:
            state_scaled: Scaled state vector
            
        Returns:
            Unscaled state vector [r_x, r_y, r_z, v_x, v_y, v_z, m]
        """
        unscaled = np.zeros_like(state_scaled)
        unscaled[0:3] = self.unscale_position(state_scaled[0:3])
        unscaled[3:6] = self.unscale_velocity(state_scaled[3:6])
        unscaled[6] = self.unscale_mass(state_scaled[6])
        return unscaled
    
    def scale_control_vector(self, control: np.ndarray) -> np.ndarray:
        """
        Scale a control vector [a_t, a_tm].
        
        Args:
            control: Control vector [a_t_x, a_t_y, a_t_z, a_tm]
            
        Returns:
            Scaled control vector
        """
        scaled = np.zeros_like(control)
        scaled[0:3] = self.scale_acceleration(control[0:3])
        scaled[3] = self.scale_acceleration(np.array([control[3]]))[0]  # a_tm is scalar acceleration
        return scaled
    
    def unscale_control_vector(self, control_scaled: np.ndarray) -> np.ndarray:
        """
        Unscale a control vector [a_t, a_tm].
        
        Args:
            control_scaled: Scaled control vector
            
        Returns:
            Unscaled control vector [a_t_x, a_t_y, a_t_z, a_tm]
        """
        unscaled = np.zeros_like(control_scaled)
        unscaled[0:3] = self.unscale_acceleration(control_scaled[0:3])
        unscaled[3] = self.unscale_acceleration(np.array([control_scaled[3]]))[0]
        return unscaled
    
    def scale_thrust_bounds(self, T_min: float, T_max: float, m_ref: float) -> Tuple[float, float]:
        """
        Scale thrust bounds for SOCP formulation.
        
        According to the lossless convexification transformation:
        a_tm_min = T_min * e^{-q₀} * [1 - (q - q₀) + 0.5(q - q₀)²]
        a_tm_max = T_max * e^{-q₀} * [1 - (q - q₀)]
        
        This function scales the physical thrust bounds to the
        acceleration bounds used in the SOCP.
        
        Args:
            T_min: Minimum thrust in N
            T_max: Maximum thrust in N
            m_ref: Reference mass in kg (typically initial mass)
            
        Returns:
            Tuple of (scaled_min, scaled_max) acceleration bounds
        """
        # Convert thrust to acceleration bounds
        a_min = T_min / m_ref
        a_max = T_max / m_ref
        
        # Scale the acceleration bounds
        a_min_scaled = self.scale_acceleration(np.array([a_min]))[0]
        a_max_scaled = self.scale_acceleration(np.array([a_max]))[0]
        
        return a_min_scaled, a_max_scaled
    
    def get_scaling_factors(self) -> ScalingFactors:
        """
        Get the scaling factors.
        
        Returns:
            ScalingFactors object containing all scaling factors
        """
        return self.factors
    
    def get_scaling_matrix(self, n_states: int = 7, n_controls: int = 4) -> Dict[str, np.ndarray]:
        """
        Get diagonal scaling matrices for state and control variables.
        
        Useful for preconditioning optimization problems.
        
        Args:
            n_states: Number of state variables (default 7: r(3), v(3), m(1))
            n_controls: Number of control variables (default 4: a_t(3), a_tm(1))
            
        Returns:
            Dictionary with 'state_scale' and 'control_scale' diagonal matrices
        """
        # State scaling: [R_sc, R_sc, R_sc, v_sc, v_sc, v_sc, m_sc]
        state_scale = np.diag([
            self.factors.R_sc, self.factors.R_sc, self.factors.R_sc,
            self.factors.v_sc, self.factors.v_sc, self.factors.v_sc,
            self.factors.m_sc
        ])
        
        # Control scaling: [g_sc, g_sc, g_sc, g_sc]
        control_scale = np.diag([
            self.factors.g_sc, self.factors.g_sc, self.factors.g_sc,
            self.factors.g_sc
        ])
        
        return {
            'state_scale': state_scale,
            'control_scale': control_scale,
            'state_inv_scale': np.linalg.inv(state_scale),
            'control_inv_scale': np.linalg.inv(control_scale)
        }
    
    def validate_scaling(self, test_values: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
        """
        Validate scaling by checking round-trip scaling/unscaling.
        
        Args:
            test_values: Optional dictionary of test values to use.
                        If None, uses default test values.
                        
        Returns:
            Dictionary of validation results
        """
        if test_values is None:
            test_values = {
                'position': np.array([1000.0, 2000.0, 3000.0]),
                'velocity': np.array([10.0, -5.0, 2.0]),
                'acceleration': np.array([0.5, -0.2, 0.1]),
                'mass': 500.0,
                'time': 100.0,
                'force': np.array([50.0, 20.0, -10.0]),
                'mu': self.asteroid.mu
            }
        
        results = {}
        
        # Test position scaling
        r = test_values['position']
        r_scaled = self.scale_position(r)
        r_unscaled = self.unscale_position(r_scaled)
        results['position'] = np.allclose(r, r_unscaled, rtol=1e-10)
        
        # Test velocity scaling
        v = test_values['velocity']
        v_scaled = self.scale_velocity(v)
        v_unscaled = self.unscale_velocity(v_scaled)
        results['velocity'] = np.allclose(v, v_unscaled, rtol=1e-10)
        
        # Test acceleration scaling
        a = test_values['acceleration']
        a_scaled = self.scale_acceleration(a)
        a_unscaled = self.unscale_acceleration(a_scaled)
        results['acceleration'] = np.allclose(a, a_unscaled, rtol=1e-10)
        
        # Test mass scaling
        m = test_values['mass']
        m_scaled = self.scale_mass(m)
        m_unscaled = self.unscale_mass(m_scaled)
        results['mass'] = np.abs(m - m_unscaled) < 1e-10
        
        # Test time scaling
        t = test_values['time']
        t_scaled = self.scale_time(t)
        t_unscaled = self.unscale_time(t_scaled)
        results['time'] = np.abs(t - t_unscaled) < 1e-10
        
        # Test force scaling
        F = test_values['force']
        F_scaled = self.scale_force(F)
        F_unscaled = self.unscale_force(F_scaled)
        results['force'] = np.allclose(F, F_unscaled, rtol=1e-10)
        
        # Test gravitational parameter scaling
        mu = test_values['mu']
        mu_scaled = self.scale_gravitational_parameter(mu)
        mu_unscaled = self.unscale_gravitational_parameter(mu_scaled)
        results['mu'] = np.abs(mu - mu_unscaled) < 1e-10
        
        # Test state vector scaling
        state = np.concatenate([r, v, [m]])
        state_scaled = self.scale_state_vector(state)
        state_unscaled = self.unscale_state_vector(state_scaled)
        results['state_vector'] = np.allclose(state, state_unscaled, rtol=1e-10)
        
        # Test control vector scaling
        control = np.concatenate([a, [np.linalg.norm(a)]])  # a_tm = ||a_t||
        control_scaled = self.scale_control_vector(control)
        control_unscaled = self.unscale_control_vector(control_scaled)
        results['control_vector'] = np.allclose(control, control_unscaled, rtol=1e-10)
        
        return results


def create_default_scaling_system(asteroid_name: str = "A1", 
                                  vehicle_thrust: str = "full") -> ScalingSystem:
    """
    Create a scaling system with default parameters.
    
    Args:
        asteroid_name: Name of asteroid ("A1", "A2", "A3", or "Castalia")
        vehicle_thrust: Thrust level ("full" or "quarter")
        
    Returns:
        Configured ScalingSystem instance
    """
    from ..config import get_asteroid_by_name, get_vehicle_by_thrust
    
    asteroid = get_asteroid_by_name(asteroid_name)
    vehicle = get_vehicle_by_thrust(vehicle_thrust)
    
    return ScalingSystem(asteroid, vehicle)


def test_scaling_system() -> bool:
    """
    Test the scaling system implementation.
    
    Returns:
        True if all tests pass
    """
    print("Testing scaling system...")
    
    # Create a test scaling system
    from ..config import ASTEROID_A1, FULL_THRUST_VEHICLE
    scaling = ScalingSystem(ASTEROID_A1, FULL_THRUST_VEHICLE)
    
    # Check scaling factors
    factors = scaling.get_scaling_factors()
    print(f"  R_sc = {factors.R_sc:.3f} m")
    print(f"  g_sc = {factors.g_sc:.6f} m/s²")
    print(f"  v_sc = {factors.v_sc:.3f} m/s")
    print(f"  t_sc = {factors.t_sc:.3f} s")
    print(f"  m_sc = {factors.m_sc:.3f} kg")
    
    # Validate relationships
    assert np.abs(factors.v_sc - np.sqrt(factors.R_sc * factors.g_sc)) < 1e-9
    assert np.abs(factors.t_sc - np.sqrt(factors.R_sc / factors.g_sc)) < 1e-9
    print("  ✓ Scaling factor relationships correct")
    
    # Test scaling/unscaling
    validation = scaling.validate_scaling()
    all_passed = all(validation.values())
    
    for key, passed in validation.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {key}")
    
    if all_passed:
        print("  All scaling tests passed!")
    else:
        print("  Some scaling tests failed!")
    
    # Test scaling matrices
    matrices = scaling.get_scaling_matrix()
    assert matrices['state_scale'].shape == (7, 7)
    assert matrices['control_scale'].shape == (4, 4)
    print("  ✓ Scaling matrices have correct dimensions")
    
    # Test thrust bound scaling
    T_min = FULL_THRUST_VEHICLE.T_min
    T_max = FULL_THRUST_VEHICLE.T_max
    m_ref = FULL_THRUST_VEHICLE.m_wet
    a_min_scaled, a_max_scaled = scaling.scale_thrust_bounds(T_min, T_max, m_ref)
    
    # Verify scaling makes sense
    a_min = T_min / m_ref
    a_max = T_max / m_ref
    a_min_expected = scaling.scale_acceleration(np.array([a_min]))[0]
    a_max_expected = scaling.scale_acceleration(np.array([a_max]))[0]
    
    assert np.abs(a_min_scaled - a_min_expected) < 1e-9
    assert np.abs(a_max_scaled - a_max_expected) < 1e-9
    print("  ✓ Thrust bound scaling correct")
    
    return all_passed


if __name__ == "__main__":
    # Run tests if module is executed directly
    success = test_scaling_system()
    if success:
        print("\nScaling system implementation is correct!")
    else:
        print("\nScaling system implementation has issues!")
        exit(1)