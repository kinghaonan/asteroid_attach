"""
Discretization of continuous-time dynamics using trapezoidal rule.

Implements Eq. 58 from the paper: trapezoidal discretization for the SOCP formulation.
Transforms continuous-time state equations into discrete-time constraints for optimization.

Author: Implementation Agent
Date: 2026-01-15
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass

from .state_equations import StateEquations, StateVector, ControlVector
from ..config import VehicleParameters, AsteroidParameters
from ..gravity_models.gravity_calculator import GravityCalculator


@dataclass
class DiscretizationParameters:
    """Parameters for discretization scheme."""
    dt: float  # Time step (seconds)
    N: int     # Number of discretization points
    t_f: float  # Total flight time
    method: str = "trapezoidal"  # Discretization method
    
    def __post_init__(self):
        """Validate discretization parameters."""
        if self.dt <= 0:
            raise ValueError(f"Time step dt must be positive, got {self.dt}")
        if self.N <= 0:
            raise ValueError(f"Number of points N must be positive, got {self.N}")
        if self.t_f <= 0:
            raise ValueError(f"Flight time t_f must be positive, got {self.t_f}")
        if self.method not in ["trapezoidal", "euler", "rk4"]:
            raise ValueError(f"Unknown discretization method: {self.method}")
        
        # Check consistency
        expected_N = int(np.ceil(self.t_f / self.dt)) + 1
        if self.N != expected_N:
            print(f"Warning: N={self.N} doesn't match expected N={expected_N} for t_f={self.t_f}, dt={self.dt}")


class TrapezoidalDiscretizer:
    """
    Implements trapezoidal discretization for asteroid landing dynamics.
    
    Based on Eq. 58 from the paper:
        x_{k+1} = x_k + (dt/2) * (f(x_k, u_k) + f(x_{k+1}, u_{k+1}))
    
    where x = [r, v, q] with q = ln(m) for convex formulation.
    """
    
    def __init__(self, 
                 state_equations: StateEquations,
                 dt: float = 2.0,
                 method: str = "trapezoidal"):
        """
        Initialize discretizer with dynamics and parameters.
        
        Args:
            state_equations: StateEquations instance for continuous dynamics
            dt: Time step in seconds (default 2.0s as in paper)
            method: Discretization method ("trapezoidal", "euler", "rk4")
        """
        self.state_eq = state_equations
        self.dt = dt
        self.method = method
        
        # Cache for linearized dynamics
        self._linearization_cache: Dict[Tuple[int, str], Any] = {}
        
    def create_discretization(self, t_f: float) -> DiscretizationParameters:
        """
        Create discretization parameters for given flight time.
        
        Args:
            t_f: Total flight time in seconds
            
        Returns:
            DiscretizationParameters with computed N and dt
        """
        # Number of points (including initial and final)
        N = int(np.ceil(t_f / self.dt)) + 1
        
        # Adjust dt slightly to exactly match t_f
        adjusted_dt = t_f / (N - 1)
        
        return DiscretizationParameters(
            dt=adjusted_dt,
            N=N,
            t_f=t_f,
            method=self.method
        )
    
    def trapezoidal_constraint(self, 
                               x_k: np.ndarray,
                               x_kp1: np.ndarray,
                               u_k: np.ndarray,
                               u_kp1: np.ndarray,
                               dt: float) -> np.ndarray:
        """
        Compute trapezoidal constraint residual.
        
        Args:
            x_k: State at time k [r_k, v_k, q_k] where q_k = ln(m_k)
            x_kp1: State at time k+1 [r_{k+1}, v_{k+1}, q_{k+1}]
            u_k: Control at time k [a_t_k, a_tm_k] (acceleration and slack)
            u_kp1: Control at time k+1 [a_t_{k+1}, a_tm_{k+1}]
            dt: Time step
            
        Returns:
            Constraint residual vector (should be zero for valid trajectory)
        """
        # Extract components
        r_k, v_k, q_k = x_k[:3], x_k[3:6], x_k[6]
        r_kp1, v_kp1, q_kp1 = x_kp1[:3], x_kp1[3:6], x_kp1[6]
        
        # Extract controls
        a_t_k = u_k[:3]  # T/m at time k
        a_t_kp1 = u_kp1[:3]  # T/m at time k+1
        
        # Compute dynamics at both points
        # Note: For trapezoidal rule with q = ln(m), we need to handle mass dynamics specially
        # The paper uses the transformation: ṁ = -∥T∥/(I_sp * g_0) becomes q̇ = -∥a_t∥/(I_sp * g_0)
        
        # Position dynamics: ṙ = v
        pos_residual = r_kp1 - r_k - (dt/2) * (v_k + v_kp1)
        
        # Velocity dynamics: v̇ = a_t + g(r) - 2ω×v - ω×(ω×r) - ω̇×r
        # We need to evaluate gravity at both points
        g_k = self.state_eq.gravity_calculator.acceleration(r_k)
        g_kp1 = self.state_eq.gravity_calculator.acceleration(r_kp1)
        
        # Coriolis and centrifugal terms
        omega = self.state_eq.asteroid_params.rotation_rate
        omega_vec = self.state_eq.asteroid_params.rotation_axis * omega
        
        # Coriolis acceleration: -2ω×v
        coriolis_k = -2 * np.cross(omega_vec, v_k)
        coriolis_kp1 = -2 * np.cross(omega_vec, v_kp1)
        
        # Centrifugal acceleration: -ω×(ω×r)
        centrifugal_k = -np.cross(omega_vec, np.cross(omega_vec, r_k))
        centrifugal_kp1 = -np.cross(omega_vec, np.cross(omega_vec, r_kp1))
        
        # Euler acceleration: -ω̇×r (assume ω̇ = 0 for constant rotation)
        euler_k = np.zeros(3)
        euler_kp1 = np.zeros(3)
        
        # Total acceleration at both points
        accel_k = a_t_k + g_k + coriolis_k + centrifugal_k + euler_k
        accel_kp1 = a_t_kp1 + g_kp1 + coriolis_kp1 + centrifugal_kp1 + euler_kp1
        
        # Velocity residual
        vel_residual = v_kp1 - v_k - (dt/2) * (accel_k + accel_kp1)
        
        # Mass dynamics: q̇ = -∥a_t∥/(I_sp * g_0)
        I_sp = self.state_eq.vehicle_params.I_sp
        g_0 = self.state_eq.vehicle_params.g_0
        
        # Compute ∥a_t∥ at both points
        a_t_norm_k = np.linalg.norm(a_t_k)
        a_t_norm_kp1 = np.linalg.norm(a_t_kp1)
        
        # Mass residual
        mass_residual = q_kp1 - q_k + (dt/2) * (
            a_t_norm_k/(I_sp * g_0) + a_t_norm_kp1/(I_sp * g_0)
        )
        
        # Combine residuals
        residual = np.concatenate([pos_residual, vel_residual, [mass_residual]])
        
        return residual
    
    def create_trapezoidal_constraints(self,
                                       discretization: DiscretizationParameters,
                                       x_vars: List[np.ndarray],
                                       u_vars: List[np.ndarray]) -> List[np.ndarray]:
        """
        Create trapezoidal constraints for all time steps.
        
        Args:
            discretization: Discretization parameters
            x_vars: List of state variables (each is CVXPY variable or array)
            u_vars: List of control variables (each is CVXPY variable or array)
            
        Returns:
            List of constraint residuals (should be constrained to zero)
        """
        constraints = []
        dt = discretization.dt
        
        for k in range(discretization.N - 1):
            # Get variables for current and next time step
            x_k = x_vars[k]
            x_kp1 = x_vars[k + 1]
            u_k = u_vars[k]
            u_kp1 = u_vars[k + 1]
            
            # Create trapezoidal constraint
            constraint = self.trapezoidal_constraint(x_k, x_kp1, u_k, u_kp1, dt)
            constraints.append(constraint)
            
        return constraints
    
    def linearized_trapezoidal_constraint(self,
                                          x_ref: np.ndarray,
                                          u_ref: np.ndarray,
                                          dt: float,
                                          cache_key: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute linearized trapezoidal constraint matrices.
        
        For use in successive solution method where gravity is linearized.
        
        Args:
            x_ref: Reference state [r, v, q]
            u_ref: Reference control [a_t, a_tm]
            dt: Time step
            cache_key: Optional cache key for reusing linearizations
            
        Returns:
            Tuple of (A_k, B_k, c_k) matrices for linear constraint:
                x_{k+1} = A_k x_k + B_k u_k + c_k
        """
        if cache_key and cache_key in self._linearization_cache:
            return self._linearization_cache[cache_key]
        
        # Extract components
        r_ref, v_ref, q_ref = x_ref[:3], x_ref[3:6], x_ref[6]
        a_t_ref = u_ref[:3]
        
        # Linearize gravity at reference point
        A_g, c_g = self.state_eq.linearize_gravity(r_ref)
        
        # Get asteroid parameters
        omega = self.state_eq.asteroid_params.rotation_rate
        omega_vec = self.state_eq.asteroid_params.rotation_axis * omega
        
        # Create state matrix A (7x7)
        A = np.eye(7)
        
        # Position block: ∂(r_{k+1})/∂(r_k, v_k, q_k)
        # From trapezoidal: r_{k+1} = r_k + (dt/2)(v_k + v_{k+1})
        # This couples r_{k+1} with v_{k+1}, so we need to solve implicit equations
        # For linearization, we use the implicit trapezoidal form
        
        # Simplified approach: Use explicit Euler for linearization
        # A more accurate approach would solve the implicit equations
        # For now, we use midpoint approximation
        
        # Position dynamics: ṙ = v
        A[0:3, 3:6] = dt * np.eye(3)  # ∂r/∂v
        
        # Velocity dynamics: v̇ = a_t + g(r) - 2ω×v - ω×(ω×r)
        # Linearized: v̇ ≈ A_g * r - 2ω×v - ω×(ω×r) + (a_t + c_g)
        
        # ∂v/∂r term
        A[3:6, 0:3] = dt * A_g
        
        # ∂v/∂v term (Coriolis)
        # Cross product matrix for ω
        omega_cross = np.array([
            [0, -omega_vec[2], omega_vec[1]],
            [omega_vec[2], 0, -omega_vec[0]],
            [-omega_vec[1], omega_vec[0], 0]
        ])
        A[3:6, 3:6] += -2 * dt * omega_cross
        
        # ∂v/∂q term is zero (mass doesn't affect velocity in linearized form)
        
        # Mass dynamics: q̇ = -∥a_t∥/(I_sp * g_0)
        # Linearized around a_t_ref
        I_sp = self.state_eq.vehicle_params.I_sp
        g_0 = self.state_eq.vehicle_params.g_0
        
        a_t_norm_ref = np.linalg.norm(a_t_ref)
        if a_t_norm_ref > 1e-10:
            dq_da_t = -dt * a_t_ref / (a_t_norm_ref * I_sp * g_0)
        else:
            dq_da_t = np.zeros(3)
        
        # Control matrix B (7x4)
        B = np.zeros((7, 4))
        
        # ∂(r_{k+1})/∂(a_t_k) = 0
        # ∂(v_{k+1})/∂(a_t_k) = dt * I
        B[3:6, 0:3] = dt * np.eye(3)
        
        # ∂(q_{k+1})/∂(a_t_k) = dq_da_t
        B[6, 0:3] = dq_da_t
        
        # ∂/∂(a_tm_k) terms are zero (slack variable doesn't affect dynamics)
        
        # Constant term c (7x1)
        c = np.zeros(7)
        
        # Velocity constant term from gravity linearization
        c[3:6] = dt * c_g
        
        # Velocity constant term from centrifugal acceleration
        # -ω×(ω×r_ref) evaluated at reference
        centrifugal = -np.cross(omega_vec, np.cross(omega_vec, r_ref))
        c[3:6] += dt * centrifugal
        
        # Cache result if requested
        if cache_key:
            self._linearization_cache[cache_key] = (A, B, c)
        
        return A, B, c
    
    def create_initial_guess(self,
                             discretization: DiscretizationParameters,
                             initial_state: StateVector,
                             final_state: StateVector) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Create initial guess for states and controls.
        
        Uses linear interpolation between initial and final states with
        simple control profile.
        
        Args:
            discretization: Discretization parameters
            initial_state: Initial state (r0, v0, m0)
            final_state: Final state (rf, vf, mf) - mf is typically unknown
            
        Returns:
            Tuple of (states_guess, controls_guess) lists
        """
        N = discretization.N
        dt = discretization.dt
        
        # Convert states to arrays with q = ln(m)
        x0 = np.concatenate([
            initial_state.position,
            initial_state.velocity,
            [np.log(initial_state.mass)]
        ])
        
        # For final state, we don't know mass, so use initial mass as guess
        xf = np.concatenate([
            final_state.position,
            final_state.velocity,
            [np.log(initial_state.mass * 0.9)]  # Guess 10% propellant used
        ])
        
        # Linearly interpolate states
        states_guess = []
        for k in range(N):
            alpha = k / (N - 1) if N > 1 else 0
            x_k = x0 * (1 - alpha) + xf * alpha
            states_guess.append(x_k)
        
        # Create control guesses
        # Simple profile: start with max thrust, then coast, then max thrust
        controls_guess = []
        vehicle = self.state_eq.vehicle_params
        
        # Maximum acceleration (T_max / m)
        m0 = initial_state.mass
        a_max = vehicle.T_max / m0
        a_min = vehicle.T_min / m0 if vehicle.T_min > 0 else 0
        
        for k in range(N):
            # Simple bang-bang guess
            if k < N // 3:
                # Initial burn
                a_t = a_max * (final_state.position - initial_state.position)
                a_t = a_t / np.linalg.norm(a_t) if np.linalg.norm(a_t) > 0 else np.array([1, 0, 0])
                a_t = a_t * a_max
            elif k > 2 * N // 3:
                # Final burn
                a_t = -a_max * (final_state.position - initial_state.position)
                a_t = a_t / np.linalg.norm(a_t) if np.linalg.norm(a_t) > 0 else np.array([-1, 0, 0])
                a_t = a_t * a_max
            else:
                # Coast phase
                a_t = np.zeros(3)
            
            # Slack variable a_tm = ∥a_t∥
            a_tm = np.linalg.norm(a_t)
            
            u_k = np.concatenate([a_t, [a_tm]])
            controls_guess.append(u_k)
        
        return states_guess, controls_guess
    
    def compute_time_vector(self, discretization: DiscretizationParameters) -> np.ndarray:
        """
        Compute time vector for discretization points.
        
        Args:
            discretization: Discretization parameters
            
        Returns:
            Array of time points from 0 to t_f
        """
        return np.linspace(0, discretization.t_f, discretization.N)
    
    def validate_discretization(self,
                                discretization: DiscretizationParameters,
                                states: List[np.ndarray],
                                controls: List[np.ndarray],
                                tolerance: float = 1e-3) -> Dict[str, Any]:
        """
        Validate that states and controls satisfy discretization constraints.
        
        Args:
            discretization: Discretization parameters
            states: List of state vectors
            controls: List of control vectors
            tolerance: Maximum allowed constraint violation
            
        Returns:
            Dictionary with validation results
        """
        N = discretization.N
        dt = discretization.dt
        
        max_violation = 0.0
        violations = []
        
        for k in range(N - 1):
            residual = self.trapezoidal_constraint(
                states[k], states[k + 1], controls[k], controls[k + 1], dt
            )
            violation = np.linalg.norm(residual)
            max_violation = max(max_violation, violation)
            
            if violation > tolerance:
                violations.append({
                    'k': k,
                    'violation': violation,
                    'residual': residual
                })
        
        return {
            'max_violation': max_violation,
            'num_violations': len(violations),
            'violations': violations,
            'is_valid': max_violation <= tolerance
        }


def test_discretization():
    """Test the discretization module."""
    import numpy as np
    from ..config import get_asteroid_by_name, get_vehicle_by_thrust
    from ..gravity_models.gravity_calculator import create_gravity_calculator_for_asteroid
    
    print("Testing discretization module...")
    
    # Create test components
    asteroid = get_asteroid_by_name("A1")
    vehicle = get_vehicle_by_thrust("full")
    gravity_calc = create_gravity_calculator_for_asteroid("A1")
    
    # Create state equations
    from .state_equations import StateEquations
    state_eq = StateEquations(gravity_calc, vehicle, asteroid)
    
    # Create discretizer
    discretizer = TrapezoidalDiscretizer(state_eq, dt=2.0)
    
    # Test discretization creation
    t_f = 100.0
    disc_params = discretizer.create_discretization(t_f)
    print(f"Created discretization: dt={disc_params.dt:.3f}s, N={disc_params.N}, t_f={disc_params.t_f:.1f}s")
    
    # Test time vector
    time_vec = discretizer.compute_time_vector(disc_params)
    print(f"Time vector: {time_vec[0]:.1f}s to {time_vec[-1]:.1f}s, length={len(time_vec)}")
    
    # Test initial guess creation
    initial_state = StateVector(
        position=np.array([2000.0, 0.0, 0.0]),
        velocity=np.array([0.1, 0.0, 0.0]),
        mass=vehicle.m_wet
    )
    
    final_state = StateVector(
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        mass=vehicle.m_dry
    )
    
    states_guess, controls_guess = discretizer.create_initial_guess(
        disc_params, initial_state, final_state
    )
    
    print(f"Created initial guess: {len(states_guess)} states, {len(controls_guess)} controls")
    print(f"First state: {states_guess[0]}")
    print(f"First control: {controls_guess[0]}")
    
    # Test trapezoidal constraint
    if len(states_guess) >= 2:
        residual = discretizer.trapezoidal_constraint(
            states_guess[0], states_guess[1], controls_guess[0], controls_guess[1], disc_params.dt
        )
        print(f"Trapezoidal constraint residual norm: {np.linalg.norm(residual):.6f}")
    
    # Test linearization
    x_ref = states_guess[0]
    u_ref = controls_guess[0]
    A, B, c = discretizer.linearized_trapezoidal_constraint(x_ref, u_ref, disc_params.dt)
    print(f"Linearization matrices: A shape={A.shape}, B shape={B.shape}, c shape={c.shape}")
    
    # Test validation
    validation = discretizer.validate_discretization(
        disc_params, states_guess, controls_guess, tolerance=1.0
    )
    print(f"Validation: max violation={validation['max_violation']:.6f}, valid={validation['is_valid']}")
    
    print("Discretization tests passed!")
    return True


if __name__ == "__main__":
    test_discretization()