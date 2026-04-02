"""
State equations for asteroid landing dynamics.

Implements the equations of motion (Eqs. 44-46) from the paper:
    ṙ = v
    v̇ = T/m + g(r) - 2ω×v - ω×(ω×r) - ω̇×r
    ṁ = -∥T∥/(I_sp * g_0)

where:
    r: position vector (m)
    v: velocity vector (m/s)
    m: mass (kg)
    T: thrust vector (N)
    g(r): gravitational acceleration (m/s²)
    ω: asteroid rotation rate vector (rad/s)
    ω̇: rotation rate derivative (assumed zero for constant rotation)
    I_sp: specific impulse (s)
    g_0: standard gravity (9.80665 m/s²)
"""

import numpy as np
from typing import Tuple, Optional, Callable
from dataclasses import dataclass

from ..gravity_models.gravity_calculator import GravityCalculator
from ..config import VehicleParameters, AsteroidParameters


@dataclass
class StateVector:
    """Container for the full state vector."""
    position: np.ndarray  # [x, y, z] (m)
    velocity: np.ndarray  # [vx, vy, vz] (m/s)
    mass: float  # (kg)
    
    def __post_init__(self):
        """Validate dimensions."""
        assert self.position.shape == (3,), "Position must be 3D vector"
        assert self.velocity.shape == (3,), "Velocity must be 3D vector"
        assert self.mass > 0, "Mass must be positive"
    
    def to_array(self) -> np.ndarray:
        """Convert to flat array [x, y, z, vx, vy, vz, m]."""
        return np.concatenate([self.position, self.velocity, [self.mass]])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'StateVector':
        """Create from flat array [x, y, z, vx, vy, vz, m]."""
        assert arr.shape == (7,), "Array must have 7 elements"
        return cls(
            position=arr[:3],
            velocity=arr[3:6],
            mass=arr[6]
        )
    
    def copy(self) -> 'StateVector':
        """Return a deep copy."""
        return StateVector(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass
        )


@dataclass
class ControlVector:
    """Container for the control (thrust) vector."""
    thrust: np.ndarray  # [Tx, Ty, Tz] (N)
    
    def __post_init__(self):
        """Validate dimensions."""
        assert self.thrust.shape == (3,), "Thrust must be 3D vector"
    
    def magnitude(self) -> float:
        """Return thrust magnitude ∥T∥."""
        return np.linalg.norm(self.thrust)
    
    def to_array(self) -> np.ndarray:
        """Convert to flat array [Tx, Ty, Tz]."""
        return self.thrust.copy()
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'ControlVector':
        """Create from flat array [Tx, Ty, Tz]."""
        assert arr.shape == (3,), "Array must have 3 elements"
        return cls(thrust=arr.copy())


class StateEquations:
    """
    Implements the state equations for asteroid landing dynamics.
    
    Equations (44-46) from the paper:
        ṙ = v
        v̇ = T/m + g(r) - 2ω×v - ω×(ω×r) - ω̇×r
        ṁ = -∥T∥/(I_sp * g_0)
    
    For constant rotation (ω̇ = 0), the equations simplify to:
        v̇ = T/m + g(r) - 2ω×v - ω×(ω×r)
    """
    
    def __init__(
        self,
        gravity_calculator: GravityCalculator,
        vehicle_params: VehicleParameters,
        asteroid_params: AsteroidParameters
    ):
        """
        Initialize state equations with physical parameters.
        
        Args:
            gravity_calculator: Gravity model for computing g(r)
            vehicle_params: Spacecraft parameters (mass, Isp, etc.)
            asteroid_params: Asteroid parameters (rotation rate, etc.)
        """
        self.gravity_calculator = gravity_calculator
        self.vehicle = vehicle_params
        self.asteroid = asteroid_params
        
        # Extract rotation parameters
        self.omega = asteroid_params.rotation_rate * asteroid_params.rotation_axis
        self.omega_dot = np.zeros(3)  # Assume constant rotation (ω̇ = 0)
        
        # Precompute omega cross product matrix
        self.omega_cross = self._cross_matrix(self.omega)
        self.omega_dot_cross = self._cross_matrix(self.omega_dot)
        
        # Precompute omega × (omega × r) term
        self.omega_squared = np.outer(self.omega, self.omega)
        self.omega_mag_squared = np.dot(self.omega, self.omega)
        self.identity = np.eye(3)
        
    def _cross_matrix(self, vector: np.ndarray) -> np.ndarray:
        """Create cross product matrix for vector."""
        return np.array([
            [0, -vector[2], vector[1]],
            [vector[2], 0, -vector[0]],
            [-vector[1], vector[0], 0]
        ])
    
    def compute_gravity(self, position: np.ndarray) -> np.ndarray:
        """
        Compute gravitational acceleration at given position.
        
        Args:
            position: Position vector (m) in asteroid-fixed frame
            
        Returns:
            Gravitational acceleration vector (m/s²)
        """
        result = self.gravity_calculator.compute(position)
        return result.acceleration
    
    def compute_coriolis_term(self, velocity: np.ndarray) -> np.ndarray:
        """
        Compute Coriolis acceleration: -2ω×v.
        
        Args:
            velocity: Velocity vector (m/s) in asteroid-fixed frame
            
        Returns:
            Coriolis acceleration vector (m/s²)
        """
        return -2 * np.cross(self.omega, velocity)
    
    def compute_centrifugal_term(self, position: np.ndarray) -> np.ndarray:
        """
        Compute centrifugal acceleration: -ω×(ω×r).
        
        Equivalent to: ω² r - (ω·r)ω
        
        Args:
            position: Position vector (m) in asteroid-fixed frame
            
        Returns:
            Centrifugal acceleration vector (m/s²)
        """
        # Method 1: Using cross products
        # return -np.cross(self.omega, np.cross(self.omega, position))
        
        # Method 2: Using matrix form (more efficient)
        omega_dot_r = np.dot(self.omega, position)
        return self.omega_mag_squared * position - omega_dot_r * self.omega
    
    def compute_euler_term(self, position: np.ndarray) -> np.ndarray:
        """
        Compute Euler acceleration: -ω̇×r.
        
        For constant rotation (ω̇ = 0), this is zero.
        
        Args:
            position: Position vector (m) in asteroid-fixed frame
            
        Returns:
            Euler acceleration vector (m/s²)
        """
        return -np.cross(self.omega_dot, position)
    
    def compute_rotation_terms(self, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """
        Compute total rotation acceleration terms.
        
        Returns: -2ω×v - ω×(ω×r) - ω̇×r
        """
        coriolis = self.compute_coriolis_term(velocity)
        centrifugal = self.compute_centrifugal_term(position)
        euler = self.compute_euler_term(position)
        return coriolis + centrifugal + euler
    
    def compute_mass_flow_rate(self, thrust_magnitude: float) -> float:
        """
        Compute mass flow rate: -∥T∥/(I_sp * g_0).
        
        Args:
            thrust_magnitude: Thrust magnitude ∥T∥ (N)
            
        Returns:
            Mass flow rate (kg/s)
        """
        if thrust_magnitude == 0:
            return 0.0
        return -thrust_magnitude / (self.vehicle.I_sp * self.vehicle.g_0)
    
    def evaluate(
        self,
        state: StateVector,
        control: ControlVector,
        time: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Evaluate the state derivatives at given state and control.
        
        Args:
            state: Current state (position, velocity, mass)
            control: Current control (thrust vector)
            time: Current time (s) - not used for time-invariant dynamics
            
        Returns:
            Tuple of (position_derivative, velocity_derivative, mass_derivative)
        """
        # Position derivative: ṙ = v
        position_dot = state.velocity
        
        # Compute gravitational acceleration
        gravity_accel = self.compute_gravity(state.position)
        
        # Compute rotation terms
        rotation_accel = self.compute_rotation_terms(state.position, state.velocity)
        
        # Thrust acceleration: T/m
        thrust_accel = control.thrust / state.mass
        
        # Velocity derivative: v̇ = T/m + g(r) - 2ω×v - ω×(ω×r) - ω̇×r
        velocity_dot = thrust_accel + gravity_accel + rotation_accel
        
        # Mass derivative: ṁ = -∥T∥/(I_sp * g_0)
        mass_dot = self.compute_mass_flow_rate(control.magnitude())
        
        return position_dot, velocity_dot, mass_dot
    
    def evaluate_flat(
        self,
        state_array: np.ndarray,
        control_array: np.ndarray,
        time: float = 0.0
    ) -> np.ndarray:
        """
        Evaluate state derivatives using flat arrays.
        
        Args:
            state_array: [x, y, z, vx, vy, vz, m]
            control_array: [Tx, Ty, Tz]
            time: Current time (s)
            
        Returns:
            State derivative array [ẋ, ẏ, ż, v̇x, v̇y, v̇z, ṁ]
        """
        state = StateVector.from_array(state_array)
        control = ControlVector.from_array(control_array)
        
        pos_dot, vel_dot, mass_dot = self.evaluate(state, control, time)
        
        return np.concatenate([pos_dot, vel_dot, [mass_dot]])
    
    def linearize_gravity(
        self,
        position: np.ndarray,
        reference_position: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize gravity about a reference position.
        
        Implements Eq. 57 from the paper:
            g(r) ≈ A(r_ref) * r + c(r_ref)
            
        For spherical gravity: A = -μ/r³ * I, c = 0
        For irregular gravity: A = ∂g/∂r evaluated at r_ref
        
        Args:
            position: Current position (for computing c)
            reference_position: Reference position for linearization
            
        Returns:
            Tuple (A_matrix, c_vector) where g(r) ≈ A*r + c
        """
        # Use the gravity calculator's linearization method
        return self.gravity_calculator.linearize(reference_position)
    
    def linearize_dynamics(
        self,
        state: StateVector,
        control: ControlVector,
        time: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Linearize dynamics about current state and control.
        
        Returns matrices A, B, d such that:
            ẋ ≈ A*x + B*u + d
            
        where x = [r, v, m] and u = [T]
        
        Args:
            state: Reference state
            control: Reference control
            time: Current time
            
        Returns:
            Tuple (A_matrix, B_matrix, d_vector)
        """
        # State dimension: 7 (3 pos + 3 vel + 1 mass)
        # Control dimension: 3 (thrust vector)
        A = np.zeros((7, 7))
        B = np.zeros((7, 3))
        d = np.zeros(7)
        
        # Position derivatives: ∂ṙ/∂r = 0, ∂ṙ/∂v = I, ∂ṙ/∂m = 0
        A[0:3, 3:6] = np.eye(3)  # ∂ṙ/∂v
        
        # Velocity derivatives
        # ∂v̇/∂r = ∂g/∂r - ∂(ω×(ω×r))/∂r - ∂(ω̇×r)/∂r
        # ∂(ω×(ω×r))/∂r = ω×ω× (cross product matrix)
        # For constant ω: ∂(ω×(ω×r))/∂r = ω²I - ωωᵀ
        
        # Linearize gravity
        A_gravity, c_gravity = self.linearize_gravity(state.position, state.position)
        
        # Centrifugal term Jacobian: ∂(ω×(ω×r))/∂r = ω²I - ωωᵀ
        centrifugal_jacobian = self.omega_mag_squared * self.identity - self.omega_squared
        
        # Euler term Jacobian (zero for constant ω)
        euler_jacobian = -self.omega_dot_cross
        
        # Total position Jacobian for velocity
        A[3:6, 0:3] = A_gravity - centrifugal_jacobian - euler_jacobian
        
        # ∂v̇/∂v = -2ω× (Coriolis term Jacobian)
        A[3:6, 3:6] = -2 * self.omega_cross
        
        # ∂v̇/∂m = -T/m²
        A[3:6, 6] = -control.thrust / (state.mass ** 2)
        
        # Control Jacobian: ∂v̇/∂T = I/m
        B[3:6, 0:3] = np.eye(3) / state.mass
        
        # Mass derivative: ∂ṁ/∂m = 0
        # ∂ṁ/∂T = -T/(I_sp*g_0*∥T∥) for T ≠ 0
        thrust_mag = control.magnitude()
        if thrust_mag > 1e-12:
            B[6, 0:3] = -control.thrust / (self.vehicle.I_sp * self.vehicle.g_0 * thrust_mag)
        
        # Constant term d = g(r_ref) - A_gravity*r_ref - (ω×(ω×r_ref)) + centrifugal_jacobian*r_ref
        # Actually, we already have c_gravity from linearization
        d[3:6] = c_gravity
        
        return A, B, d
    
    def create_integrator(
        self,
        dt: float,
        method: str = 'rk4'
    ) -> Callable[[StateVector, ControlVector, float], StateVector]:
        """
        Create an integrator function for propagating dynamics.
        
        Args:
            dt: Time step (s)
            method: Integration method ('euler', 'rk4')
            
        Returns:
            Function that takes (state, control, time) and returns next state
        """
        if method == 'euler':
            def integrator(state: StateVector, control: ControlVector, time: float) -> StateVector:
                pos_dot, vel_dot, mass_dot = self.evaluate(state, control, time)
                return StateVector(
                    position=state.position + pos_dot * dt,
                    velocity=state.velocity + vel_dot * dt,
                    mass=state.mass + mass_dot * dt
                )
        
        elif method == 'rk4':
            def integrator(state: StateVector, control: ControlVector, time: float) -> StateVector:
                # RK4 integration
                k1_pos, k1_vel, k1_mass = self.evaluate(state, control, time)
                
                state2 = StateVector(
                    position=state.position + k1_pos * dt/2,
                    velocity=state.velocity + k1_vel * dt/2,
                    mass=state.mass + k1_mass * dt/2
                )
                k2_pos, k2_vel, k2_mass = self.evaluate(state2, control, time + dt/2)
                
                state3 = StateVector(
                    position=state.position + k2_pos * dt/2,
                    velocity=state.velocity + k2_vel * dt/2,
                    mass=state.mass + k2_mass * dt/2
                )
                k3_pos, k3_vel, k3_mass = self.evaluate(state3, control, time + dt/2)
                
                state4 = StateVector(
                    position=state.position + k3_pos * dt,
                    velocity=state.velocity + k3_vel * dt,
                    mass=state.mass + k3_mass * dt
                )
                k4_pos, k4_vel, k4_mass = self.evaluate(state4, control, time + dt)
                
                next_pos = state.position + (k1_pos + 2*k2_pos + 2*k3_pos + k4_pos) * dt/6
                next_vel = state.velocity + (k1_vel + 2*k2_vel + 2*k3_vel + k4_vel) * dt/6
                next_mass = state.mass + (k1_mass + 2*k2_mass + 2*k3_mass + k4_mass) * dt/6
                
                return StateVector(position=next_pos, velocity=next_vel, mass=next_mass)
        
        else:
            raise ValueError(f"Unknown integration method: {method}")
        
        return integrator


def test_state_equations():
    """Test the state equations implementation."""
    import numpy as np
    from ..config import get_asteroid_by_name, get_vehicle_by_thrust
    from ..gravity_models.gravity_calculator import create_gravity_calculator_for_asteroid
    
    print("Testing StateEquations...")
    
    # Create test asteroid and vehicle
    asteroid = get_asteroid_by_name("A1")
    vehicle = get_vehicle_by_thrust("full")
    gravity_calc = create_gravity_calculator_for_asteroid("A1")
    
    # Create state equations
    dynamics = StateEquations(gravity_calc, vehicle, asteroid)
    
    # Test state
    test_state = StateVector(
        position=np.array([1000.0, 0.0, 0.0]),
        velocity=np.array([0.0, 0.1, 0.0]),
        mass=vehicle.m_wet
    )
    
    # Test control
    test_control = ControlVector(thrust=np.array([10.0, 0.0, 0.0]))
    
    # Evaluate derivatives
    pos_dot, vel_dot, mass_dot = dynamics.evaluate(test_state, test_control)
    
    print(f"Position derivative: {pos_dot}")
    print(f"Velocity derivative: {vel_dot}")
    print(f"Mass derivative: {mass_dot}")
    
    # Test flat array evaluation
    state_array = test_state.to_array()
    control_array = test_control.to_array()
    state_dot_array = dynamics.evaluate_flat(state_array, control_array)
    
    print(f"State derivative array: {state_dot_array}")
    
    # Test linearization
    A, B, d = dynamics.linearize_dynamics(test_state, test_control)
    print(f"A matrix shape: {A.shape}")
    print(f"B matrix shape: {B.shape}")
    print(f"d vector shape: {d.shape}")
    
    # Test integrator
    integrator = dynamics.create_integrator(dt=1.0, method='rk4')
    next_state = integrator(test_state, test_control, 0.0)
    print(f"Next state mass: {next_state.mass}")
    
    print("State equations test completed successfully!")
    return True


if __name__ == "__main__":
    test_state_equations()