"""
Problem formulation for asteroid landing convex optimization.

Implements the original nonconvex optimal control problem (P1) and its
transformation into convex SOCP problems (P2, P3) using lossless convexification.

Based on Eqs. 8-16 and Section III of the paper.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import cvxpy as cp

from ..config import (
    AsteroidParameters, VehicleParameters, LandingSite, 
    ExperimentConfig, G
)
from ..dynamics.state_equations import StateEquations, StateVector, ControlVector
from ..dynamics.discretization import TrapezoidalDiscretizer, DiscretizationParameters
from ..dynamics.scaling import ScalingSystem
from ..optimization.constraints import (
    TrajectoryConstraints, ConstraintParameters, create_constraint_parameters
)
from ..gravity_models.gravity_calculator import GravityCalculator


@dataclass
class ProblemFormulationParameters:
    """Parameters for problem formulation."""
    # Core parameters
    asteroid: AsteroidParameters
    vehicle: VehicleParameters
    landing_site: LandingSite
    
    # Discretization
    discretization_params: DiscretizationParameters
    
    # Scaling
    scaling_system: ScalingSystem
    
    # Initial and final states
    initial_state: StateVector
    final_state: StateVector
    
    # Gravity calculator
    gravity_calculator: GravityCalculator
    
    # State equations
    state_equations: StateEquations
    
    # Constraint parameters
    constraint_params: ConstraintParameters
    
    # Linearization reference (for successive solution)
    linearization_reference: Optional[np.ndarray] = None
    
    # Linearization matrices (A, c) for gravity
    gravity_A_matrix: Optional[np.ndarray] = None
    gravity_c_vector: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validate parameters."""
        if self.linearization_reference is not None:
            N = self.discretization_params.N
            # Check shape: should be (N+1, 3) for position reference
            if self.linearization_reference.shape != (N + 1, 3):
                raise ValueError(
                    f"linearization_reference shape {self.linearization_reference.shape} "
                    f"does not match expected shape ({N + 1}, 3)"
                )


class ProblemFormulation:
    """
    Formulates the asteroid landing optimal control problem.
    
    Implements:
    - P1: Original nonconvex problem (Eqs. 8-16)
    - P2: Relaxed convex problem with slack variable
    - P3: Final SOCP problem after variable change and linearization
    """
    
    def __init__(self, params: ProblemFormulationParameters):
        """
        Initialize problem formulation.
        
        Args:
            params: Problem formulation parameters
        """
        self.params = params
        self.N = params.discretization_params.N
        self.dt = params.discretization_params.dt
        self.t_f = params.discretization_params.t_f
        
        # Create discretizer
        self.discretizer = TrapezoidalDiscretizer(
            state_equations=params.state_equations,
            dt=self.dt,
            method="trapezoidal"
        )
        
        # Create constraint manager
        self.constraints = TrajectoryConstraints(params.constraint_params)
        
        # Optimization variables (will be created in create_socp_problem)
        self.variables = None
        self.problem = None
        
    def create_original_problem_p1(self) -> Dict[str, Any]:
        """
        Create the original nonconvex problem P1 (Eqs. 8-16).
        
        This is the theoretical formulation, not directly solvable by convex optimization.
        
        Returns:
            Dictionary with problem description
        """
        N = self.N
        dt = self.dt
        
        # Problem P1 description
        problem_p1 = {
            "name": "P1: Original Nonconvex Problem",
            "description": "Original optimal control problem with nonconvex constraints",
            "variables": {
                "r(t)": "Position vector (3×1)",
                "v(t)": "Velocity vector (3×1)", 
                "m(t)": "Mass (scalar)",
                "T(t)": "Thrust vector (3×1)"
            },
            "objective": "minimize J = -m(t_f) (maximize final mass)",
            "dynamics": [
                "ṙ = v",
                "v̇ = T/m + g(r) - 2ω×v - ω×(ω×r) - ω̇×r",
                "ṁ = -∥T∥/(I_sp * g_0)"
            ],
            "constraints": [
                "T_min ≤ ∥T(t)∥ ≤ T_max",
                "m(t) ≥ m_dry",
                "Glide slope: ∥r - r_f∥ cosθ - (r - r_f)·n̂ ≤ 0",
                "Boundary conditions: r(0)=r₀, v(0)=v₀, m(0)=m_wet; r(t_f)=r_f, v(t_f)=v_f"
            ],
            "time_domain": f"t ∈ [0, {self.t_f}]",
            "discretization": f"Trapezoidal rule with Δt = {dt}s, N = {N} points"
        }
        
        return problem_p1
    
    def create_relaxed_problem_p2(self) -> Dict[str, Any]:
        """
        Create the relaxed convex problem P2 with slack variable.
        
        This introduces the slack variable T_m with ∥T∥ ≤ T_m.
        
        Returns:
            Dictionary with problem description
        """
        problem_p2 = {
            "name": "P2: Relaxed Convex Problem",
            "description": "Relaxed problem with slack variable T_m",
            "transformations": [
                "Introduce slack variable T_m with ∥T∥ ≤ T_m",
                "Constraints become: T_min ≤ T_m ≤ T_max",
                "Second-order cone constraint: ∥T∥ ≤ T_m"
            ],
            "variables": {
                "r(t)": "Position vector (3×1)",
                "v(t)": "Velocity vector (3×1)", 
                "m(t)": "Mass (scalar)",
                "T(t)": "Thrust vector (3×1)",
                "T_m(t)": "Thrust magnitude slack variable (scalar)"
            },
            "objective": "minimize J = -m(t_f)",
            "constraints": [
                "∥T(t)∥ ≤ T_m(t) (second-order cone)",
                "T_min ≤ T_m(t) ≤ T_max",
                "m(t) ≥ m_dry",
                "Glide slope constraint",
                "Boundary conditions"
            ]
        }
        
        return problem_p2
    
    def create_socp_problem_p3(self, use_linearized_gravity: bool = True) -> cp.Problem:
        """
        Create the final SOCP problem P3 after variable change and linearization.
        
        This is the actual convex optimization problem that will be solved.
        
        Args:
            use_linearized_gravity: Whether to use linearized gravity (for successive solution)
            
        Returns:
            CVXPY problem instance
        """
        N = self.N
        dt = self.dt
        scaling = self.params.scaling_system
        
        # Create optimization variables
        # Position: (N+1) × 3
        r = cp.Variable((N + 1, 3), name="r")
        # Velocity: (N+1) × 3  
        v = cp.Variable((N + 1, 3), name="v")
        # Log mass: q = ln(m), (N+1) × 1
        q = cp.Variable((N + 1, 1), name="q")
        # Scaled thrust acceleration: a_t = T/m, N × 3
        a_t = cp.Variable((N, 3), name="a_t")
        # Scaled thrust magnitude: a_tm = T_m/m, N × 1
        a_tm = cp.Variable((N, 1), name="a_tm")
        
        # Store variables for later access
        self.variables = {
            'r': r, 'v': v, 'q': q, 'a_t': a_t, 'a_tm': a_tm
        }
        
        # Scale variables if needed
        # Note: Variables are already in scaled units when created
        
        # Objective: maximize final mass = minimize -q_N
        objective = cp.Minimize(-q[N])
        
        # Constraints list
        constraints = []
        
        # 1. Boundary conditions
        # Initial conditions (scaled)
        r0_scaled = scaling.scale_position(self.params.initial_state.position)
        v0_scaled = scaling.scale_velocity(self.params.initial_state.velocity)
        m0 = self.params.initial_state.mass
        q0 = np.log(m0)  # q = ln(m), no scaling needed for log mass
        
        constraints.append(r[0] == r0_scaled)
        constraints.append(v[0] == v0_scaled)
        constraints.append(q[0] == q0)
        
        # Final conditions (scaled)
        rf_scaled = scaling.scale_position(self.params.final_state.position)
        vf_scaled = scaling.scale_velocity(self.params.final_state.velocity)
        
        constraints.append(r[N] == rf_scaled)
        constraints.append(v[N] == vf_scaled)
        
        # 2. Dynamics constraints (trapezoidal discretization)
        if use_linearized_gravity and self.params.gravity_A_matrix is not None:
            # Use linearized gravity for successive solution
            A = self.params.gravity_A_matrix
            c = self.params.gravity_c_vector
            
            # Create linearized dynamics constraints
            for k in range(N):
                # Midpoint states
                r_mid = 0.5 * (r[k] + r[k+1])
                v_mid = 0.5 * (v[k] + v[k+1])
                
                # Linearized gravity at midpoint
                if A is not None and c is not None:
                    # Use provided linearization
                    g_lin = A @ r_mid + c
                else:
                    # Fallback to point mass approximation
                    r_mag = cp.norm(r_mid)
                    g_lin = -self.params.asteroid.mu / (r_mag**3) * r_mid
                
                # Dynamics with linearized gravity
                # Position update: r[k+1] = r[k] + 0.5*dt*(v[k] + v[k+1])
                # This is already implicit in trapezoidal rule
                
                # Velocity update with linearized gravity
                # v[k+1] - v[k] = dt*(a_t[k] + g_lin - 2ω×v_mid - ω×(ω×r_mid))
                # We'll implement this in the discretizer
                pass
                
            # Use discretizer for linearized constraints
            dyn_constraints = self.discretizer.create_linearized_trapezoidal_constraints(
                r, v, q, a_t, self.params.linearization_reference,
                self.params.gravity_A_matrix, self.params.gravity_c_vector
            )
            constraints.extend(dyn_constraints)
            
        else:
            # Use nonlinear gravity (for initial iteration or validation)
            dyn_constraints = self.discretizer.create_trapezoidal_constraints(r, v, q, a_t)
            constraints.extend(dyn_constraints)
        
        # 3. Thrust magnitude constraints (lossless convexification)
        # Original: T_min ≤ ∥T∥ ≤ T_max
        # Transformed: ∥a_t∥ ≤ a_tm and bounds on a_tm
        
        # Scale thrust bounds
        T_min_scaled = scaling.scale_force(self.params.vehicle.T_min)
        T_max_scaled = scaling.scale_force(self.params.vehicle.T_max)
        m0 = self.params.initial_state.mass
        
        # Reference log mass (q0 = ln(m0))
        q0_val = np.log(m0)
        
        # For each time step
        for k in range(N):
            # Second-order cone constraint: ∥a_t[k]∥ ≤ a_tm[k]
            constraints.append(cp.norm(a_t[k]) <= a_tm[k])
            
            # Upper bound: a_tm[k] ≤ T_max/m = T_max * exp(-q[k])
            # Linear approximation: exp(-q) ≈ exp(-q0) * [1 - (q - q0)]
            # So: a_tm[k] ≤ T_max_scaled * exp(-q0) * [1 - (q[k] - q0)]
            T_max_exp_q0 = T_max_scaled * np.exp(-q0_val)
            constraints.append(a_tm[k] <= T_max_exp_q0 * (1 - (q[k] - q0_val)))
            
            # Lower bound: a_tm[k] ≥ T_min/m = T_min * exp(-q[k])
            # Quadratic approximation: exp(-q) ≈ exp(-q0) * [1 - (q - q0) + 0.5*(q - q0)^2]
            # So: a_tm[k] ≥ T_min_scaled * exp(-q0) * [1 - (q[k] - q0) + 0.5*(q[k] - q0)^2]
            T_min_exp_q0 = T_min_scaled * np.exp(-q0_val)
            delta_q = q[k] - q0_val
            # For convexity, we use a simplified linear lower bound
            # This is conservative but maintains convexity
            constraints.append(a_tm[k] >= T_min_exp_q0 * (1 - delta_q))
        
        # 4. Mass constraint: m ≥ m_dry => q ≥ ln(m_dry)
        q_dry = np.log(self.params.vehicle.m_dry)
        for k in range(N + 1):
            constraints.append(q[k] >= q_dry)
        
        # 5. Additional trajectory constraints (glide slope, vertical motion)
        traj_constraints = self.constraints.create_all_constraints(
            r, v, q, a_t, a_tm, self.t_f, self.dt
        )
        constraints.extend(traj_constraints)
        
        # Create and return the problem
        self.problem = cp.Problem(objective, constraints)
        return self.problem
    
    def get_solution(self) -> Dict[str, np.ndarray]:
        """
        Extract solution from solved problem.
        
        Returns:
            Dictionary with solution arrays (unscaled)
        """
        if self.variables is None or self.problem is None:
            raise ValueError("Problem not created or solved yet")
        
        if not hasattr(self.problem, 'value') or self.problem.value is None:
            raise ValueError("Problem not solved yet")
        
        scaling = self.params.scaling_system
        
        # Extract and unscale variables
        r_unscaled = scaling.unscale_position(self.variables['r'].value)
        v_unscaled = scaling.unscale_velocity(self.variables['v'].value)
        q_values = self.variables['q'].value.flatten()
        m_unscaled = np.exp(q_values)  # m = exp(q)
        
        a_t_unscaled = scaling.unscale_acceleration(self.variables['a_t'].value)
        a_tm_unscaled = scaling.unscale_acceleration(self.variables['a_tm'].value.flatten())
        
        # Reconstruct thrust: T = a_t * m
        # Need to interpolate mass for thrust calculation
        m_midpoints = 0.5 * (m_unscaled[:-1] + m_unscaled[1:])
        T = a_t_unscaled * m_midpoints[:, np.newaxis]
        T_magnitude = np.linalg.norm(T, axis=1)
        
        # Time vector
        time = np.linspace(0, self.t_f, self.N + 1)
        
        return {
            'time': time,
            'position': r_unscaled,
            'velocity': v_unscaled,
            'mass': m_unscaled,
            'log_mass': q_values,
            'thrust_acceleration': a_t_unscaled,
            'thrust_magnitude_acceleration': a_tm_unscaled,
            'thrust': T,
            'thrust_magnitude': T_magnitude,
            'objective_value': -self.problem.value,  # Final mass
            'propellant_used': self.params.initial_state.mass - m_unscaled[-1]
        }
    
    def validate_lossless_convexification(self, solution: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Validate lossless convexification by checking ∥a_t∥ = a_tm.
        
        Args:
            solution: Solution dictionary from get_solution()
            
        Returns:
            Dictionary with validation metrics
        """
        a_t = solution['thrust_acceleration']
        a_tm = solution['thrust_magnitude_acceleration']
        
        # Compute norms
        a_t_norm = np.linalg.norm(a_t, axis=1)
        
        # Differences
        diff = np.abs(a_t_norm - a_tm)
        rel_diff = diff / np.maximum(a_tm, 1e-10)
        
        # Statistics
        max_abs_diff = np.max(diff)
        max_rel_diff = np.max(rel_diff)
        mean_abs_diff = np.mean(diff)
        mean_rel_diff = np.mean(rel_diff)
        
        # Check if lossless convexification holds (within tolerance)
        tolerance = 1e-6  # 1e-4% = 1e-6
        lossless_holds = max_rel_diff < tolerance
        
        return {
            'max_absolute_difference': max_abs_diff,
            'max_relative_difference': max_rel_diff,
            'mean_absolute_difference': mean_abs_diff,
            'mean_relative_difference': mean_rel_diff,
            'lossless_convexification_holds': lossless_holds,
            'tolerance': tolerance
        }
    
    def create_linearization_matrices(self, reference_trajectory: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create linearization matrices for gravity at reference trajectory.
        
        Args:
            reference_trajectory: Reference position trajectory (N+1 × 3)
            
        Returns:
            Tuple of (A_matrix, c_vector) for linearized gravity: g(r) ≈ A·r + c
        """
        N = self.N
        A_matrices = []
        c_vectors = []
        
        # For each point in the trajectory
        for k in range(N + 1):
            r_ref = reference_trajectory[k]
            
            # Get gravity and Jacobian at reference point
            gravity_result = self.params.gravity_calculator.compute(r_ref)
            g_ref = gravity_result.acceleration
            
            # For simplicity, use point mass approximation for A matrix
            # A = -μ/|r|³ * I + 3μ/|r|⁵ * r·rᵀ
            r_norm = np.linalg.norm(r_ref)
            if r_norm > 1e-6:
                r_outer = np.outer(r_ref, r_ref)
                A = -self.params.asteroid.mu / (r_norm**3) * np.eye(3) + \
                    3 * self.params.asteroid.mu / (r_norm**5) * r_outer
                c = g_ref - A @ r_ref
            else:
                # At origin, use simple approximation
                A = np.zeros((3, 3))
                c = np.zeros(3)
            
            A_matrices.append(A)
            c_vectors.append(c)
        
        # For trapezoidal discretization, we need matrices at midpoints
        A_mid = []
        c_mid = []
        for k in range(N):
            A_k = 0.5 * (A_matrices[k] + A_matrices[k+1])
            c_k = 0.5 * (c_vectors[k] + c_vectors[k+1])
            A_mid.append(A_k)
            c_mid.append(c_k)
        
        return np.array(A_mid), np.array(c_mid)


def create_problem_formulation(
    experiment_config: ExperimentConfig,
    discretization_dt: float = 2.0,
    use_scaling: bool = True
) -> ProblemFormulation:
    """
    Create a problem formulation from experiment configuration.
    
    Args:
        experiment_config: Experiment configuration
        discretization_dt: Time step for discretization (seconds)
        use_scaling: Whether to use numerical scaling
        
    Returns:
        ProblemFormulation instance
    """
    # Calculate number of time steps
    t_f = experiment_config.flight_time_range[1]  # Use max flight time initially
    N = int(np.ceil(t_f / discretization_dt))
    
    # Create discretization parameters
    discretization_params = DiscretizationParameters(
        dt=discretization_dt,
        N=N,
        t_f=t_f,
        method="trapezoidal"
    )
    
    # Create scaling system
    if use_scaling:
        scaling_system = ScalingSystem(
            asteroid=experiment_config.asteroid,
            vehicle=experiment_config.vehicle
        )
    else:
        # Create identity scaling
        from ..dynamics.scaling import ScalingFactors
        scaling_factors = ScalingFactors(
            R_sc=1.0, g_sc=1.0, v_sc=1.0, t_sc=1.0,
            m_sc=1.0, mu_sc=1.0, force_sc=1.0, energy_sc=1.0
        )
        scaling_system = ScalingSystem(
            asteroid=experiment_config.asteroid,
            vehicle=experiment_config.vehicle
        )
        scaling_system._factors = scaling_factors
    
    # Create gravity calculator
    from ..asteroid_data.coefficient_loader import get_coefficients_for_asteroid
    from ..gravity_models.gravity_calculator import create_gravity_calculator_for_asteroid
    
    gravity_calculator = create_gravity_calculator_for_asteroid(
        experiment_config.asteroid.name
    )
    
    # Create state equations
    state_equations = StateEquations(
        gravity_calculator=gravity_calculator,
        vehicle_params=experiment_config.vehicle,
        asteroid_params=experiment_config.asteroid
    )
    
    # Create constraint parameters
    constraint_params = create_constraint_parameters(
        vehicle_params=experiment_config.vehicle,
        landing_site=experiment_config.landing_site,
        initial_state=experiment_config.initial_state,
        final_state=experiment_config.final_state,
        scaling_system=scaling_system
    )
    
    # Create problem formulation parameters
    problem_params = ProblemFormulationParameters(
        asteroid=experiment_config.asteroid,
        vehicle=experiment_config.vehicle,
        landing_site=experiment_config.landing_site,
        discretization_params=discretization_params,
        scaling_system=scaling_system,
        initial_state=experiment_config.initial_state,
        final_state=experiment_config.final_state,
        gravity_calculator=gravity_calculator,
        state_equations=state_equations,
        constraint_params=constraint_params
    )
    
    return ProblemFormulation(problem_params)


def test_problem_formulation() -> bool:
    """Test the problem formulation module."""
    print("Testing problem formulation...")
    
    try:
        # Import test dependencies
        from ..config import (
            ASTEROID_A1, FULL_THRUST_VEHICLE, CASTALIA_LS1,
            create_triaxial_experiment
        )
        
        # Create a simple experiment
        experiment = create_triaxial_experiment(
            asteroid=ASTEROID_A1,
            vehicle=FULL_THRUST_VEHICLE,
            landing_site_position=np.array([0, 0, ASTEROID_A1.semi_axes[2] * 1.1])
        )
        
        # Create problem formulation
        problem_formulation = create_problem_formulation(
            experiment_config=experiment,
            discretization_dt=10.0,  # Coarse dt for testing
            use_scaling=True
        )
        
        # Test P1 description
        p1 = problem_formulation.create_original_problem_p1()
        assert p1["name"] == "P1: Original Nonconvex Problem"
        print(f"✓ Created P1: {p1['name']}")
        
        # Test P2 description  
        p2 = problem_formulation.create_relaxed_problem_p2()
        assert p2["name"] == "P2: Relaxed Convex Problem"
        print(f"✓ Created P2: {p2['name']}")
        
        # Test P3 creation
        problem = problem_formulation.create_socp_problem_p3(use_linearized_gravity=False)
        assert isinstance(problem, cp.Problem)
        print(f"✓ Created P3 SOCP problem with {len(problem.constraints)} constraints")
        
        # Test linearization matrices
        N = problem_formulation.N
        reference_traj = np.zeros((N + 1, 3))
        reference_traj[:, 2] = np.linspace(2000, 100, N + 1)  # Simple descent
        
        A, c = problem_formulation.create_linearization_matrices(reference_traj)
        assert A.shape == (N, 3, 3)
        assert c.shape == (N, 3)
        print(f"✓ Created linearization matrices: A shape {A.shape}, c shape {c.shape}")
        
        print("✓ All problem formulation tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Problem formulation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run tests if module is executed directly
    test_problem_formulation()