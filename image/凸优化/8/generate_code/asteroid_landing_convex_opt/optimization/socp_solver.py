"""
SOCP Solver for Asteroid Landing Convex Optimization

Implements the fixed-time Second-Order Cone Program (SOCP) solver using CVXPY,
corresponding to Eqs. 58 in the paper. This module solves the convexified
optimal control problem for a given flight time.

Author: Implementation based on "Trajectory Design Employing Convex Optimization 
for Landing on Irregularly Shaped Asteroids"
"""

import numpy as np
import cvxpy as cp
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging

from ..config import (
    VehicleParameters, AsteroidParameters, LandingSite, 
    ExperimentConfig, G, DEFAULT_SOLVER, SOLVER_SETTINGS
)
from ..dynamics.state_equations import StateVector, ControlVector
from ..dynamics.discretization import TrapezoidalDiscretizer, DiscretizationParameters
from ..dynamics.scaling import ScalingSystem
from ..optimization.constraints import TrajectoryConstraints, ConstraintParameters
from ..optimization.problem_formulation import ProblemFormulation, ProblemFormulationParameters
from ..gravity_models.gravity_calculator import GravityCalculator

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class SOCPSolution:
    """Container for SOCP solution results."""
    
    # Optimization variables
    position: np.ndarray  # Shape: (N+1, 3) - position at each time step
    velocity: np.ndarray  # Shape: (N+1, 3) - velocity at each time step
    mass: np.ndarray      # Shape: (N+1,) - mass at each time step
    thrust: np.ndarray    # Shape: (N, 3) - thrust vector at each time step
    thrust_magnitude: np.ndarray  # Shape: (N,) - thrust magnitude slack variable
    
    # Log-mass variables (used in convexification)
    q: np.ndarray         # Shape: (N+1,) - q = ln(m) at each time step
    a_t: np.ndarray       # Shape: (N, 3) - a_t = T/m (acceleration due to thrust)
    a_tm: np.ndarray      # Shape: (N,) - a_tm = T_m/m (thrust magnitude slack divided by mass)
    
    # Time information
    time_vector: np.ndarray  # Shape: (N+1,) - time at each discretization point
    dt: float                # Time step
    N: int                   # Number of discretization intervals
    t_f: float               # Total flight time
    
    # Optimization metrics
    objective_value: float   # Final objective value (-q_N = -ln(m_f))
    solve_time: float       # Time taken to solve SOCP (seconds)
    solver_status: str      # Solver status (e.g., 'optimal', 'infeasible')
    optimal: bool           # Whether solution is optimal
    
    # Additional information
    scaled_variables: Dict[str, Any] = field(default_factory=dict)  # Scaled variables before unscaling
    constraint_violations: Dict[str, float] = field(default_factory=dict)  # Constraint violations
    
    def __post_init__(self):
        """Validate solution dimensions."""
        assert self.position.shape == (self.N + 1, 3), f"Position shape mismatch: {self.position.shape}"
        assert self.velocity.shape == (self.N + 1, 3), f"Velocity shape mismatch: {self.velocity.shape}"
        assert self.mass.shape == (self.N + 1,), f"Mass shape mismatch: {self.mass.shape}"
        assert self.thrust.shape == (self.N, 3), f"Thrust shape mismatch: {self.thrust.shape}"
        assert self.thrust_magnitude.shape == (self.N,), f"Thrust magnitude shape mismatch: {self.thrust_magnitude.shape}"
        assert self.q.shape == (self.N + 1,), f"q shape mismatch: {self.q.shape}"
        assert self.a_t.shape == (self.N, 3), f"a_t shape mismatch: {self.a_t.shape}"
        assert self.a_tm.shape == (self.N,), f"a_tm shape mismatch: {self.a_tm.shape}"
        assert self.time_vector.shape == (self.N + 1,), f"Time vector shape mismatch: {self.time_vector.shape}"
    
    @property
    def propellant_used(self) -> float:
        """Calculate propellant used (kg)."""
        return self.mass[0] - self.mass[-1]
    
    @property
    def final_mass(self) -> float:
        """Final mass (kg)."""
        return self.mass[-1]
    
    @property
    def initial_mass(self) -> float:
        """Initial mass (kg)."""
        return self.mass[0]
    
    def get_state_at_time_index(self, idx: int) -> StateVector:
        """Get state vector at given time index."""
        return StateVector(
            position=self.position[idx],
            velocity=self.velocity[idx],
            mass=self.mass[idx]
        )
    
    def get_control_at_time_index(self, idx: int) -> ControlVector:
        """Get control vector at given time index."""
        return ControlVector(thrust=self.thrust[idx])
    
    def compute_thrust_profile(self) -> Dict[str, np.ndarray]:
        """Compute thrust magnitude profile and switching times."""
        thrust_mag = np.linalg.norm(self.thrust, axis=1)
        
        # Find switching times (where thrust magnitude changes significantly)
        thrust_diff = np.abs(np.diff(thrust_mag))
        switch_indices = np.where(thrust_diff > 0.1 * np.max(thrust_mag))[0]
        switch_times = self.time_vector[switch_indices]
        
        return {
            'thrust_magnitude': thrust_mag,
            'switch_indices': switch_indices,
            'switch_times': switch_times,
            'max_thrust': np.max(thrust_mag),
            'min_thrust': np.min(thrust_mag),
            'mean_thrust': np.mean(thrust_mag)
        }


class SOCPSolver:
    """
    SOCP Solver for fixed-time asteroid landing problems.
    
    Solves the convexified optimal control problem (P3) using CVXPY.
    Implements the trapezoidal discretization and handles scaling/unscaling
    for numerical conditioning.
    """
    
    def __init__(
        self,
        problem_formulation: ProblemFormulation,
        solver: str = DEFAULT_SOLVER,
        solver_settings: Optional[Dict[str, Any]] = None,
        use_scaling: bool = True,
        verbose: bool = False
    ):
        """
        Initialize SOCP solver.
        
        Args:
            problem_formulation: ProblemFormulation instance with all parameters
            solver: Solver to use ('MOSEK', 'SCS', 'ECOS', etc.)
            solver_settings: Additional solver settings
            use_scaling: Whether to use scaling for numerical conditioning
            verbose: Whether to print solver progress
        """
        self.problem_formulation = problem_formulation
        self.params = problem_formulation.params
        self.solver = solver
        self.solver_settings = solver_settings or SOLVER_SETTINGS.get(solver, {})
        self.use_scaling = use_scaling
        self.verbose = verbose
        
        # Get components from problem formulation
        self.scaling_system = self.params.scaling_system if use_scaling else None
        self.state_equations = self.params.state_equations
        self.gravity_calculator = self.params.gravity_calculator
        self.constraint_params = self.params.constraint_params
        self.discretization_params = self.params.discretization_params
        
        # Create discretizer
        self.discretizer = TrapezoidalDiscretizer(
            state_equations=self.state_equations,
            dt=self.discretization_params.dt,
            method=self.discretization_params.method
        )
        
        # Create constraints
        self.constraints = TrajectoryConstraints(constraint_params=self.constraint_params)
        
        # Initialize problem variables
        self.problem = None
        self.variables = {}
        self.constraint_list = []
        
        # Solution storage
        self.solution = None
        self.last_solve_time = 0.0
        
        logger.info(f"SOCP Solver initialized with solver={solver}, use_scaling={use_scaling}")
    
    def create_socp_problem(self) -> cp.Problem:
        """
        Create the SOCP problem (P3) using CVXPY.
        
        Returns:
            CVXPY Problem object
        """
        N = self.discretization_params.N
        dt = self.discretization_params.dt
        
        # Create CVXPY variables
        # Position (N+1, 3)
        r = cp.Variable((N + 1, 3), name="position")
        # Velocity (N+1, 3)
        v = cp.Variable((N + 1, 3), name="velocity")
        # Log-mass (N+1,)
        q = cp.Variable(N + 1, name="log_mass")
        # Thrust acceleration (N, 3)
        a_t = cp.Variable((N, 3), name="thrust_acceleration")
        # Thrust magnitude slack (N,)
        a_tm = cp.Variable(N, name="thrust_magnitude_slack")
        
        # Store variables
        self.variables = {
            'r': r,
            'v': v,
            'q': q,
            'a_t': a_t,
            'a_tm': a_tm
        }
        
        # Initialize constraint list
        constraints = []
        
        # 1. Dynamics constraints (trapezoidal discretization)
        logger.debug("Adding dynamics constraints...")
        dynamics_constraints = self.discretizer.create_trapezoidal_constraints(
            r, v, q, a_t, dt, self.params.gravity_A_matrix, self.params.gravity_c_vector
        )
        constraints.extend(dynamics_constraints)
        
        # 2. Thrust magnitude constraints (second-order cone)
        logger.debug("Adding thrust magnitude constraints...")
        for k in range(N):
            # SOC constraint: ||a_t[k]|| <= a_tm[k]
            constraints.append(cp.SOC(a_tm[k], a_t[k]))
        
        # 3. Thrust bounds (convex approximation)
        logger.debug("Adding thrust bound constraints...")
        thrust_constraints = self.constraints.add_thrust_magnitude_constraints(
            a_tm, q, self.params.linearization_reference.q_ref if hasattr(self.params.linearization_reference, 'q_ref') else q[0]
        )
        constraints.extend(thrust_constraints)
        
        # 4. Mass constraints
        logger.debug("Adding mass constraints...")
        mass_constraints = self.constraints.add_mass_constraints(q)
        constraints.extend(mass_constraints)
        
        # 5. Boundary conditions
        logger.debug("Adding boundary constraints...")
        boundary_constraints = self.constraints.add_boundary_constraints(
            r, v, q, 
            self.constraint_params.initial_state,
            self.constraint_params.final_state
        )
        constraints.extend(boundary_constraints)
        
        # 6. Glide slope constraints
        if self.constraint_params.glide_slope_active:
            logger.debug("Adding glide slope constraints...")
            glide_constraints = self.constraints.add_glide_slope_constraints(
                r, self.constraint_params.final_state.position
            )
            constraints.extend(glide_constraints)
        
        # 7. Vertical motion constraints
        if self.constraint_params.vertical_motion_active:
            logger.debug("Adding vertical motion constraints...")
            vertical_constraints = self.constraints.add_vertical_motion_constraints(
                r, self.constraint_params.final_state.position,
                self.discretizer.compute_time_vector()
            )
            constraints.extend(vertical_constraints)
        
        # 8. Scaling constraints (if using scaling)
        if self.use_scaling and self.scaling_system:
            logger.debug("Adding scaling constraints...")
            # Note: Variables are already in scaled units when passed to constraints
            # Scaling is handled by transforming the variables before creating constraints
        
        # Objective: maximize final mass = minimize -q_N
        objective = cp.Minimize(-q[N])
        
        # Create problem
        self.problem = cp.Problem(objective, constraints)
        self.constraint_list = constraints
        
        logger.info(f"SOCP problem created with {len(constraints)} constraints, N={N}, dt={dt}s")
        return self.problem
    
    def solve(self, warm_start: Optional[Dict[str, np.ndarray]] = None) -> SOCPSolution:
        """
        Solve the SOCP problem.
        
        Args:
            warm_start: Dictionary of variable values for warm start
            
        Returns:
            SOCPSolution object with solution data
        """
        if self.problem is None:
            self.create_socp_problem()
        
        # Apply warm start if provided
        if warm_start:
            self._apply_warm_start(warm_start)
        
        # Solve the problem
        logger.info(f"Solving SOCP with {self.solver} solver...")
        start_time = time.time()
        
        try:
            # Use solver settings
            solver_kwargs = self.solver_settings.copy()
            if self.verbose:
                solver_kwargs['verbose'] = True
            
            # Solve
            self.problem.solve(solver=self.solver, **solver_kwargs)
            solve_time = time.time() - start_time
            self.last_solve_time = solve_time
            
            # Check solution status
            status = self.problem.status
            optimal = status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]
            
            logger.info(f"Solver finished with status: {status}, solve time: {solve_time:.2f}s")
            
            if not optimal:
                logger.warning(f"Solver did not find optimal solution. Status: {status}")
                if status == cp.INFEASIBLE:
                    logger.error("Problem is infeasible!")
                elif status == cp.UNBOUNDED:
                    logger.error("Problem is unbounded!")
            
            # Extract solution
            solution = self._extract_solution(solve_time, status, optimal)
            self.solution = solution
            
            # Validate lossless convexification
            if optimal:
                self._validate_lossless_convexification(solution)
            
            return solution
            
        except Exception as e:
            logger.error(f"Error solving SOCP: {e}")
            raise
    
    def _apply_warm_start(self, warm_start: Dict[str, np.ndarray]):
        """Apply warm start values to CVXPY variables."""
        for var_name, value in warm_start.items():
            if var_name in self.variables:
                var = self.variables[var_name]
                if var.value is None or var.shape == value.shape:
                    var.value = value
                    logger.debug(f"Applied warm start to {var_name}")
                else:
                    logger.warning(f"Warm start shape mismatch for {var_name}: {var.shape} vs {value.shape}")
    
    def _extract_solution(self, solve_time: float, status: str, optimal: bool) -> SOCPSolution:
        """Extract solution from CVXPY variables and convert to physical units."""
        N = self.discretization_params.N
        dt = self.discretization_params.dt
        t_f = self.discretization_params.t_f
        
        # Get scaled variables
        r_scaled = self.variables['r'].value
        v_scaled = self.variables['v'].value
        q_scaled = self.variables['q'].value
        a_t_scaled = self.variables['a_t'].value
        a_tm_scaled = self.variables['a_tm'].value
        
        # Store scaled variables
        scaled_vars = {
            'r': r_scaled.copy(),
            'v': v_scaled.copy(),
            'q': q_scaled.copy(),
            'a_t': a_t_scaled.copy(),
            'a_tm': a_tm_scaled.copy()
        }
        
        # Unscale if using scaling
        if self.use_scaling and self.scaling_system:
            r = self.scaling_system.unscale_position(r_scaled)
            v = self.scaling_system.unscale_velocity(v_scaled)
            # q is dimensionless, no scaling needed
            a_t = self.scaling_system.unscale_acceleration(a_t_scaled)
            a_tm = self.scaling_system.unscale_acceleration(a_tm_scaled)
        else:
            r = r_scaled
            v = v_scaled
            q = q_scaled
            a_t = a_t_scaled
            a_tm = a_tm_scaled
        
        # Convert log-mass to mass
        mass = np.exp(q)
        
        # Compute thrust from a_t and mass
        # T = a_t * m, but careful with dimensions: a_t is defined at interval midpoints
        thrust = np.zeros((N, 3))
        thrust_magnitude = np.zeros(N)
        for k in range(N):
            # Use average mass over interval for thrust calculation
            m_avg = 0.5 * (mass[k] + mass[k+1])
            thrust[k] = a_t[k] * m_avg
            thrust_magnitude[k] = a_tm[k] * m_avg
        
        # Create time vector
        time_vector = self.discretizer.compute_time_vector()
        
        # Compute constraint violations
        constraint_violations = self._compute_constraint_violations(
            r, v, mass, thrust, thrust_magnitude, q, a_t, a_tm
        )
        
        # Create solution object
        solution = SOCPSolution(
            position=r,
            velocity=v,
            mass=mass,
            thrust=thrust,
            thrust_magnitude=thrust_magnitude,
            q=q,
            a_t=a_t,
            a_tm=a_tm,
            time_vector=time_vector,
            dt=dt,
            N=N,
            t_f=t_f,
            objective_value=self.problem.value if self.problem.value is not None else float('inf'),
            solve_time=solve_time,
            solver_status=status,
            optimal=optimal,
            scaled_variables=scaled_vars,
            constraint_violations=constraint_violations
        )
        
        return solution
    
    def _compute_constraint_violations(
        self, r, v, mass, thrust, thrust_magnitude, q, a_t, a_tm
    ) -> Dict[str, float]:
        """Compute constraint violations for validation."""
        violations = {}
        N = self.discretization_params.N
        
        # 1. Thrust magnitude SOC constraint violations
        soc_violations = []
        for k in range(N):
            a_t_norm = np.linalg.norm(a_t[k])
            soc_violation = max(0, a_t_norm - a_tm[k])
            soc_violations.append(soc_violation)
        violations['max_soc_violation'] = max(soc_violations) if soc_violations else 0.0
        violations['mean_soc_violation'] = np.mean(soc_violations) if soc_violations else 0.0
        
        # 2. Thrust bound violations
        vehicle = self.params.vehicle
        T_min = vehicle.T_min
        T_max = vehicle.T_max
        
        thrust_mags = np.linalg.norm(thrust, axis=1)
        lower_violations = np.maximum(0, T_min - thrust_mags)
        upper_violations = np.maximum(0, thrust_mags - T_max)
        
        violations['max_thrust_lower_violation'] = np.max(lower_violations)
        violations['max_thrust_upper_violation'] = np.max(upper_violations)
        violations['mean_thrust_violation'] = np.mean(np.concatenate([lower_violations, upper_violations]))
        
        # 3. Mass constraint violations
        m_dry = vehicle.m_dry
        mass_violations = np.maximum(0, m_dry - mass)
        violations['max_mass_violation'] = np.max(mass_violations)
        violations['mean_mass_violation'] = np.mean(mass_violations)
        
        # 4. Boundary condition violations
        r0 = self.constraint_params.initial_state.position
        v0 = self.constraint_params.initial_state.velocity
        m0 = self.constraint_params.initial_state.mass
        rf = self.constraint_params.final_state.position
        vf = self.constraint_params.final_state.velocity
        
        violations['initial_position_error'] = np.linalg.norm(r[0] - r0)
        violations['initial_velocity_error'] = np.linalg.norm(v[0] - v0)
        violations['initial_mass_error'] = abs(mass[0] - m0)
        violations['final_position_error'] = np.linalg.norm(r[-1] - rf)
        violations['final_velocity_error'] = np.linalg.norm(v[-1] - vf)
        
        return violations
    
    def _validate_lossless_convexification(self, solution: SOCPSolution):
        """
        Validate lossless convexification by checking if ||a_t|| ≈ a_tm.
        
        According to the paper, the error should be < 1e-4% for lossless convexification.
        """
        max_error = 0.0
        max_error_idx = -1
        
        for k in range(solution.N):
            a_t_norm = np.linalg.norm(solution.a_t[k])
            a_tm_val = solution.a_tm[k]
            
            if a_tm_val > 1e-12:  # Avoid division by zero
                error = abs(a_t_norm - a_tm_val) / a_tm_val * 100  # Percentage error
                if error > max_error:
                    max_error = error
                    max_error_idx = k
        
        logger.info(f"Lossless convexification validation: max error = {max_error:.2e}% at k={max_error_idx}")
        
        if max_error > 1e-4:  # 1e-4% threshold from paper
            logger.warning(f"Lossless convexification error ({max_error:.2e}%) exceeds threshold (1e-4%)")
        else:
            logger.info("Lossless convexification validated successfully")
    
    def get_initial_guess(self) -> Dict[str, np.ndarray]:
        """
        Generate initial guess for optimization variables.
        
        Returns:
            Dictionary with initial guess values for all variables
        """
        N = self.discretization_params.N
        
        # Get initial guess from discretizer
        initial_guess = self.discretizer.create_initial_guess()
        
        # Convert to appropriate format for warm start
        warm_start = {
            'r': initial_guess['position'],
            'v': initial_guess['velocity'],
            'q': initial_guess['log_mass'],
            'a_t': np.zeros((N, 3)),
            'a_tm': np.zeros(N)
        }
        
        # Initialize a_t and a_tm based on initial thrust estimate
        vehicle = self.params.vehicle
        m0 = self.constraint_params.initial_state.mass
        
        # Initial thrust estimate (hover thrust to counteract gravity)
        r0 = self.constraint_params.initial_state.position
        g0 = self.gravity_calculator.acceleration(r0)
        T_initial = -m0 * g0  # Thrust to counteract gravity
        
        # Bound thrust by vehicle limits
        T_mag = np.linalg.norm(T_initial)
        if T_mag > vehicle.T_max:
            T_initial = T_initial * (vehicle.T_max / T_mag)
        elif T_mag < vehicle.T_min:
            T_initial = T_initial * (vehicle.T_min / max(T_mag, 1e-6))
        
        # Convert to a_t and a_tm
        for k in range(N):
            # Use linear interpolation from initial to final
            alpha = k / max(N-1, 1)
            T_k = (1 - alpha) * T_initial  # Reduce thrust as we approach landing
            
            # Compute a_t and a_tm
            m_k = initial_guess['mass'][k]
            m_kp1 = initial_guess['mass'][k+1]
            m_avg = 0.5 * (m_k + m_kp1)
            
            warm_start['a_t'][k] = T_k / m_avg
            warm_start['a_tm'][k] = np.linalg.norm(T_k) / m_avg
        
        return warm_start


def solve_fixed_time_socp(
    experiment_config: ExperimentConfig,
    t_f: float,
    dt: float = 2.0,
    solver: str = DEFAULT_SOLVER,
    use_scaling: bool = True,
    verbose: bool = False
) -> SOCPSolution:
    """
    Convenience function to solve fixed-time SOCP for given experiment.
    
    Args:
        experiment_config: Experiment configuration
        t_f: Flight time (seconds)
        dt: Time step for discretization (seconds)
        solver: Solver to use
        use_scaling: Whether to use scaling
        verbose: Verbose output
        
    Returns:
        SOCPSolution object
    """
    # Create problem formulation
    problem_formulation = ProblemFormulation.create_problem_formulation(
        experiment_config, discretization_dt=dt, use_scaling=use_scaling
    )
    
    # Update flight time
    problem_formulation.params.discretization_params.t_f = t_f
    N = int(np.ceil(t_f / dt))
    problem_formulation.params.discretization_params.N = N
    
    # Create and solve SOCP
    solver = SOCPSolver(
        problem_formulation=problem_formulation,
        solver=solver,
        use_scaling=use_scaling,
        verbose=verbose
    )
    
    solution = solver.solve()
    return solution


def test_socp_solver() -> bool:
    """Test the SOCP solver with a simple example."""
    import numpy as np
    from ..config import create_triaxial_experiment, ASTEROID_A1, FULL_THRUST_VEHICLE
    
    print("Testing SOCP solver...")
    
    try:
        # Create simple experiment
        experiment = create_triaxial_experiment(
            asteroid=ASTEROID_A1,
            vehicle=FULL_THRUST_VEHICLE
        )
        
        # Set flight time
        t_f = 300.0  # 5 minutes
        
        # Solve SOCP
        print(f"Solving SOCP for asteroid A1, t_f={t_f}s...")
        solution = solve_fixed_time_socp(
            experiment_config=experiment,
            t_f=t_f,
            dt=5.0,  # Coarse dt for faster testing
            solver='ECOS',  # Use ECOS for testing (no license required)
            use_scaling=True,
            verbose=False
        )
        
        # Check solution
        print(f"Solution status: {solution.solver_status}")
        print(f"Optimal: {solution.optimal}")
        print(f"Solve time: {solution.solve_time:.2f}s")
        print(f"Propellant used: {solution.propellant_used:.2f} kg")
        print(f"Final mass: {solution.final_mass:.2f} kg")
        
        # Check constraint violations
        print("\nConstraint violations:")
        for name, value in solution.constraint_violations.items():
            if 'error' in name or 'violation' in name:
                print(f"  {name}: {value:.2e}")
        
        # Basic validation
        if solution.optimal:
            print("\nSOCP solver test PASSED")
            return True
        else:
            print("\nSOCP solver test FAILED - Solution not optimal")
            return False
            
    except Exception as e:
        print(f"SOCP solver test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run tests if executed directly
    test_socp_solver()