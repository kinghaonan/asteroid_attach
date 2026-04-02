"""
Successive solution solver for asteroid landing convex optimization.

Implements Algorithm 1 from the paper: iterative linearization of nonlinear gravity
to solve sequence of SOCPs until convergence (position tolerance < 0.5m).
"""

import numpy as np
import cvxpy as cp
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import time

from ..config import (
    ExperimentConfig, VehicleParameters, AsteroidParameters, LandingSite,
    DEFAULT_SOLVER, SOLVER_SETTINGS, VALIDATION_TOLERANCES
)
from ..dynamics.state_equations import StateVector, ControlVector
from ..dynamics.discretization import TrapezoidalDiscretizer, DiscretizationParameters
from ..dynamics.scaling import ScalingSystem
from ..optimization.constraints import TrajectoryConstraints, ConstraintParameters
from ..optimization.problem_formulation import ProblemFormulation, ProblemFormulationParameters
from ..optimization.socp_solver import SOCPSolver, SOCPSolution
from ..gravity_models.gravity_calculator import GravityCalculator

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class SuccessiveSolutionParameters:
    """Parameters for successive solution algorithm."""
    max_iterations: int = 10
    position_tolerance: float = 0.5  # meters
    max_linearization_error: float = 1e-4
    use_adaptive_linearization: bool = True
    adaptive_tolerance_factor: float = 0.1
    store_intermediate_solutions: bool = False
    verbose: bool = True
    
    def __post_init__(self):
        """Validate parameters."""
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.position_tolerance <= 0:
            raise ValueError("position_tolerance must be positive")
        if self.max_linearization_error <= 0:
            raise ValueError("max_linearization_error must be positive")


@dataclass
class SuccessiveSolutionResult:
    """Results from successive solution algorithm."""
    converged: bool
    iteration_count: int
    final_position_error: float
    solution: SOCPSolution
    intermediate_solutions: List[SOCPSolution] = field(default_factory=list)
    iteration_errors: List[float] = field(default_factory=list)
    computation_time: float = 0.0
    linearization_errors: List[float] = field(default_factory=list)
    
    def summary(self) -> str:
        """Return summary string."""
        return (f"Successive solution: converged={self.converged}, "
                f"iterations={self.iteration_count}, "
                f"final_error={self.final_position_error:.4f}m, "
                f"time={self.computation_time:.2f}s")


class SuccessiveSolver:
    """
    Successive solution solver for nonlinear gravity fields.
    
    Implements Algorithm 1 from the paper:
    1. Initialize reference trajectory (initial guess)
    2. While max∥r_new(t) - r_ref(t)∥ > tolerance:
        a. Linearize gravity: g(r) ≈ A(r_ref)r + c(r_ref)
        b. Form SOCP with linearized dynamics
        c. Solve SOCP to get new trajectory r_new(t)
        d. Update r_ref = r_new
    3. Return converged trajectory
    """
    
    def __init__(
        self,
        experiment_config: ExperimentConfig,
        discretization_dt: float = 2.0,
        solver: str = DEFAULT_SOLVER,
        solver_settings: Optional[Dict] = None,
        use_scaling: bool = True,
        params: Optional[SuccessiveSolutionParameters] = None
    ):
        """
        Initialize successive solver.
        
        Args:
            experiment_config: Complete experiment configuration
            discretization_dt: Time step for discretization (seconds)
            solver: SOCP solver name ('MOSEK', 'SCS', etc.)
            solver_settings: Solver-specific settings
            use_scaling: Whether to use numerical scaling
            params: Successive solution parameters
        """
        self.experiment_config = experiment_config
        self.discretization_dt = discretization_dt
        self.solver = solver
        self.solver_settings = solver_settings or SOLVER_SETTINGS.get(solver, {})
        self.use_scaling = use_scaling
        self.params = params or SuccessiveSolutionParameters()
        
        # Initialize components
        self.gravity_calculator = GravityCalculator(
            coefficients=experiment_config.asteroid.castalia_coeffs_file,
            mu=experiment_config.asteroid.mu,
            R_b=experiment_config.asteroid.R_b
        )
        
        self.scaling_system = ScalingSystem(
            asteroid=experiment_config.asteroid,
            vehicle=experiment_config.vehicle
        ) if use_scaling else None
        
        # Create discretization parameters
        self.discretization_params = DiscretizationParameters(
            dt=discretization_dt,
            N=None,  # Will be set based on flight time
            t_f=None,  # Will be set when solving
            method="trapezoidal"
        )
        
        # Create constraint parameters
        self.constraint_params = ConstraintParameters(
            glide_slope_angle=experiment_config.landing_site.glide_slope_angle,
            glide_slope_active=True,
            vertical_motion_active=True,
            vertical_motion_start_time=0.0,
            vertical_motion_min_altitude=0.0,
            min_mass=experiment_config.vehicle.m_dry,
            initial_state=experiment_config.initial_state,
            final_state=experiment_config.final_state,
            scaling_system=self.scaling_system
        )
        
        # Will be initialized when solving
        self.problem_formulation = None
        self.socp_solver = None
        
    def create_initial_guess(self, t_f: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create initial guess for trajectory.
        
        Creates a simple linear interpolation between initial and final states.
        
        Args:
            t_f: Flight time (seconds)
            
        Returns:
            Tuple of (positions, velocities, masses) arrays
        """
        # Update discretization parameters
        N = int(np.ceil(t_f / self.discretization_dt))
        self.discretization_params.N = N
        self.discretization_params.t_f = t_f
        
        # Time vector
        t = np.linspace(0, t_f, N)
        
        # Linear interpolation for position
        r0 = self.experiment_config.initial_state[:3]
        rf = self.experiment_config.final_state[:3]
        positions = np.zeros((N, 3))
        for i in range(N):
            alpha = t[i] / t_f
            positions[i] = r0 * (1 - alpha) + rf * alpha
        
        # Linear interpolation for velocity
        v0 = self.experiment_config.initial_state[3:6]
        vf = self.experiment_config.final_state[3:6]
        velocities = np.zeros((N, 3))
        for i in range(N):
            alpha = t[i] / t_f
            velocities[i] = v0 * (1 - alpha) + vf * alpha
        
        # Mass: linear decrease from wet to dry mass
        m_wet = self.experiment_config.vehicle.m_wet
        m_dry = self.experiment_config.vehicle.m_dry
        masses = np.zeros(N)
        for i in range(N):
            alpha = t[i] / t_f
            # Use quadratic profile for better initial guess
            masses[i] = m_wet - (m_wet - m_dry) * alpha**2
        
        return positions, velocities, masses
    
    def compute_position_error(
        self,
        positions1: np.ndarray,
        positions2: np.ndarray
    ) -> float:
        """
        Compute maximum position difference between two trajectories.
        
        Args:
            positions1: First trajectory positions (N×3)
            positions2: Second trajectory positions (N×3)
            
        Returns:
            Maximum Euclidean distance between corresponding points
        """
        if positions1.shape != positions2.shape:
            raise ValueError(f"Shape mismatch: {positions1.shape} vs {positions2.shape}")
        
        differences = positions1 - positions2
        distances = np.linalg.norm(differences, axis=1)
        return np.max(distances)
    
    def linearize_gravity_along_trajectory(
        self,
        positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize gravity field along a trajectory.
        
        For each position r_i, compute:
            g(r) ≈ A_i * r + c_i
            
        where A_i is the dominant term matrix and c_i is the residual.
        
        Args:
            positions: Trajectory positions (N×3)
            
        Returns:
            Tuple of (A_matrices, c_vectors) where:
                A_matrices: (N, 3, 3) array of linearization matrices
                c_vectors: (N, 3) array of residual vectors
        """
        N = len(positions)
        A_matrices = np.zeros((N, 3, 3))
        c_vectors = np.zeros((N, 3))
        
        for i in range(N):
            r = positions[i]
            # Get linearization from gravity calculator
            A, c = self.gravity_calculator.linearize(r)
            A_matrices[i] = A
            c_vectors[i] = c
            
            # Validate linearization
            g_exact = self.gravity_calculator.acceleration(r)
            g_linear = A @ r + c
            linearization_error = np.linalg.norm(g_exact - g_linear) / np.linalg.norm(g_exact)
            
            if linearization_error > self.params.max_linearization_error:
                logger.warning(
                    f"Large linearization error at point {i}: {linearization_error:.2e} > "
                    f"{self.params.max_linearization_error:.2e}"
                )
        
        return A_matrices, c_vectors
    
    def create_linearized_problem_formulation(
        self,
        positions_ref: np.ndarray,
        t_f: float
    ) -> ProblemFormulation:
        """
        Create problem formulation with linearized gravity.
        
        Args:
            positions_ref: Reference trajectory for linearization (N×3)
            t_f: Flight time (seconds)
            
        Returns:
            ProblemFormulation with linearized gravity
        """
        # Linearize gravity along reference trajectory
        A_matrices, c_vectors = self.linearize_gravity_along_trajectory(positions_ref)
        
        # Create problem formulation parameters
        params = ProblemFormulationParameters(
            asteroid=self.experiment_config.asteroid,
            vehicle=self.experiment_config.vehicle,
            landing_site=self.experiment_config.landing_site,
            discretization_params=self.discretization_params,
            scaling_system=self.scaling_system,
            initial_state=self.experiment_config.initial_state,
            final_state=self.experiment_config.final_state,
            gravity_calculator=self.gravity_calculator,
            state_equations=None,  # Will be created internally
            constraint_params=self.constraint_params,
            linearization_reference=positions_ref,
            gravity_A_matrix=A_matrices,
            gravity_c_vector=c_vectors
        )
        
        return ProblemFormulation(params)
    
    def solve_iteration(
        self,
        positions_ref: np.ndarray,
        t_f: float
    ) -> SOCPSolution:
        """
        Solve one iteration of successive solution.
        
        Args:
            positions_ref: Reference trajectory for linearization
            t_f: Flight time (seconds)
            
        Returns:
            SOCP solution for this iteration
        """
        # Create linearized problem formulation
        self.problem_formulation = self.create_linearized_problem_formulation(positions_ref, t_f)
        
        # Create SOCP solver
        self.socp_solver = SOCPSolver(
            problem_formulation=self.problem_formulation,
            solver=self.solver,
            solver_settings=self.solver_settings,
            use_scaling=self.use_scaling,
            verbose=False  # Keep quiet during iterations
        )
        
        # Solve SOCP
        solution = self.socp_solver.solve()
        
        return solution
    
    def solve(
        self,
        t_f: float,
        initial_guess: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    ) -> SuccessiveSolutionResult:
        """
        Solve the successive solution problem for given flight time.
        
        Args:
            t_f: Flight time (seconds)
            initial_guess: Optional initial guess (positions, velocities, masses)
            
        Returns:
            SuccessiveSolutionResult with convergence information
        """
        start_time = time.time()
        
        # Update discretization for this flight time
        N = int(np.ceil(t_f / self.discretization_dt))
        self.discretization_params.N = N
        self.discretization_params.t_f = t_f
        
        # Create or use provided initial guess
        if initial_guess is None:
            positions_ref, velocities_ref, masses_ref = self.create_initial_guess(t_f)
        else:
            positions_ref, velocities_ref, masses_ref = initial_guess
            
            # Ensure correct shape
            if len(positions_ref) != N:
                raise ValueError(
                    f"Initial guess has {len(positions_ref)} points, "
                    f"but discretization requires {N} points"
                )
        
        # Initialize result tracking
        intermediate_solutions = []
        iteration_errors = []
        linearization_errors = []
        
        # Successive solution loop
        converged = False
        iteration = 0
        
        if self.params.verbose:
            logger.info(f"Starting successive solution for t_f={t_f:.1f}s, N={N}")
        
        while iteration < self.params.max_iterations:
            iteration_start = time.time()
            
            # Solve SOCP with current linearization
            solution = self.solve_iteration(positions_ref, t_f)
            
            # Extract new trajectory
            positions_new = solution.positions
            velocities_new = solution.velocities
            masses_new = solution.masses
            
            # Compute position error
            position_error = self.compute_position_error(positions_new, positions_ref)
            iteration_errors.append(position_error)
            
            # Store intermediate solution if requested
            if self.params.store_intermediate_solutions:
                intermediate_solutions.append(solution)
            
            # Compute linearization error for this iteration
            if iteration > 0:
                # Compare exact vs linearized gravity at reference points
                total_lin_error = 0.0
                for i in range(N):
                    r = positions_ref[i]
                    g_exact = self.gravity_calculator.acceleration(r)
                    A, c = self.gravity_calculator.linearize(r)
                    g_linear = A @ r + c
                    error = np.linalg.norm(g_exact - g_linear) / np.linalg.norm(g_exact)
                    total_lin_error += error
                linearization_errors.append(total_lin_error / N)
            
            iteration_time = time.time() - iteration_start
            
            if self.params.verbose:
                logger.info(
                    f"Iteration {iteration+1}: error={position_error:.4f}m, "
                    f"time={iteration_time:.2f}s"
                )
            
            # Check convergence
            if position_error < self.params.position_tolerance:
                converged = True
                if self.params.verbose:
                    logger.info(f"Converged after {iteration+1} iterations")
                break
            
            # Update reference trajectory for next iteration
            positions_ref = positions_new.copy()
            iteration += 1
        
        if not converged and self.params.verbose:
            logger.warning(
                f"Failed to converge after {self.params.max_iterations} iterations. "
                f"Final error: {iteration_errors[-1]:.4f}m"
            )
        
        # Create result object
        computation_time = time.time() - start_time
        result = SuccessiveSolutionResult(
            converged=converged,
            iteration_count=iteration + 1,
            final_position_error=iteration_errors[-1] if iteration_errors else 0.0,
            solution=solution,
            intermediate_solutions=intermediate_solutions,
            iteration_errors=iteration_errors,
            computation_time=computation_time,
            linearization_errors=linearization_errors
        )
        
        if self.params.verbose:
            logger.info(result.summary())
        
        return result
    
    def solve_with_adaptive_linearization(
        self,
        t_f: float,
        initial_guess: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    ) -> SuccessiveSolutionResult:
        """
        Solve with adaptive linearization strategy.
        
        Uses tighter linearization tolerance when close to convergence.
        
        Args:
            t_f: Flight time (seconds)
            initial_guess: Optional initial guess
            
        Returns:
            SuccessiveSolutionResult
        """
        if not self.params.use_adaptive_linearization:
            return self.solve(t_f, initial_guess)
        
        # Save original parameters
        original_tolerance = self.params.position_tolerance
        original_max_lin_error = self.params.max_linearization_error
        
        try:
            # First pass: coarse tolerance
            self.params.position_tolerance = original_tolerance * 5
            self.params.max_linearization_error = original_max_lin_error * 10
            
            result_coarse = self.solve(t_f, initial_guess)
            
            if not result_coarse.converged:
                # If coarse didn't converge, return as-is
                return result_coarse
            
            # Second pass: fine tolerance using coarse solution as initial guess
            self.params.position_tolerance = original_tolerance
            self.params.max_linearization_error = original_max_lin_error
            
            # Use coarse solution as initial guess
            positions_fine = result_coarse.solution.positions
            velocities_fine = result_coarse.solution.velocities
            masses_fine = result_coarse.solution.masses
            
            result_fine = self.solve(
                t_f,
                initial_guess=(positions_fine, velocities_fine, masses_fine)
            )
            
            # Combine results
            result_fine.iteration_count += result_coarse.iteration_count
            result_fine.computation_time += result_coarse.computation_time
            result_fine.intermediate_solutions = (
                result_coarse.intermediate_solutions + result_fine.intermediate_solutions
            )
            result_fine.iteration_errors = (
                result_coarse.iteration_errors + result_fine.iteration_errors
            )
            
            return result_fine
            
        finally:
            # Restore original parameters
            self.params.position_tolerance = original_tolerance
            self.params.max_linearization_error = original_max_lin_error


def solve_successive_solution(
    experiment_config: ExperimentConfig,
    t_f: float,
    discretization_dt: float = 2.0,
    solver: str = DEFAULT_SOLVER,
    use_scaling: bool = True,
    params: Optional[SuccessiveSolutionParameters] = None
) -> SuccessiveSolutionResult:
    """
    Convenience function to solve successive solution problem.
    
    Args:
        experiment_config: Experiment configuration
        t_f: Flight time (seconds)
        discretization_dt: Time step for discretization
        solver: SOCP solver name
        use_scaling: Whether to use numerical scaling
        params: Successive solution parameters
        
    Returns:
        SuccessiveSolutionResult
    """
    solver = SuccessiveSolver(
        experiment_config=experiment_config,
        discretization_dt=discretization_dt,
        solver=solver,
        use_scaling=use_scaling,
        params=params
    )
    
    return solver.solve(t_f)


def test_successive_solver() -> bool:
    """Test the successive solver implementation."""
    import numpy as np
    from ..config import (
        get_asteroid_by_name, get_vehicle_by_thrust, get_landing_site_by_name,
        create_triaxial_experiment
    )
    
    print("Testing successive solver...")
    
    try:
        # Create a simple test case with asteroid A1
        asteroid = get_asteroid_by_name("A1")
        vehicle = get_vehicle_by_thrust("full")
        
        # Create experiment configuration
        experiment_config = create_triaxial_experiment(asteroid, vehicle)
        
        # Create successive solver
        solver = SuccessiveSolver(
            experiment_config=experiment_config,
            discretization_dt=10.0,  # Use coarse dt for faster testing
            solver="ECOS",  # Use ECOS for testing (no MOSEK required)
            use_scaling=True,
            params=SuccessiveSolutionParameters(
                max_iterations=3,
                position_tolerance=5.0,  # Relaxed tolerance for testing
                verbose=False
            )
        )
        
        # Test with a short flight time
        t_f = 100.0
        
        # Test initial guess creation
        positions, velocities, masses = solver.create_initial_guess(t_f)
        N = len(positions)
        assert positions.shape == (N, 3), "Positions shape incorrect"
        assert velocities.shape == (N, 3), "Velocities shape incorrect"
        assert masses.shape == (N,), "Masses shape incorrect"
        print("✓ Initial guess creation passed")
        
        # Test position error computation
        error = solver.compute_position_error(positions, positions * 1.1)
        assert error > 0, "Position error should be positive"
        print("✓ Position error computation passed")
        
        # Test linearization
        A_matrices, c_vectors = solver.linearize_gravity_along_trajectory(positions[:5])  # Test with first 5 points
        assert A_matrices.shape == (5, 3, 3), "A_matrices shape incorrect"
        assert c_vectors.shape == (5, 3), "c_vectors shape incorrect"
        print("✓ Gravity linearization passed")
        
        # Note: Full solve test is skipped in unit tests because it requires
        # a proper SOCP solver and can be time-consuming
        
        print("All tests passed!")
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run tests if module is executed directly
    test_successive_solver()