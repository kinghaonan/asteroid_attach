"""
Flight Time Optimizer using Brent's Method

Implements Algorithm 3 from the paper: Brent's method wrapper for finding the optimal
flight time that minimizes propellant usage. Uses a coarse-to-fine strategy with
different discretization time steps for the Brent search (10s) and final solution (2s).

Reference: Section V of the paper
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional, Callable, Union
from dataclasses import dataclass, field
import time

from scipy.optimize import brent

from ..config import (
    ExperimentConfig, VehicleParameters, AsteroidParameters, LandingSite,
    DEFAULT_SOLVER, SOLVER_SETTINGS, VALIDATION_TOLERANCES
)
from ..dynamics.state_equations import StateVector, ControlVector
from ..dynamics.discretization import DiscretizationParameters
from ..dynamics.scaling import ScalingSystem
from .successive_solver import SuccessiveSolver, SuccessiveSolutionResult, SuccessiveSolutionParameters
from .socp_solver import SOCPSolution

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class FlightTimeOptimizerParameters:
    """Parameters for the flight time optimization algorithm."""
    
    # Brent's method parameters
    brent_tolerance: float = 1.0  # Tolerance for Brent's method (seconds)
    brent_max_iter: int = 50      # Maximum iterations for Brent's method
    
    # Search range parameters
    t_min_factor: float = 0.3     # Minimum flight time as fraction of initial guess
    t_max_factor: float = 3.0     # Maximum flight time as fraction of initial guess
    
    # Discretization strategy
    coarse_dt: float = 10.0       # Time step for coarse search (seconds)
    fine_dt: float = 2.0          # Time step for final solution (seconds)
    
    # Successive solver parameters
    successive_params: SuccessiveSolutionParameters = field(
        default_factory=lambda: SuccessiveSolutionParameters()
    )
    
    # Solver settings
    solver: str = DEFAULT_SOLVER
    solver_settings: Dict = field(default_factory=dict)
    use_scaling: bool = True
    verbose: bool = False
    
    def __post_init__(self):
        """Validate parameters."""
        if self.brent_tolerance <= 0:
            raise ValueError("brent_tolerance must be positive")
        if self.brent_max_iter <= 0:
            raise ValueError("brent_max_iter must be positive")
        if self.t_min_factor <= 0 or self.t_max_factor <= self.t_min_factor:
            raise ValueError("Invalid time factor range")
        if self.coarse_dt <= 0 or self.fine_dt <= 0:
            raise ValueError("Time steps must be positive")
        
        # Update solver settings with defaults if not provided
        if not self.solver_settings:
            self.solver_settings = SOLVER_SETTINGS.get(self.solver, {}).copy()


@dataclass
class FlightTimeOptimizationResult:
    """Results of flight time optimization."""
    
    # Optimal solution
    optimal_flight_time: float                    # t_f* (seconds)
    optimal_solution: SOCPSolution                # Solution at optimal flight time
    optimal_propellant: float                     # Propellant used at optimum (kg)
    
    # Search information
    brent_iterations: int = 0                     # Number of Brent iterations
    function_evaluations: int = 0                 # Number of propellant function evaluations
    search_range: Tuple[float, float] = (0.0, 0.0)  # [t_min, t_max] search range
    
    # Performance metrics
    total_computation_time: float = 0.0           # Total wall time (seconds)
    coarse_search_time: float = 0.0               # Time spent in coarse evaluations
    fine_solution_time: float = 0.0               # Time spent in final fine solution
    
    # Convergence information
    converged: bool = False                       # Whether optimization converged
    convergence_message: str = ""                 # Convergence message
    
    # Additional data for analysis
    evaluation_history: Dict[float, float] = field(default_factory=dict)  # t_f -> propellant
    coarse_solutions: Dict[float, SOCPSolution] = field(default_factory=dict)  # Coarse solutions
    
    def __post_init__(self):
        """Validate result."""
        if self.optimal_flight_time <= 0:
            raise ValueError("Optimal flight time must be positive")
        if self.optimal_propellant < 0:
            raise ValueError("Propellant usage cannot be negative")


class FlightTimeOptimizer:
    """
    Flight time optimizer using Brent's method.
    
    Implements Algorithm 3 from the paper:
    1. Define function f(t_f) = propellant_used(t_f) via inner SOCP solver
    2. Use Brent's method on f(t_f) over [t_min, t_max]
    3. Coarse search: Δt = 10s for initial Brent iterations
    4. Fine solution: Δt = 2s at optimal t_f*
    """
    
    def __init__(
        self,
        experiment_config: ExperimentConfig,
        params: Optional[FlightTimeOptimizerParameters] = None
    ):
        """
        Initialize the flight time optimizer.
        
        Args:
            experiment_config: Configuration for the landing experiment
            params: Optimization parameters (uses defaults if None)
        """
        self.experiment_config = experiment_config
        self.params = params or FlightTimeOptimizerParameters()
        
        # Initialize state
        self._evaluation_count = 0
        self._evaluation_history = {}
        self._coarse_solutions = {}
        
        # Create scaling system for the problem
        self.scaling_system = ScalingSystem(
            asteroid=experiment_config.asteroid,
            vehicle=experiment_config.vehicle
        )
        
        logger.info(f"Initialized flight time optimizer for {experiment_config.asteroid.name}")
    
    def evaluate_propellant_coarse(self, t_f: float) -> float:
        """
        Evaluate propellant usage for a given flight time using coarse discretization.
        
        This is the objective function for Brent's method.
        
        Args:
            t_f: Flight time to evaluate (seconds)
            
        Returns:
            Propellant usage (kg) for the given flight time
        """
        start_time = time.time()
        
        # Check cache first
        if t_f in self._evaluation_history:
            return self._evaluation_history[t_f]
        
        # Update evaluation count
        self._evaluation_count += 1
        
        # Create a modified experiment config with the specified flight time
        # (We'll use the coarse discretization for Brent iterations)
        modified_config = ExperimentConfig(
            asteroid=self.experiment_config.asteroid,
            vehicle=self.experiment_config.vehicle,
            landing_site=self.experiment_config.landing_site,
            initial_state=self.experiment_config.initial_state,
            final_state=self.experiment_config.final_state,
            flight_time_range=(t_f, t_f),  # Single value
            discretization_dt=self.params.coarse_dt,
            convergence_tolerance=self.experiment_config.convergence_tolerance,
            max_iterations=self.experiment_config.max_iterations
        )
        
        try:
            # Solve the fixed-time problem with coarse discretization
            from .successive_solver import solve_successive_solution
            
            result = solve_successive_solution(
                experiment_config=modified_config,
                t_f=t_f,
                discretization_dt=self.params.coarse_dt,
                solver=self.params.solver,
                use_scaling=self.params.use_scaling,
                params=self.params.successive_params
            )
            
            if result.converged and result.solution is not None:
                propellant = result.solution.propellant_used
                
                # Store in cache
                self._evaluation_history[t_f] = propellant
                self._coarse_solutions[t_f] = result.solution
                
                elapsed = time.time() - start_time
                logger.debug(f"Coarse evaluation t_f={t_f:.1f}s -> propellant={propellant:.3f}kg "
                           f"(iter={result.iterations}, time={elapsed:.1f}s)")
                
                return propellant
            else:
                # If solution didn't converge, return a large penalty
                logger.warning(f"Coarse solution failed to converge for t_f={t_f:.1f}s")
                penalty = 1e6 * (1.0 + abs(t_f - 300.0) / 100.0)  # Penalty based on deviation from typical t_f
                self._evaluation_history[t_f] = penalty
                return penalty
                
        except Exception as e:
            logger.error(f"Error in coarse evaluation for t_f={t_f:.1f}s: {e}")
            # Return a large penalty for failed evaluations
            penalty = 1e6 * (1.0 + abs(t_f - 300.0) / 100.0)
            self._evaluation_history[t_f] = penalty
            return penalty
    
    def compute_fine_solution(self, t_f: float) -> SOCPSolution:
        """
        Compute a high-accuracy solution using fine discretization.
        
        Args:
            t_f: Flight time for fine solution (seconds)
            
        Returns:
            High-accuracy SOCP solution with fine discretization
        """
        start_time = time.time()
        
        # Create a modified experiment config with fine discretization
        modified_config = ExperimentConfig(
            asteroid=self.experiment_config.asteroid,
            vehicle=self.experiment_config.vehicle,
            landing_site=self.experiment_config.landing_site,
            initial_state=self.experiment_config.initial_state,
            final_state=self.experiment_config.final_state,
            flight_time_range=(t_f, t_f),  # Single value
            discretization_dt=self.params.fine_dt,
            convergence_tolerance=self.experiment_config.convergence_tolerance,
            max_iterations=self.experiment_config.max_iterations
        )
        
        from .successive_solver import solve_successive_solution
        
        result = solve_successive_solution(
            experiment_config=modified_config,
            t_f=t_f,
            discretization_dt=self.params.fine_dt,
            solver=self.params.solver,
            use_scaling=self.params.use_scaling,
            params=self.params.successive_params
        )
        
        if not result.converged or result.solution is None:
            raise RuntimeError(f"Fine solution failed to converge for t_f={t_f:.1f}s")
        
        elapsed = time.time() - start_time
        logger.info(f"Fine solution t_f={t_f:.1f}s computed in {elapsed:.1f}s "
                   f"(propellant={result.solution.propellant_used:.3f}kg)")
        
        return result.solution
    
    def determine_search_range(self, initial_guess: Optional[float] = None) -> Tuple[float, float]:
        """
        Determine the search range for flight time optimization.
        
        Args:
            initial_guess: Initial guess for flight time (if None, uses config range midpoint)
            
        Returns:
            Tuple of (t_min, t_max) defining the search range
        """
        # Use initial guess or midpoint of config range
        if initial_guess is None:
            t_low, t_high = self.experiment_config.flight_time_range
            initial_guess = (t_low + t_high) / 2.0
        
        # Apply scaling factors to get search range
        t_min = initial_guess * self.params.t_min_factor
        t_max = initial_guess * self.params.t_max_factor
        
        # Ensure range is within config bounds
        config_min, config_max = self.experiment_config.flight_time_range
        t_min = max(t_min, config_min)
        t_max = min(t_max, config_max)
        
        # Ensure minimum range
        if t_max - t_min < 10.0:
            t_max = t_min + 100.0  # Extend range if too small
        
        logger.info(f"Search range: [{t_min:.1f}, {t_max:.1f}] seconds "
                   f"(initial guess: {initial_guess:.1f}s)")
        
        return (t_min, t_max)
    
    def optimize(self, initial_guess: Optional[float] = None) -> FlightTimeOptimizationResult:
        """
        Find optimal flight time using Brent's method.
        
        Args:
            initial_guess: Initial guess for flight time (optional)
            
        Returns:
            FlightTimeOptimizationResult with optimal solution
        """
        total_start_time = time.time()
        
        # Reset evaluation counters
        self._evaluation_count = 0
        self._evaluation_history = {}
        self._coarse_solutions = {}
        
        # Determine search range
        t_min, t_max = self.determine_search_range(initial_guess)
        
        logger.info(f"Starting flight time optimization using Brent's method "
                   f"over [{t_min:.1f}, {t_max:.1f}] seconds")
        
        # Define the objective function for Brent
        def objective(t_f: float) -> float:
            """Objective function: propellant usage as function of flight time."""
            return self.evaluate_propellant_coarse(t_f)
        
        # Run Brent's method
        brent_start_time = time.time()
        
        try:
            # Use SciPy's brent method for minimization
            result = brent(
                objective,
                brack=(t_min, t_max),
                tol=self.params.brent_tolerance,
                maxiter=self.params.brent_max_iter,
                full_output=True
            )
            
            if len(result) == 4:  # brent returns (xmin, fval, iter, funcalls)
                t_opt, f_opt, brent_iter, func_calls = result
                success = True
                message = "Brent's method converged"
            else:
                # Fallback if return format differs
                t_opt = result[0]
                f_opt = objective(t_opt)
                brent_iter = self.params.brent_max_iter
                func_calls = self._evaluation_count
                success = True
                message = "Brent's method completed"
                
        except Exception as e:
            logger.error(f"Brent's method failed: {e}")
            # Fallback: use golden section search or simple sampling
            t_opt, f_opt, brent_iter, func_calls = self._fallback_optimization(
                objective, t_min, t_max
            )
            success = t_opt is not None
            message = f"Brent failed, used fallback: {e}" if success else "Optimization failed"
        
        brent_time = time.time() - brent_start_time
        
        if not success or t_opt is None:
            logger.error("Flight time optimization failed")
            return FlightTimeOptimizationResult(
                optimal_flight_time=0.0,
                optimal_solution=None,
                optimal_propellant=0.0,
                brent_iterations=brent_iter,
                function_evaluations=func_calls,
                search_range=(t_min, t_max),
                total_computation_time=time.time() - total_start_time,
                coarse_search_time=brent_time,
                converged=False,
                convergence_message=message
            )
        
        logger.info(f"Brent's method found optimum at t_f={t_opt:.1f}s "
                   f"with propellant={f_opt:.3f}kg "
                   f"(iterations={brent_iter}, evaluations={func_calls})")
        
        # Compute fine solution at the optimum
        fine_start_time = time.time()
        try:
            fine_solution = self.compute_fine_solution(t_opt)
            fine_time = time.time() - fine_start_time
            
            # Create final result
            total_time = time.time() - total_start_time
            
            result = FlightTimeOptimizationResult(
                optimal_flight_time=t_opt,
                optimal_solution=fine_solution,
                optimal_propellant=fine_solution.propellant_used,
                brent_iterations=brent_iter,
                function_evaluations=func_calls,
                search_range=(t_min, t_max),
                total_computation_time=total_time,
                coarse_search_time=brent_time,
                fine_solution_time=fine_time,
                converged=True,
                convergence_message=message,
                evaluation_history=self._evaluation_history.copy(),
                coarse_solutions=self._coarse_solutions.copy()
            )
            
            logger.info(f"Flight time optimization completed in {total_time:.1f}s: "
                       f"t_f*={t_opt:.1f}s, propellant={fine_solution.propellant_used:.3f}kg")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to compute fine solution at optimum: {e}")
            return FlightTimeOptimizationResult(
                optimal_flight_time=t_opt,
                optimal_solution=None,
                optimal_propellant=f_opt,
                brent_iterations=brent_iter,
                function_evaluations=func_calls,
                search_range=(t_min, t_max),
                total_computation_time=time.time() - total_start_time,
                coarse_search_time=brent_time,
                converged=False,
                convergence_message=f"Found optimum but fine solution failed: {e}"
            )
    
    def _fallback_optimization(
        self, 
        objective: Callable[[float], float], 
        t_min: float, 
        t_max: float
    ) -> Tuple[Optional[float], float, int, int]:
        """
        Fallback optimization method if Brent's method fails.
        
        Uses golden section search with sampling.
        
        Args:
            objective: Objective function
            t_min: Minimum flight time
            t_max: Maximum flight time
            
        Returns:
            Tuple of (optimal_t, optimal_value, iterations, function_evaluations)
        """
        logger.warning("Using fallback optimization (golden section search)")
        
        # Sample the objective at several points
        n_samples = 7
        t_samples = np.linspace(t_min, t_max, n_samples)
        f_samples = []
        
        for t in t_samples:
            f_samples.append(objective(t))
        
        # Find minimum among samples
        min_idx = np.argmin(f_samples)
        t_best = t_samples[min_idx]
        f_best = f_samples[min_idx]
        
        # Refine with golden section search around the best point
        if min_idx > 0 and min_idx < n_samples - 1:
            # We have neighbors to define a bracket
            t_left = t_samples[min_idx - 1]
            t_right = t_samples[min_idx + 1]
            
            # Perform a few golden section iterations
            golden_ratio = (np.sqrt(5) - 1) / 2  # ≈ 0.618
            
            for i in range(5):  # 5 iterations of golden section
                # Define two interior points
                d = golden_ratio * (t_right - t_left)
                t_a = t_right - d
                t_b = t_left + d
                
                # Evaluate at interior points
                f_a = objective(t_a)
                f_b = objective(t_b)
                
                # Update bracket
                if f_a < f_b:
                    t_right = t_b
                    f_best = f_a if f_a < f_best else f_best
                    t_best = t_a if f_a < f_best else t_best
                else:
                    t_left = t_a
                    f_best = f_b if f_b < f_best else f_best
                    t_best = t_b if f_b < f_best else t_best
        
        return t_best, f_best, 5, self._evaluation_count
    
    def analyze_unimodality(self, n_points: int = 20) -> Dict[str, np.ndarray]:
        """
        Analyze the unimodality of the propellant vs flight time function.
        
        Args:
            n_points: Number of points to sample
            
        Returns:
            Dictionary with 'flight_times' and 'propellant_values' arrays
        """
        t_min, t_max = self.determine_search_range()
        flight_times = np.linspace(t_min, t_max, n_points)
        propellant_values = []
        
        logger.info(f"Sampling propellant function at {n_points} points "
                   f"over [{t_min:.1f}, {t_max:.1f}]")
        
        for t in flight_times:
            propellant = self.evaluate_propellant_coarse(t)
            propellant_values.append(propellant)
        
        return {
            'flight_times': flight_times,
            'propellant_values': np.array(propellant_values)
        }


def optimize_flight_time(
    experiment_config: ExperimentConfig,
    params: Optional[FlightTimeOptimizerParameters] = None
) -> FlightTimeOptimizationResult:
    """
    Convenience function to optimize flight time.
    
    Args:
        experiment_config: Experiment configuration
        params: Optimizer parameters (optional)
        
    Returns:
        FlightTimeOptimizationResult
    """
    optimizer = FlightTimeOptimizer(experiment_config, params)
    return optimizer.optimize()


def test_flight_time_optimizer() -> bool:
    """Test the flight time optimizer module."""
    import numpy as np
    from ..config import (
        ASTEROID_A1, FULL_THRUST_VEHICLE, 
        create_triaxial_experiment, ExperimentConfig
    )
    
    print("Testing flight time optimizer...")
    
    try:
        # Create a simple test experiment
        asteroid = ASTEROID_A1
        vehicle = FULL_THRUST_VEHICLE
        
        # Create experiment config
        experiment_config = create_triaxial_experiment(
            asteroid=asteroid,
            vehicle=vehicle,
            landing_site_position=np.array([0.0, 0.0, asteroid.semi_axes[2] * 0.9])
        )
        
        # Modify flight time range for testing
        experiment_config = ExperimentConfig(
            asteroid=experiment_config.asteroid,
            vehicle=experiment_config.vehicle,
            landing_site=experiment_config.landing_site,
            initial_state=experiment_config.initial_state,
            final_state=experiment_config.final_state,
            flight_time_range=(100.0, 600.0),  # Reasonable range for testing
            discretization_dt=2.0,
            convergence_tolerance=0.5,
            max_iterations=10
        )
        
        # Create optimizer with test parameters
        params = FlightTimeOptimizerParameters(
            brent_tolerance=5.0,  # Larger tolerance for faster testing
            brent_max_iter=10,
            coarse_dt=20.0,  # Larger dt for faster testing
            fine_dt=10.0,
            successive_params=SuccessiveSolutionParameters(
                max_iterations=3,  # Fewer iterations for testing
                position_tolerance=5.0  # Larger tolerance
            ),
            verbose=False
        )
        
        optimizer = FlightTimeOptimizer(experiment_config, params)
        
        # Test search range determination
        t_min, t_max = optimizer.determine_search_range(initial_guess=300.0)
        print(f"  Search range: [{t_min:.1f}, {t_max:.1f}]")
        assert t_min < t_max, "Invalid search range"
        
        # Test unimodality analysis (just check it runs)
        analysis = optimizer.analyze_unimodality(n_points=5)
        assert 'flight_times' in analysis
        assert 'propellant_values' in analysis
        assert len(analysis['flight_times']) == 5
        assert len(analysis['propellant_values']) == 5
        print(f"  Unimodality analysis: sampled {len(analysis['flight_times'])} points")
        
        # Test single evaluation
        propellant = optimizer.evaluate_propellant_coarse(300.0)
        print(f"  Propellant at t_f=300s: {propellant:.3f}kg")
        assert isinstance(propellant, float), "Propellant should be float"
        
        # Note: We don't run full optimization in unit test to avoid long computation
        print("  Flight time optimizer tests passed (skipping full optimization for speed)")
        
        return True
        
    except Exception as e:
        print(f"  Flight time optimizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run tests if module is executed directly
    test_flight_time_optimizer()