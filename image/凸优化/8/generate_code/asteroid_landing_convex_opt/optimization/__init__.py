"""
Optimization module for asteroid landing convex optimization.

This module provides the core optimization algorithms for solving the propellant-optimal
powered descent trajectory problem on irregularly shaped asteroids using convex
optimization techniques.

The module implements:
1. Problem formulation and lossless convexification (P1 → P3)
2. Fixed-time SOCP solver with CVXPY
3. Successive solution algorithm with gravity linearization
4. Flight time optimization using Brent's method
5. Trajectory constraints (glide slope, vertical motion, etc.)
"""

from .constraints import (
    ConstraintParameters,
    TrajectoryConstraints,
    create_constraint_parameters,
    test_constraints,
)

from .problem_formulation import (
    ProblemFormulationParameters,
    ProblemFormulation,
    create_problem_formulation,
    test_problem_formulation,
)

from .socp_solver import (
    SOCPSolution,
    SOCPSolver,
    solve_fixed_time_socp,
    test_socp_solver,
)

from .successive_solver import (
    SuccessiveSolutionParameters,
    SuccessiveSolutionResult,
    SuccessiveSolver,
    solve_successive_solution,
    test_successive_solver,
)

from .flight_time_optimizer import (
    FlightTimeOptimizerParameters,
    FlightTimeOptimizationResult,
    FlightTimeOptimizer,
    optimize_flight_time,
    test_flight_time_optimizer,
)

__all__ = [
    # Constraints
    "ConstraintParameters",
    "TrajectoryConstraints",
    "create_constraint_parameters",
    "test_constraints",
    
    # Problem formulation
    "ProblemFormulationParameters",
    "ProblemFormulation",
    "create_problem_formulation",
    "test_problem_formulation",
    
    # SOCP solver
    "SOCPSolution",
    "SOCPSolver",
    "solve_fixed_time_socp",
    "test_socp_solver",
    
    # Successive solver
    "SuccessiveSolutionParameters",
    "SuccessiveSolutionResult",
    "SuccessiveSolver",
    "solve_successive_solution",
    "test_successive_solver",
    
    # Flight time optimizer
    "FlightTimeOptimizerParameters",
    "FlightTimeOptimizationResult",
    "FlightTimeOptimizer",
    "optimize_flight_time",
    "test_flight_time_optimizer",
]

# Convenience functions for common operations
def create_default_optimization_components(experiment_config, discretization_dt=2.0):
    """
    Create default optimization components for a given experiment configuration.
    
    Args:
        experiment_config: ExperimentConfig instance
        discretization_dt: Time step for discretization (seconds)
        
    Returns:
        Tuple of (problem_formulation, scaling_system, constraint_params)
    """
    from .problem_formulation import create_problem_formulation
    from ..dynamics.scaling import create_default_scaling_system
    
    # Create scaling system
    scaling_system = create_default_scaling_system(
        asteroid_name=experiment_config.asteroid.name,
        vehicle_thrust="full" if experiment_config.vehicle.T_max == 80.0 else "quarter"
    )
    
    # Create problem formulation
    problem_formulation = create_problem_formulation(
        experiment_config=experiment_config,
        discretization_dt=discretization_dt,
        use_scaling=True
    )
    
    # Create constraint parameters
    constraint_params = create_constraint_parameters(
        vehicle_params=experiment_config.vehicle,
        landing_site=experiment_config.landing_site,
        initial_state=experiment_config.initial_state,
        final_state=experiment_config.final_state,
        scaling_system=scaling_system
    )
    
    return problem_formulation, scaling_system, constraint_params

def solve_asteroid_landing_problem(experiment_config, t_f, dt=2.0, solver="ECOS", use_scaling=True):
    """
    Solve a complete asteroid landing problem for a given flight time.
    
    This is a high-level convenience function that combines successive solution
    and SOCP solving.
    
    Args:
        experiment_config: ExperimentConfig instance
        t_f: Flight time (seconds)
        dt: Discretization time step (seconds)
        solver: SOCP solver to use ("ECOS", "SCS", or "MOSEK")
        use_scaling: Whether to use numerical scaling
        
    Returns:
        Tuple of (solution, successive_result)
    """
    from .successive_solver import solve_successive_solution
    
    # Solve using successive solution algorithm
    successive_result = solve_successive_solution(
        experiment_config=experiment_config,
        t_f=t_f,
        discretization_dt=dt,
        solver=solver,
        use_scaling=use_scaling
    )
    
    return successive_result.solution, successive_result

def optimize_asteroid_landing(experiment_config, optimizer_params=None):
    """
    Optimize both trajectory and flight time for an asteroid landing problem.
    
    This is the highest-level function that performs complete optimization
    (flight time + trajectory) using Brent's method.
    
    Args:
        experiment_config: ExperimentConfig instance
        optimizer_params: FlightTimeOptimizerParameters instance (optional)
        
    Returns:
        FlightTimeOptimizationResult instance
    """
    from .flight_time_optimizer import optimize_flight_time
    
    return optimize_flight_time(
        experiment_config=experiment_config,
        params=optimizer_params
    )

# Test function for the entire optimization module
def test_optimization_module():
    """
    Run comprehensive tests for the entire optimization module.
    
    Returns:
        bool: True if all tests pass, False otherwise
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("Testing optimization module...")
    
    # Test each submodule
    test_results = []
    
    try:
        test_results.append(("constraints", test_constraints()))
    except Exception as e:
        logger.error(f"Constraints test failed: {e}")
        test_results.append(("constraints", False))
    
    try:
        test_results.append(("problem_formulation", test_problem_formulation()))
    except Exception as e:
        logger.error(f"Problem formulation test failed: {e}")
        test_results.append(("problem_formulation", False))
    
    try:
        test_results.append(("socp_solver", test_socp_solver()))
    except Exception as e:
        logger.error(f"SOCP solver test failed: {e}")
        test_results.append(("socp_solver", False))
    
    try:
        test_results.append(("successive_solver", test_successive_solver()))
    except Exception as e:
        logger.error(f"Successive solver test failed: {e}")
        test_results.append(("successive_solver", False))
    
    try:
        test_results.append(("flight_time_optimizer", test_flight_time_optimizer()))
    except Exception as e:
        logger.error(f"Flight time optimizer test failed: {e}")
        test_results.append(("flight_time_optimizer", False))
    
    # Check all results
    all_passed = all(result for _, result in test_results)
    
    if all_passed:
        logger.info("All optimization module tests passed!")
    else:
        logger.error("Some optimization module tests failed:")
        for module_name, passed in test_results:
            status = "PASSED" if passed else "FAILED"
            logger.error(f"  {module_name}: {status}")
    
    return all_passed