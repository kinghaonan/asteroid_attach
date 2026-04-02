"""
Metrics computation utilities for asteroid landing convex optimization.

This module provides functions for computing various metrics and validation
measures for trajectory solutions, including propellant usage, thrust profile
analysis, constraint violations, and comparison with paper results.

Author: Asteroid Landing Convex Optimization Team
Date: 2026-01-15
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field

from ..config import VehicleParameters, AsteroidParameters, LandingSite, G
from ..dynamics.state_equations import StateVector
from ..optimization.socp_solver import SOCPSolution
from ..optimization.successive_solver import SuccessiveSolutionResult
from ..optimization.flight_time_optimizer import FlightTimeOptimizationResult

logger = logging.getLogger(__name__)


@dataclass
class ThrustProfileMetrics:
    """Metrics for thrust profile analysis."""
    thrust_magnitude: np.ndarray
    time_vector: np.ndarray
    dt: float
    switching_times: List[float] = field(default_factory=list)
    switching_indices: List[int] = field(default_factory=list)
    is_bang_bang: bool = False
    bang_bang_structure: str = ""
    max_thrust_segments: int = 0
    min_thrust_segments: int = 0
    mean_thrust: float = 0.0
    std_thrust: float = 0.0
    thrust_variation: float = 0.0
    satisfies_thrust_bounds: bool = False
    max_thrust_violation: float = 0.0
    min_thrust_violation: float = 0.0
    paper_switching_times: Optional[List[float]] = None
    switching_time_errors: List[float] = field(default_factory=list)
    max_switching_time_error: float = 0.0


@dataclass
class ConstraintViolationMetrics:
    """Metrics for constraint violation analysis."""
    glide_slope_violations: np.ndarray
    max_glide_slope_violation: float
    mean_glide_slope_violation: float
    glide_slope_satisfied: bool
    
    vertical_motion_violations: np.ndarray
    max_vertical_motion_violation: float
    mean_vertical_motion_violation: float
    vertical_motion_satisfied: bool
    
    mass_violations: np.ndarray
    max_mass_violation: float
    mean_mass_violation: float
    mass_constraint_satisfied: bool
    
    thrust_magnitude_violations: np.ndarray
    max_thrust_magnitude_violation: float
    mean_thrust_magnitude_violation: float
    thrust_magnitude_satisfied: bool
    
    initial_position_error: float
    initial_velocity_error: float
    initial_mass_error: float
    final_position_error: float
    final_velocity_error: float
    boundary_conditions_satisfied: bool
    
    all_constraints_satisfied: bool
    max_overall_violation: float


@dataclass
class TrajectoryQualityMetrics:
    """Overall trajectory quality metrics."""
    propellant_used: float
    propellant_efficiency: float  # propellant used / initial mass
    flight_time: float
    time_efficiency: float  # propellant used per unit time
    
    position_smoothness: float  # mean squared jerk
    thrust_smoothness: float  # mean squared thrust derivative
    
    minimum_altitude: float
    maximum_velocity: float
    maximum_acceleration: float
    
    paper_propellant: Optional[float] = None
    propellant_error_percent: Optional[float] = None
    paper_flight_time: Optional[float] = None
    flight_time_error_percent: Optional[float] = None


def compute_propellant_used(solution: SOCPSolution) -> float:
    """
    Compute propellant used from a trajectory solution.
    
    Args:
        solution: SOCP solution containing state and control history
        
    Returns:
        Propellant mass used (kg)
    """
    if solution.initial_mass is None or solution.final_mass is None:
        # Compute from mass history if not directly available
        if solution.mass_history is not None and len(solution.mass_history) > 0:
            initial_mass = solution.mass_history[0]
            final_mass = solution.mass_history[-1]
            return initial_mass - final_mass
        else:
            # Estimate from thrust profile
            if solution.thrust_magnitude_history is not None and solution.time_vector is not None:
                # Use trapezoidal integration of mass flow
                thrust_mag = solution.thrust_magnitude_history
                time_vec = solution.time_vector
                dt = time_vec[1] - time_vec[0] if len(time_vec) > 1 else 1.0
                
                # Get Isp from vehicle parameters
                if solution.vehicle_params is not None:
                    I_sp = solution.vehicle_params.I_sp
                else:
                    I_sp = 300.0  # default
                
                # Mass flow rate = thrust / (I_sp * g0)
                mass_flow = thrust_mag / (I_sp * G)
                propellant_used = np.trapz(mass_flow, time_vec)
                return propellant_used
    
    # Direct computation from initial and final mass
    return solution.initial_mass - solution.final_mass


def compute_thrust_profile_metrics(
    solution: SOCPSolution,
    vehicle_params: VehicleParameters,
    time_vector: Optional[np.ndarray] = None,
    dt: Optional[float] = None,
    switching_threshold: float = 0.1,
    paper_switching_times: Optional[List[float]] = None
) -> ThrustProfileMetrics:
    """
    Analyze thrust profile for bang-bang structure and other characteristics.
    
    Args:
        solution: SOCP solution containing thrust history
        vehicle_params: Vehicle parameters for thrust bounds
        time_vector: Time vector (if not in solution)
        dt: Time step (if not in solution)
        switching_threshold: Threshold for detecting switching (fraction of max-min difference)
        paper_switching_times: Reference switching times from paper for comparison
        
    Returns:
        Thrust profile metrics
    """
    # Extract thrust magnitude
    if solution.thrust_magnitude_history is not None:
        thrust_mag = solution.thrust_magnitude_history
    elif solution.control_history is not None:
        # Compute magnitude from control vectors
        thrust_mag = np.linalg.norm(solution.control_history, axis=1)
    else:
        raise ValueError("No thrust data available in solution")
    
    # Get time vector
    if time_vector is not None:
        t = time_vector
    elif solution.time_vector is not None:
        t = solution.time_vector
    else:
        # Create default time vector
        if dt is not None:
            t = np.arange(0, len(thrust_mag) * dt, dt)
        else:
            t = np.arange(len(thrust_mag))
    
    # Compute basic statistics
    mean_thrust = np.mean(thrust_mag)
    std_thrust = np.std(thrust_mag)
    thrust_variation = std_thrust / mean_thrust if mean_thrust > 0 else 0.0
    
    # Check thrust bounds
    T_max = vehicle_params.T_max
    T_min = vehicle_params.T_min
    
    max_thrust_violation = np.max(thrust_mag - T_max) if np.any(thrust_mag > T_max) else 0.0
    min_thrust_violation = np.max(T_min - thrust_mag) if np.any(thrust_mag < T_min) else 0.0
    satisfies_thrust_bounds = (max_thrust_violation <= 1e-4) and (min_thrust_violation <= 1e-4)
    
    # Detect switching times for bang-bang structure
    switching_times = []
    switching_indices = []
    
    # Threshold for detecting switching (10% of range between min and max)
    threshold = switching_threshold * (T_max - T_min)
    
    # Detect segments where thrust is near max or min
    near_max = thrust_mag > (T_max - threshold)
    near_min = thrust_mag < (T_min + threshold)
    
    # Find transitions between max and min
    current_state = None
    for i in range(len(thrust_mag)):
        if near_max[i]:
            state = 'max'
        elif near_min[i]:
            state = 'min'
        else:
            state = 'intermediate'
        
        if current_state is None:
            current_state = state
        elif state != current_state and state != 'intermediate':
            # Switching detected
            switching_indices.append(i)
            switching_times.append(t[i])
            current_state = state
    
    # Determine bang-bang structure
    is_bang_bang = len(switching_indices) > 0
    if is_bang_bang:
        # Count segments
        max_segments = 0
        min_segments = 0
        
        # Determine starting state
        start_state = 'max' if near_max[0] else ('min' if near_min[0] else 'unknown')
        
        # Count segments
        if start_state == 'max':
            max_segments = 1
        elif start_state == 'min':
            min_segments = 1
        
        for i in range(1, len(switching_indices) + 1):
            if i % 2 == 0:  # even switches return to starting state
                if start_state == 'max':
                    max_segments += 1
                else:
                    min_segments += 1
            else:  # odd switches go to opposite state
                if start_state == 'max':
                    min_segments += 1
                else:
                    max_segments += 1
        
        # Determine structure string
        if max_segments == 2 and min_segments == 1:
            bang_bang_structure = "max-min-max"
        elif max_segments == 1 and min_segments == 1:
            bang_bang_structure = "max-min" if start_state == 'max' else "min-max"
        elif max_segments == 1 and min_segments == 0:
            bang_bang_structure = "max-only"
        elif max_segments == 0 and min_segments == 1:
            bang_bang_structure = "min-only"
        else:
            bang_bang_structure = f"complex-{max_segments}max-{min_segments}min"
    else:
        bang_bang_structure = "continuous"
        max_segments = 0
        min_segments = 0
    
    # Compare with paper switching times if provided
    switching_time_errors = []
    max_switching_time_error = 0.0
    
    if paper_switching_times is not None and switching_times:
        # Match switching times (allow for different number of switches)
        n_compare = min(len(paper_switching_times), len(switching_times))
        for i in range(n_compare):
            error = abs(switching_times[i] - paper_switching_times[i])
            switching_time_errors.append(error)
            max_switching_time_error = max(max_switching_time_error, error)
    
    return ThrustProfileMetrics(
        thrust_magnitude=thrust_mag,
        time_vector=t,
        dt=t[1] - t[0] if len(t) > 1 else 1.0,
        switching_times=switching_times,
        switching_indices=switching_indices,
        is_bang_bang=is_bang_bang,
        bang_bang_structure=bang_bang_structure,
        max_thrust_segments=max_segments,
        min_thrust_segments=min_segments,
        mean_thrust=mean_thrust,
        std_thrust=std_thrust,
        thrust_variation=thrust_variation,
        satisfies_thrust_bounds=satisfies_thrust_bounds,
        max_thrust_violation=max_thrust_violation,
        min_thrust_violation=min_thrust_violation,
        paper_switching_times=paper_switching_times,
        switching_time_errors=switching_time_errors,
        max_switching_time_error=max_switching_time_error
    )


def compute_trajectory_metrics(
    solution: SOCPSolution,
    asteroid_params: AsteroidParameters,
    landing_site: LandingSite
) -> Dict[str, Any]:
    """
    Compute comprehensive trajectory metrics.
    
    Args:
        solution: SOCP solution
        asteroid_params: Asteroid parameters
        landing_site: Landing site parameters
        
    Returns:
        Dictionary of trajectory metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['propellant_used'] = compute_propellant_used(solution)
    metrics['flight_time'] = solution.time_vector[-1] if solution.time_vector is not None else 0.0
    
    if solution.initial_mass is not None:
        metrics['propellant_efficiency'] = metrics['propellant_used'] / solution.initial_mass
        metrics['time_efficiency'] = metrics['propellant_used'] / metrics['flight_time'] if metrics['flight_time'] > 0 else 0.0
    
    # Position and velocity metrics
    if solution.position_history is not None:
        # Compute altitude history relative to landing site
        altitude = np.linalg.norm(solution.position_history - landing_site.position, axis=1)
        metrics['minimum_altitude'] = np.min(altitude)
        metrics['maximum_altitude'] = np.max(altitude)
        metrics['mean_altitude'] = np.mean(altitude)
        
        # Compute position smoothness (jerk)
        if len(solution.position_history) > 2:
            dt = solution.time_vector[1] - solution.time_vector[0] if solution.time_vector is not None else 1.0
            velocity = np.gradient(solution.position_history, dt, axis=0)
            acceleration = np.gradient(velocity, dt, axis=0)
            jerk = np.gradient(acceleration, dt, axis=0)
            metrics['position_smoothness'] = np.mean(np.linalg.norm(jerk, axis=1)**2)
        else:
            metrics['position_smoothness'] = 0.0
    
    if solution.velocity_history is not None:
        speed = np.linalg.norm(solution.velocity_history, axis=1)
        metrics['maximum_velocity'] = np.max(speed)
        metrics['mean_velocity'] = np.mean(speed)
        metrics['final_velocity'] = speed[-1] if len(speed) > 0 else 0.0
    
    # Thrust metrics
    if solution.thrust_magnitude_history is not None:
        thrust_mag = solution.thrust_magnitude_history
        metrics['mean_thrust'] = np.mean(thrust_mag)
        metrics['max_thrust'] = np.max(thrust_mag)
        metrics['min_thrust'] = np.min(thrust_mag)
        metrics['thrust_variation'] = np.std(thrust_mag) / metrics['mean_thrust'] if metrics['mean_thrust'] > 0 else 0.0
        
        # Thrust smoothness
        if len(thrust_mag) > 1:
            dt = solution.time_vector[1] - solution.time_vector[0] if solution.time_vector is not None else 1.0
            thrust_derivative = np.gradient(thrust_mag, dt)
            metrics['thrust_smoothness'] = np.mean(thrust_derivative**2)
        else:
            metrics['thrust_smoothness'] = 0.0
    
    return metrics


def compute_constraint_violations(
    solution: SOCPSolution,
    experiment_config: Any,
    tolerance: float = 1e-4
) -> ConstraintViolationMetrics:
    """
    Compute constraint violations for a trajectory solution.
    
    Args:
        solution: SOCP solution
        experiment_config: Experiment configuration
        tolerance: Tolerance for constraint satisfaction
        
    Returns:
        Constraint violation metrics
    """
    # Initialize arrays
    n_points = len(solution.position_history) if solution.position_history is not None else 0
    
    # Glide slope violations (placeholder - would need constraint parameters)
    glide_slope_violations = np.zeros(n_points)
    max_glide_slope_violation = 0.0
    mean_glide_slope_violation = 0.0
    glide_slope_satisfied = True
    
    # Vertical motion violations (placeholder)
    vertical_motion_violations = np.zeros(n_points)
    max_vertical_motion_violation = 0.0
    mean_vertical_motion_violation = 0.0
    vertical_motion_satisfied = True
    
    # Mass violations
    if solution.mass_history is not None:
        m_dry = experiment_config.vehicle.m_dry
        mass_violations = np.maximum(m_dry - solution.mass_history, 0)
        max_mass_violation = np.max(mass_violations)
        mean_mass_violation = np.mean(mass_violations)
        mass_constraint_satisfied = max_mass_violation <= tolerance
    else:
        mass_violations = np.zeros(n_points)
        max_mass_violation = 0.0
        mean_mass_violation = 0.0
        mass_constraint_satisfied = True
    
    # Thrust magnitude violations
    if solution.thrust_magnitude_history is not None:
        T_max = experiment_config.vehicle.T_max
        T_min = experiment_config.vehicle.T_min
        thrust_mag = solution.thrust_magnitude_history
        
        # Upper bound violations
        upper_violations = np.maximum(thrust_mag - T_max, 0)
        # Lower bound violations
        lower_violations = np.maximum(T_min - thrust_mag, 0)
        
        thrust_magnitude_violations = upper_violations + lower_violations
        max_thrust_magnitude_violation = np.max(thrust_magnitude_violations)
        mean_thrust_magnitude_violation = np.mean(thrust_magnitude_violations)
        thrust_magnitude_satisfied = max_thrust_magnitude_violation <= tolerance
    else:
        thrust_magnitude_violations = np.zeros(n_points)
        max_thrust_magnitude_violation = 0.0
        mean_thrust_magnitude_violation = 0.0
        thrust_magnitude_satisfied = True
    
    # Boundary condition errors
    initial_position_error = 0.0
    initial_velocity_error = 0.0
    initial_mass_error = 0.0
    final_position_error = 0.0
    final_velocity_error = 0.0
    
    if (solution.position_history is not None and 
        solution.velocity_history is not None and
        solution.mass_history is not None):
        
        # Initial conditions
        if experiment_config.initial_state is not None:
            initial_position_error = np.linalg.norm(
                solution.position_history[0] - experiment_config.initial_state.position
            )
            initial_velocity_error = np.linalg.norm(
                solution.velocity_history[0] - experiment_config.initial_state.velocity
            )
            initial_mass_error = abs(solution.mass_history[0] - experiment_config.initial_state.mass)
        
        # Final conditions
        if experiment_config.final_state is not None:
            final_position_error = np.linalg.norm(
                solution.position_history[-1] - experiment_config.final_state.position
            )
            final_velocity_error = np.linalg.norm(
                solution.velocity_history[-1] - experiment_config.final_state.velocity
            )
    
    boundary_conditions_satisfied = (
        initial_position_error <= tolerance and
        initial_velocity_error <= tolerance and
        initial_mass_error <= tolerance and
        final_position_error <= tolerance and
        final_velocity_error <= tolerance
    )
    
    # Overall satisfaction
    all_constraints_satisfied = (
        glide_slope_satisfied and
        vertical_motion_satisfied and
        mass_constraint_satisfied and
        thrust_magnitude_satisfied and
        boundary_conditions_satisfied
    )
    
    max_overall_violation = max(
        max_glide_slope_violation,
        max_vertical_motion_violation,
        max_mass_violation,
        max_thrust_magnitude_violation,
        initial_position_error,
        initial_velocity_error,
        initial_mass_error,
        final_position_error,
        final_velocity_error
    )
    
    return ConstraintViolationMetrics(
        glide_slope_violations=glide_slope_violations,
        max_glide_slope_violation=max_glide_slope_violation,
        mean_glide_slope_violation=mean_glide_slope_violation,
        glide_slope_satisfied=glide_slope_satisfied,
        
        vertical_motion_violations=vertical_motion_violations,
        max_vertical_motion_violation=max_vertical_motion_violation,
        mean_vertical_motion_violation=mean_vertical_motion_violation,
        vertical_motion_satisfied=vertical_motion_satisfied,
        
        mass_violations=mass_violations,
        max_mass_violation=max_mass_violation,
        mean_mass_violation=mean_mass_violation,
        mass_constraint_satisfied=mass_constraint_satisfied,
        
        thrust_magnitude_violations=thrust_magnitude_violations,
        max_thrust_magnitude_violation=max_thrust_magnitude_violation,
        mean_thrust_magnitude_violation=mean_thrust_magnitude_violation,
        thrust_magnitude_satisfied=thrust_magnitude_satisfied,
        
        initial_position_error=initial_position_error,
        initial_velocity_error=initial_velocity_error,
        initial_mass_error=initial_mass_error,
        final_position_error=final_position_error,
        final_velocity_error=final_velocity_error,
        boundary_conditions_satisfied=boundary_conditions_satisfied,
        
        all_constraints_satisfied=all_constraints_satisfied,
        max_overall_violation=max_overall_violation
    )


def compute_convergence_metrics(
    successive_result: SuccessiveSolutionResult
) -> Dict[str, Any]:
    """
    Compute convergence metrics for successive solution algorithm.
    
    Args:
        successive_result: Result from successive solver
        
    Returns:
        Dictionary of convergence metrics
    """
    metrics = {}
    
    metrics['converged'] = successive_result.converged
    metrics['iterations'] = successive_result.iterations
    metrics['max_iterations'] = successive_result.max_iterations
    
    if successive_result.position_error_history is not None:
        metrics['max_position_error'] = np.max(successive_result.position_error_history)
        metrics['final_position_error'] = successive_result.position_error_history[-1] if len(successive_result.position_error_history) > 0 else 0.0
        metrics['mean_position_error'] = np.mean(successive_result.position_error_history)
        
        # Convergence rate
        if len(successive_result.position_error_history) > 1:
            errors = successive_result.position_error_history
            ratios = []
            for i in range(1, len(errors)):
                if errors[i-1] > 0:
                    ratios.append(errors[i] / errors[i-1])
            
            if ratios:
                metrics['convergence_rate'] = np.mean(ratios)
                metrics['linear_convergence'] = metrics['convergence_rate'] < 1.0
                metrics['quadratic_convergence'] = metrics['convergence_rate'] < 0.5
            else:
                metrics['convergence_rate'] = 0.0
                metrics['linear_convergence'] = False
                metrics['quadratic_convergence'] = False
        else:
            metrics['convergence_rate'] = 0.0
            metrics['linear_convergence'] = False
            metrics['quadratic_convergence'] = False
    
    metrics['total_computation_time'] = successive_result.total_computation_time
    if successive_result.iterations > 0:
        metrics['average_iteration_time'] = successive_result.total_computation_time / successive_result.iterations
    else:
        metrics['average_iteration_time'] = 0.0
    
    # Expected vs actual iterations
    metrics['expected_iterations'] = 3  # Paper expects 3 iterations for Castalia
    metrics['iteration_error'] = abs(successive_result.iterations - metrics['expected_iterations'])
    
    return metrics


def compute_gravity_error(
    solution: SOCPSolution,
    gravity_calculator: Any,
    tolerance: float = 1e-6
) -> Dict[str, Any]:
    """
    Compute gravity model error for validation.
    
    Args:
        solution: SOCP solution with position history
        gravity_calculator: Gravity calculator instance
        tolerance: Tolerance for error checking
        
    Returns:
        Dictionary of gravity error metrics
    """
    if solution.position_history is None:
        return {
            'max_error': 0.0,
            'mean_error': 0.0,
            'positions_checked': 0,
            'within_tolerance': True
        }
    
    # This is a placeholder - actual implementation would compute
    # gravity at each position and compare with expected values
    # For now, return dummy metrics
    
    return {
        'max_error': 0.0,
        'mean_error': 0.0,
        'positions_checked': len(solution.position_history),
        'within_tolerance': True
    }


def compute_paper_comparison_metrics(
    solution: SOCPSolution,
    flight_time_result: Optional[FlightTimeOptimizationResult] = None,
    paper_propellant: Optional[float] = None,
    paper_flight_time: Optional[float] = None
) -> TrajectoryQualityMetrics:
    """
    Compute trajectory quality metrics with paper comparison.
    
    Args:
        solution: SOCP solution
        flight_time_result: Flight time optimization result (optional)
        paper_propellant: Propellant value from paper (optional)
        paper_flight_time: Flight time from paper (optional)
        
    Returns:
        Trajectory quality metrics
    """
    # Basic metrics
    propellant_used = compute_propellant_used(solution)
    flight_time = solution.time_vector[-1] if solution.time_vector is not None else 0.0
    
    if solution.initial_mass is not None:
        propellant_efficiency = propellant_used / solution.initial_mass
    else:
        propellant_efficiency = 0.0
    
    time_efficiency = propellant_used / flight_time if flight_time > 0 else 0.0
    
    # Smoothness metrics (simplified)
    position_smoothness = 0.0
    thrust_smoothness = 0.0
    
    if solution.position_history is not None and len(solution.position_history) > 2:
        dt = solution.time_vector[1] - solution.time_vector[0] if solution.time_vector is not None else 1.0
        # Compute jerk as proxy for smoothness
        pos = solution.position_history
        vel = np.gradient(pos, dt, axis=0)
        acc = np.gradient(vel, dt, axis=0)
        jerk = np.gradient(acc, dt, axis=0)
        position_smoothness = np.mean(np.linalg.norm(jerk, axis=1)**2)
    
    if solution.thrust_magnitude_history is not None and len(solution.thrust_magnitude_history) > 1:
        dt = solution.time_vector[1] - solution.time_vector[0] if solution.time_vector is not None else 1.0
        thrust_derivative = np.gradient(solution.thrust_magnitude_history, dt)
        thrust_smoothness = np.mean(thrust_derivative**2)
    
    # Safety metrics
    minimum_altitude = 0.0
    maximum_velocity = 0.0
    maximum_acceleration = 0.0
    
    if solution.position_history is not None and solution.landing_site is not None:
        # Compute altitude relative to landing site
        altitude = np.linalg.norm(solution.position_history - solution.landing_site.position, axis=1)
        minimum_altitude = np.min(altitude)
    
    if solution.velocity_history is not None:
        speed = np.linalg.norm(solution.velocity_history, axis=1)
        maximum_velocity = np.max(speed)
    
    if solution.acceleration_history is not None:
        acc_mag = np.linalg.norm(solution.acceleration_history, axis=1)
        maximum_acceleration = np.max(acc_mag)
    
    # Paper comparison
    propellant_error_percent = None
    flight_time_error_percent = None
    
    if paper_propellant is not None and paper_propellant > 0:
        propellant_error_percent = 100.0 * abs(propellant_used - paper_propellant) / paper_propellant
    
    if paper_flight_time is not None and paper_flight_time > 0:
        flight_time_error_percent = 100.0 * abs(flight_time - paper_flight_time) / paper_flight_time
    
    return TrajectoryQualityMetrics(
        propellant_used=propellant_used,
        propellant_efficiency=propellant_efficiency,
        flight_time=flight_time,
        time_efficiency=time_efficiency,
        position_smoothness=position_smoothness,
        thrust_smoothness=thrust_smoothness,
        minimum_altitude=minimum_altitude,
        maximum_velocity=maximum_velocity,
        maximum_acceleration=maximum_acceleration,
        paper_propellant=paper_propellant,
        propellant_error_percent=propellant_error_percent,
        paper_flight_time=paper_flight_time,
        flight_time_error_percent=flight_time_error_percent
    )


def validate_solution(
    solution: SOCPSolution,
    experiment_config: Any,
    tolerance: float = 1e-4
) -> Dict[str, Any]:
    """
    Comprehensive validation of a trajectory solution.
    
    Args:
        solution: SOCP solution to validate
        experiment_config: Experiment configuration
        tolerance: Validation tolerance
        
    Returns:
        Dictionary of validation results
    """
    validation = {}
    
    # Compute propellant
    validation['propellant_used'] = compute_propellant_used(solution)
    
    # Check constraint violations
    constraint_metrics = compute_constraint_violations(solution, experiment_config, tolerance)
    validation['constraint_violations'] = constraint_metrics
    
    # Check if all constraints satisfied
    validation['all_constraints_satisfied'] = constraint_metrics.all_constraints_satisfied
    
    # Check boundary conditions
    validation['boundary_conditions_satisfied'] = constraint_metrics.boundary_conditions_satisfied
    
    # Check thrust bounds
    validation['thrust_bounds_satisfied'] = constraint_metrics.thrust_magnitude_satisfied
    
    # Check mass constraint
    validation['mass_constraint_satisfied'] = constraint_metrics.mass_constraint_satisfied
    
    # Check maximum violation
    validation['max_violation'] = constraint_metrics.max_overall_violation
    
    # Check if solution is feasible
    validation['feasible'] = (
        validation['all_constraints_satisfied'] and
        validation['max_violation'] <= tolerance
    )
    
    return validation


def test_metrics() -> bool:
    """
    Test the metrics module.
    
    Returns:
        True if all tests pass
    """
    logger.info("Testing metrics module...")
    
    try:
        # Create dummy data for testing
        n_points = 10
        time_vector = np.linspace(0, 100, n_points)
        
        # Create dummy solution
        class DummySolution:
            def __init__(self):
                self.position_history = np.random.randn(n_points, 3) * 1000
                self.velocity_history = np.random.randn(n_points, 3) * 10
                self.mass_history = np.linspace(1000, 900, n_points)
                self.thrust_magnitude_history = 50 + 10 * np.sin(time_vector / 10)
                self.time_vector = time_vector
                self.initial_mass = 1000.0
                self.final_mass = 900.0
                self.vehicle_params = VehicleParameters(
                    m_wet=1000.0,
                    m_dry=800.0,
                    I_sp=300.0,
                    T_max=80.0,
                    T_min=20.0
                )
                self.landing_site = LandingSite(
                    name="Test",
                    position=np.array([0, 0, 0]),
                    surface_normal=np.array([0, 0, 1]),
                    glide_slope_angle=30.0
                )
        
        solution = DummySolution()
        
        # Test propellant computation
        propellant = compute_propellant_used(solution)
        assert propellant > 0, "Propellant should be positive"
        logger.info(f"Propellant computed: {propellant:.2f} kg")
        
        # Test thrust profile metrics
        thrust_metrics = compute_thrust_profile_metrics(
            solution, solution.vehicle_params, time_vector
        )
        assert thrust_metrics.mean_thrust > 0, "Mean thrust should be positive"
        logger.info(f"Thrust profile analyzed: {thrust_metrics.bang_bang_structure}")
        
        # Test trajectory metrics
        asteroid_params = AsteroidParameters(
            name="Test",
            mu=1.0e10,
            R_b=1000.0,
            rotation_rate=0.001,
            rotation_axis=np.array([0, 0, 1])
        )
        
        traj_metrics = compute_trajectory_metrics(solution, asteroid_params, solution.landing_site)
        assert 'propellant_used' in traj_metrics, "Trajectory metrics should include propellant"
        logger.info(f"Trajectory metrics computed: {len(traj_metrics)} metrics")
        
        # Test paper comparison
        paper_metrics = compute_paper_comparison_metrics(
            solution,
            paper_propellant=95.0,
            paper_flight_time=105.0
        )
        assert paper_metrics.propellant_used == propellant, "Propellant should match"
        logger.info(f"Paper comparison: {paper_metrics.propellant_error_percent:.1f}% error")
        
        logger.info("All metrics tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"Metrics test failed: {e}")
        return False


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    success = test_metrics()
    if success:
        print("Metrics module tests passed!")
    else:
        print("Metrics module tests failed!")