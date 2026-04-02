"""
Trajectory analysis module for asteroid landing convex optimization.

This module provides functions to analyze trajectory solutions, including:
- Thrust profile analysis (switching times, bang-bang structure)
- Convergence metrics for successive solution algorithm
- Constraint violation analysis
- Trajectory quality metrics
- Comparison with paper results (Tables 5, 7)

Author: Implementation Team
Date: 2026-01-15
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from pathlib import Path

from ..optimization.socp_solver import SOCPSolution
from ..optimization.successive_solver import SuccessiveSolutionResult
from ..optimization.flight_time_optimizer import FlightTimeOptimizationResult
from ..config import VehicleParameters, AsteroidParameters, LandingSite
from ..dynamics.state_equations import StateVector
from ..utils.metrics import compute_thrust_profile_metrics, compute_propellant_used
from ..utils.convergence_check import check_convergence_metrics

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ThrustProfileAnalysis:
    """Analysis results for a thrust profile."""
    
    # Basic metrics
    thrust_magnitude: np.ndarray
    time_vector: np.ndarray
    dt: float
    
    # Switching times (where thrust magnitude changes significantly)
    switching_times: List[float] = field(default_factory=list)
    switching_indices: List[int] = field(default_factory=list)
    
    # Bang-bang structure analysis
    is_bang_bang: bool = False
    bang_bang_structure: str = ""  # e.g., "max-min-max", "max-min"
    max_thrust_segments: List[Tuple[int, int]] = field(default_factory=list)
    min_thrust_segments: List[Tuple[int, int]] = field(default_factory=list)
    
    # Statistical metrics
    mean_thrust: float = 0.0
    std_thrust: float = 0.0
    thrust_variation: float = 0.0  # (max - min) / max
    
    # Constraint satisfaction
    satisfies_thrust_bounds: bool = False
    max_thrust_violation: float = 0.0
    min_thrust_violation: float = 0.0
    
    # Paper comparison metrics (for Castalia experiments)
    paper_switching_times: Optional[List[float]] = None
    switching_time_errors: Optional[List[float]] = None
    max_switching_time_error: float = 0.0
    
    def __post_init__(self):
        """Compute derived metrics after initialization."""
        if len(self.thrust_magnitude) > 0:
            self.mean_thrust = float(np.mean(self.thrust_magnitude))
            self.std_thrust = float(np.std(self.thrust_magnitude))
            thrust_min = np.min(self.thrust_magnitude)
            thrust_max = np.max(self.thrust_magnitude)
            if thrust_max > 0:
                self.thrust_variation = (thrust_max - thrust_min) / thrust_max


@dataclass
class TrajectoryConvergenceAnalysis:
    """Analysis of convergence for successive solution algorithm."""
    
    # Convergence metrics
    converged: bool = False
    iterations: int = 0
    max_position_error: float = 0.0
    final_position_error: float = 0.0
    position_error_history: List[float] = field(default_factory=list)
    
    # Timing metrics
    total_computation_time: float = 0.0
    average_iteration_time: float = 0.0
    
    # Convergence rate
    convergence_rate: float = 0.0  # Average reduction per iteration
    linear_convergence: bool = False
    quadratic_convergence: bool = False
    
    # Paper comparison
    expected_iterations: Optional[int] = None
    iteration_error: Optional[int] = None
    
    def __post_init__(self):
        """Compute derived convergence metrics."""
        if self.iterations > 0:
            self.average_iteration_time = self.total_computation_time / self.iterations
            
        # Compute convergence rate if we have error history
        if len(self.position_error_history) >= 2:
            errors = np.array(self.position_error_history)
            if errors[0] > 0:
                # Compute average reduction factor
                reductions = errors[1:] / errors[:-1]
                self.convergence_rate = float(np.mean(reductions))
                
                # Check convergence type
                self.linear_convergence = 0.1 < self.convergence_rate < 0.9
                self.quadratic_convergence = self.convergence_rate < 0.1


@dataclass
class ConstraintViolationAnalysis:
    """Analysis of constraint violations in a trajectory."""
    
    # Glide slope violations
    glide_slope_violations: np.ndarray = field(default_factory=lambda: np.array([]))
    max_glide_slope_violation: float = 0.0
    mean_glide_slope_violation: float = 0.0
    glide_slope_satisfied: bool = False
    
    # Vertical motion violations
    vertical_motion_violations: np.ndarray = field(default_factory=lambda: np.array([]))
    max_vertical_motion_violation: float = 0.0
    mean_vertical_motion_violation: float = 0.0
    vertical_motion_satisfied: bool = False
    
    # Mass constraint violations
    mass_violations: np.ndarray = field(default_factory=lambda: np.array([]))
    max_mass_violation: float = 0.0
    mean_mass_violation: float = 0.0
    mass_constraint_satisfied: bool = False
    
    # Thrust magnitude violations (from lossless convexification)
    thrust_magnitude_violations: np.ndarray = field(default_factory=lambda: np.array([]))
    max_thrust_magnitude_violation: float = 0.0
    mean_thrust_magnitude_violation: float = 0.0
    thrust_magnitude_satisfied: bool = False
    
    # Boundary condition violations
    initial_position_error: float = 0.0
    initial_velocity_error: float = 0.0
    initial_mass_error: float = 0.0
    final_position_error: float = 0.0
    final_velocity_error: float = 0.0
    boundary_conditions_satisfied: bool = False
    
    # Overall satisfaction
    all_constraints_satisfied: bool = False
    max_overall_violation: float = 0.0


@dataclass
class TrajectoryQualityMetrics:
    """Overall quality metrics for a trajectory."""
    
    # Propellant metrics
    propellant_used: float = 0.0
    propellant_efficiency: float = 0.0  # propellant_used / theoretical_minimum
    
    # Time metrics
    flight_time: float = 0.0
    time_efficiency: float = 0.0  # actual_time / optimal_time
    
    # Smoothness metrics
    position_smoothness: float = 0.0  # Integrated squared jerk
    thrust_smoothness: float = 0.0  # Integrated squared thrust rate
    
    # Safety metrics
    minimum_altitude: float = 0.0
    maximum_velocity: float = 0.0
    maximum_acceleration: float = 0.0
    
    # Paper comparison
    paper_propellant: Optional[float] = None
    propellant_error_percent: float = 0.0
    paper_flight_time: Optional[float] = None
    flight_time_error_percent: float = 0.0


@dataclass
class CompleteTrajectoryAnalysis:
    """Complete analysis of a trajectory solution."""
    
    # Basic information
    asteroid_name: str
    landing_site_name: str
    thrust_config: str  # "full" or "quarter"
    
    # Component analyses
    thrust_analysis: ThrustProfileAnalysis
    convergence_analysis: TrajectoryConvergenceAnalysis
    constraint_analysis: ConstraintViolationAnalysis
    quality_metrics: TrajectoryQualityMetrics
    
    # Solution reference
    solution: Optional[SOCPSolution] = None
    successive_result: Optional[SuccessiveSolutionResult] = None
    flight_time_result: Optional[FlightTimeOptimizationResult] = None
    
    # Analysis metadata
    analysis_timestamp: str = ""
    computation_time: float = 0.0


def analyze_thrust_profile(
    solution: SOCPSolution,
    vehicle_params: VehicleParameters,
    time_vector: Optional[np.ndarray] = None,
    dt: Optional[float] = None,
    switching_threshold: float = 0.1,  # 10% change threshold
    paper_switching_times: Optional[List[float]] = None
) -> ThrustProfileAnalysis:
    """
    Analyze thrust profile for bang-bang structure and switching times.
    
    Args:
        solution: SOCP solution containing thrust profile
        vehicle_params: Vehicle parameters for thrust bounds
        time_vector: Time vector (if None, computed from solution)
        dt: Time step (if None, inferred from solution)
        switching_threshold: Relative change threshold for detecting switching
        paper_switching_times: Switching times from paper for comparison
        
    Returns:
        ThrustProfileAnalysis object with analysis results
    """
    # Extract thrust profile
    thrust_profile = solution.compute_thrust_profile()
    
    if time_vector is None:
        # Create time vector from solution
        N = len(thrust_profile)
        if dt is None:
            dt = solution.discretization_params.dt if hasattr(solution, 'discretization_params') else 2.0
        time_vector = np.arange(N) * dt
    
    # Create analysis object
    analysis = ThrustProfileAnalysis(
        thrust_magnitude=thrust_profile,
        time_vector=time_vector,
        dt=dt if dt is not None else time_vector[1] - time_vector[0],
        paper_switching_times=paper_switching_times
    )
    
    # Check thrust bounds satisfaction
    T_max = vehicle_params.T_max
    T_min = vehicle_params.T_min
    
    # Compute violations
    max_violation = np.max(thrust_profile - T_max)
    min_violation = np.max(T_min - thrust_profile)
    
    analysis.satisfies_thrust_bounds = (max_violation <= 1e-4 and min_violation <= 1e-4)
    analysis.max_thrust_violation = float(max_violation)
    analysis.min_thrust_violation = float(min_violation)
    
    # Detect switching times
    if len(thrust_profile) > 1:
        # Compute relative changes
        changes = np.abs(np.diff(thrust_profile)) / (np.max(thrust_profile) + 1e-10)
        
        # Find indices where change exceeds threshold
        switch_indices = np.where(changes > switching_threshold)[0]
        
        # Convert to times
        analysis.switching_indices = switch_indices.tolist()
        analysis.switching_times = time_vector[switch_indices].tolist()
        
        # Analyze bang-bang structure
        _analyze_bang_bang_structure(analysis, thrust_profile, T_max, T_min)
        
        # Compare with paper switching times if provided
        if paper_switching_times is not None:
            analysis.switching_time_errors = []
            for i, paper_time in enumerate(paper_switching_times):
                if i < len(analysis.switching_times):
                    error = abs(analysis.switching_times[i] - paper_time)
                    analysis.switching_time_errors.append(error)
                else:
                    analysis.switching_time_errors.append(float('inf'))
            
            if analysis.switching_time_errors:
                analysis.max_switching_time_error = max(analysis.switching_time_errors)
    
    return analysis


def _analyze_bang_bang_structure(
    analysis: ThrustProfileAnalysis,
    thrust_profile: np.ndarray,
    T_max: float,
    T_min: float
) -> None:
    """
    Analyze bang-bang structure of thrust profile.
    
    Args:
        analysis: ThrustProfileAnalysis to update
        thrust_profile: Thrust magnitude array
        T_max: Maximum thrust bound
        T_min: Minimum thrust bound
    """
    # Tolerance for considering thrust at bound
    tol = 0.05 * (T_max - T_min)
    
    # Check if profile is bang-bang
    at_max = np.abs(thrust_profile - T_max) < tol
    at_min = np.abs(thrust_profile - T_min) < tol
    at_bounds = at_max | at_min
    
    analysis.is_bang_bang = np.all(at_bounds)
    
    if not analysis.is_bang_bang:
        return
    
    # Identify segments
    segments = []
    current_segment = 0
    current_value = thrust_profile[0]
    
    for i in range(1, len(thrust_profile)):
        if abs(thrust_profile[i] - current_value) > tol:
            segments.append((current_segment, i-1, current_value))
            current_segment = i
            current_value = thrust_profile[i]
    
    # Add last segment
    segments.append((current_segment, len(thrust_profile)-1, current_value))
    
    # Classify structure
    if len(segments) == 2:
        if segments[0][2] == T_max and segments[1][2] == T_min:
            analysis.bang_bang_structure = "max-min"
        elif segments[0][2] == T_min and segments[1][2] == T_max:
            analysis.bang_bang_structure = "min-max"
    elif len(segments) == 3:
        if (segments[0][2] == T_max and segments[1][2] == T_min and 
            segments[2][2] == T_max):
            analysis.bang_bang_structure = "max-min-max"
        elif (segments[0][2] == T_min and segments[1][2] == T_max and 
              segments[2][2] == T_min):
            analysis.bang_bang_structure = "min-max-min"
    
    # Store segment information
    for start, end, value in segments:
        if abs(value - T_max) < tol:
            analysis.max_thrust_segments.append((start, end))
        elif abs(value - T_min) < tol:
            analysis.min_thrust_segments.append((start, end))


def analyze_convergence(
    successive_result: SuccessiveSolutionResult,
    expected_iterations: Optional[int] = None
) -> TrajectoryConvergenceAnalysis:
    """
    Analyze convergence of successive solution algorithm.
    
    Args:
        successive_result: Result from successive solver
        expected_iterations: Expected number of iterations from paper
        
    Returns:
        TrajectoryConvergenceAnalysis object
    """
    analysis = TrajectoryConvergenceAnalysis(
        converged=successive_result.converged,
        iterations=successive_result.iterations,
        max_position_error=successive_result.max_position_error,
        final_position_error=successive_result.final_position_error,
        position_error_history=successive_result.position_error_history,
        total_computation_time=successive_result.total_computation_time,
        expected_iterations=expected_iterations
    )
    
    # Compute iteration error if expected iterations provided
    if expected_iterations is not None:
        analysis.iteration_error = analysis.iterations - expected_iterations
    
    return analysis


def analyze_constraint_violations(
    solution: SOCPSolution,
    experiment_config,
    tolerance: float = 1e-4
) -> ConstraintViolationAnalysis:
    """
    Analyze constraint violations in a trajectory solution.
    
    Args:
        solution: SOCP solution to analyze
        experiment_config: Experiment configuration
        tolerance: Tolerance for constraint satisfaction
        
    Returns:
        ConstraintViolationAnalysis object
    """
    analysis = ConstraintViolationAnalysis()
    
    # TODO: Implement detailed constraint violation analysis
    # This would check:
    # 1. Glide slope constraints at each time step
    # 2. Vertical motion constraints
    # 3. Mass constraints
    # 4. Thrust magnitude constraints (lossless convexification)
    # 5. Boundary conditions
    
    # For now, set basic satisfaction based on solution status
    analysis.all_constraints_satisfied = solution.success
    
    return analysis


def analyze_trajectory_quality(
    solution: SOCPSolution,
    flight_time_result: Optional[FlightTimeOptimizationResult] = None,
    paper_propellant: Optional[float] = None,
    paper_flight_time: Optional[float] = None
) -> TrajectoryQualityMetrics:
    """
    Analyze overall quality metrics for a trajectory.
    
    Args:
        solution: SOCP solution
        flight_time_result: Flight time optimization result (optional)
        paper_propellant: Propellant value from paper for comparison
        paper_flight_time: Flight time from paper for comparison
        
    Returns:
        TrajectoryQualityMetrics object
    """
    metrics = TrajectoryQualityMetrics(
        paper_propellant=paper_propellant,
        paper_flight_time=paper_flight_time
    )
    
    # Propellant metrics
    metrics.propellant_used = compute_propellant_used(solution)
    metrics.flight_time = solution.discretization_params.t_f
    
    # Paper comparison
    if paper_propellant is not None and paper_propellant > 0:
        metrics.propellant_error_percent = abs(
            (metrics.propellant_used - paper_propellant) / paper_propellant * 100
        )
    
    if paper_flight_time is not None and paper_flight_time > 0:
        metrics.flight_time_error_percent = abs(
            (metrics.flight_time - paper_flight_time) / paper_flight_time * 100
        )
    
    # TODO: Compute additional quality metrics:
    # - Smoothness (jerk, thrust rate)
    # - Safety metrics (minimum altitude, maximum velocity/acceleration)
    # - Efficiency metrics
    
    return metrics


def analyze_complete_trajectory(
    solution: SOCPSolution,
    successive_result: SuccessiveSolutionResult,
    flight_time_result: Optional[FlightTimeOptimizationResult],
    experiment_config,
    paper_data: Optional[Dict[str, Any]] = None
) -> CompleteTrajectoryAnalysis:
    """
    Perform complete analysis of a trajectory solution.
    
    Args:
        solution: SOCP solution
        successive_result: Successive solution result
        flight_time_result: Flight time optimization result
        experiment_config: Experiment configuration
        paper_data: Paper data for comparison (optional)
        
    Returns:
        CompleteTrajectoryAnalysis object
    """
    # Extract paper data if provided
    paper_switching_times = None
    paper_propellant = None
    paper_flight_time = None
    expected_iterations = None
    
    if paper_data:
        paper_switching_times = paper_data.get('switching_times')
        paper_propellant = paper_data.get('propellant_used')
        paper_flight_time = paper_data.get('optimal_flight_time')
        expected_iterations = paper_data.get('expected_iterations')
    
    # Perform component analyses
    thrust_analysis = analyze_thrust_profile(
        solution=solution,
        vehicle_params=experiment_config.vehicle,
        paper_switching_times=paper_switching_times
    )
    
    convergence_analysis = analyze_convergence(
        successive_result=successive_result,
        expected_iterations=expected_iterations
    )
    
    constraint_analysis = analyze_constraint_violations(
        solution=solution,
        experiment_config=experiment_config
    )
    
    quality_metrics = analyze_trajectory_quality(
        solution=solution,
        flight_time_result=flight_time_result,
        paper_propellant=paper_propellant,
        paper_flight_time=paper_flight_time
    )
    
    # Create complete analysis
    analysis = CompleteTrajectoryAnalysis(
        asteroid_name=experiment_config.asteroid.name,
        landing_site_name=experiment_config.landing_site.name,
        thrust_config="full" if experiment_config.vehicle.T_max > 40 else "quarter",
        thrust_analysis=thrust_analysis,
        convergence_analysis=convergence_analysis,
        constraint_analysis=constraint_analysis,
        quality_metrics=quality_metrics,
        solution=solution,
        successive_result=successive_result,
        flight_time_result=flight_time_result
    )
    
    return analysis


def print_trajectory_analysis_summary(analysis: CompleteTrajectoryAnalysis) -> None:
    """
    Print a summary of trajectory analysis results.
    
    Args:
        analysis: Complete trajectory analysis
    """
    print("\n" + "="*80)
    print("TRAJECTORY ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\nScenario: {analysis.asteroid_name} - {analysis.landing_site_name}")
    print(f"Thrust configuration: {analysis.thrust_config}")
    
    print("\n1. PROPULSION PERFORMANCE")
    print(f"   Propellant used: {analysis.quality_metrics.propellant_used:.3f} kg")
    if analysis.quality_metrics.paper_propellant is not None:
        print(f"   Paper value: {analysis.quality_metrics.paper_propellant:.3f} kg")
        print(f"   Error: {analysis.quality_metrics.propellant_error_percent:.2f}%")
    
    print(f"\n2. FLIGHT TIME")
    print(f"   Optimal flight time: {analysis.quality_metrics.flight_time:.1f} s")
    if analysis.quality_metrics.paper_flight_time is not None:
        print(f"   Paper value: {analysis.quality_metrics.paper_flight_time:.1f} s")
        print(f"   Error: {analysis.quality_metrics.flight_time_error_percent:.2f}%")
    
    print("\n3. THRUST PROFILE")
    print(f"   Bang-bang structure: {analysis.thrust_analysis.bang_bang_structure}")
    print(f"   Switching times: {analysis.thrust_analysis.switching_times}")
    if analysis.thrust_analysis.paper_switching_times:
        print(f"   Paper switching times: {analysis.thrust_analysis.paper_switching_times}")
        if analysis.thrust_analysis.switching_time_errors:
            print(f"   Switching time errors: {analysis.thrust_analysis.switching_time_errors}")
            print(f"   Max error: {analysis.thrust_analysis.max_switching_time_error:.2f} s")
    
    print("\n4. CONVERGENCE")
    print(f"   Converged: {analysis.convergence_analysis.converged}")
    print(f"   Iterations: {analysis.convergence_analysis.iterations}")
    if analysis.convergence_analysis.expected_iterations is not None:
        print(f"   Expected iterations: {analysis.convergence_analysis.expected_iterations}")
        if analysis.convergence_analysis.iteration_error is not None:
            print(f"   Iteration error: {analysis.convergence_analysis.iteration_error}")
    print(f"   Final position error: {analysis.convergence_analysis.final_position_error:.3f} m")
    print(f"   Convergence rate: {analysis.convergence_analysis.convergence_rate:.3f}")
    
    print("\n5. CONSTRAINT SATISFACTION")
    print(f"   All constraints satisfied: {analysis.constraint_analysis.all_constraints_satisfied}")
    print(f"   Thrust bounds satisfied: {analysis.thrust_analysis.satisfies_thrust_bounds}")
    
    print("\n" + "="*80)


def plot_trajectory_analysis(
    analysis: CompleteTrajectoryAnalysis,
    save_plots: bool = False,
    output_dir: str = "results/trajectory_analysis"
) -> None:
    """
    Create comprehensive plots for trajectory analysis.
    
    Args:
        analysis: Complete trajectory analysis
        save_plots: Whether to save plots to files
        output_dir: Directory to save plots
    """
    if save_plots:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        base_filename = f"{analysis.asteroid_name}_{analysis.landing_site_name}_{analysis.thrust_config}"
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Thrust profile
    ax1 = plt.subplot(2, 3, 1)
    time = analysis.thrust_analysis.time_vector
    thrust = analysis.thrust_analysis.thrust_magnitude
    
    ax1.plot(time, thrust, 'b-', linewidth=2)
    ax1.axhline(y=analysis.solution.vehicle_params.T_max, color='r', linestyle='--', alpha=0.5, label='T_max')
    ax1.axhline(y=analysis.solution.vehicle_params.T_min, color='g', linestyle='--', alpha=0.5, label='T_min')
    
    # Mark switching times
    for t in analysis.thrust_analysis.switching_times:
        ax1.axvline(x=t, color='k', linestyle=':', alpha=0.3)
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Thrust (N)')
    ax1.set_title('Thrust Profile')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Convergence history
    ax2 = plt.subplot(2, 3, 2)
    iterations = range(1, len(analysis.convergence_analysis.position_error_history) + 1)
    errors = analysis.convergence_analysis.position_error_history
    
    ax2.semilogy(iterations, errors, 'bo-', linewidth=2, markersize=8)
    ax2.axhline(y=0.5, color='r', linestyle='--', label='Tolerance (0.5m)')
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Max Position Error (m)')
    ax2.set_title('Convergence History')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Mass profile
    ax3 = plt.subplot(2, 3, 3)
    mass_profile = analysis.solution.mass_profile
    
    ax3.plot(time[:len(mass_profile)], mass_profile, 'g-', linewidth=2)
    ax3.axhline(y=analysis.solution.vehicle_params.m_dry, color='r', linestyle='--', label='Dry Mass')
    
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Mass (kg)')
    ax3.set_title('Mass Profile')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Altitude profile
    ax4 = plt.subplot(2, 3, 4)
    positions = analysis.solution.position_profile
    landing_site_pos = analysis.solution.experiment_config.landing_site.position
    
    # Compute altitude relative to landing site
    altitudes = np.linalg.norm(positions - landing_site_pos, axis=1)
    
    ax4.plot(time[:len(altitudes)], altitudes, 'm-', linewidth=2)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Altitude (m)')
    ax4.set_title('Altitude Profile')
    ax4.grid(True, alpha=0.3)
    
    # 5. Velocity profile
    ax5 = plt.subplot(2, 3, 5)
    velocities = analysis.solution.velocity_profile
    speed = np.linalg.norm(velocities, axis=1)
    
    ax5.plot(time[:len(speed)], speed, 'c-', linewidth=2)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Speed (m/s)')
    ax5.set_title('Velocity Profile')
    ax5.grid(True, alpha=0.3)
    
    # 6. Brent search history (if available)
    ax6 = plt.subplot(2, 3, 6)
    if analysis.flight_time_result and analysis.flight_time_result.evaluation_history:
        flight_times = [e['flight_time'] for e in analysis.flight_time_result.evaluation_history]
        propellant = [e['propellant_used'] for e in analysis.flight_time_result.evaluation_history]
        
        ax6.plot(flight_times, propellant, 'ko-', linewidth=2, markersize=6)
        ax6.axvline(x=analysis.quality_metrics.flight_time, color='r', linestyle='--', label='Optimum')
        
        ax6.set_xlabel('Flight Time (s)')
        ax6.set_ylabel('Propellant Used (kg)')
        ax6.set_title('Brent Search History')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
    else:
        ax6.text(0.5, 0.5, 'No Brent search data', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax6.transAxes)
        ax6.set_title('Brent Search History')
    
    plt.suptitle(f'Trajectory Analysis: {analysis.asteroid_name} - {analysis.landing_site_name} - {analysis.thrust_config}',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_plots:
        plot_path = Path(output_dir) / f"{base_filename}_analysis.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved analysis plot to: {plot_path}")
    
    plt.show()


def compare_with_paper_results(
    analysis: CompleteTrajectoryAnalysis,
    paper_reference: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare analysis results with paper reference values.
    
    Args:
        analysis: Complete trajectory analysis
        paper_reference: Dictionary with paper reference values
        
    Returns:
        Dictionary with comparison results
    """
    comparison = {
        'scenario': f"{analysis.asteroid_name}_{analysis.landing_site_name}_{analysis.thrust_config}",
        'propellant_match': False,
        'flight_time_match': False,
        'switching_times_match': False,
        'convergence_match': False,
        'overall_match': False
    }
    
    # Propellant comparison
    if 'propellant_used' in paper_reference:
        paper_propellant = paper_reference['propellant_used']
        actual_propellant = analysis.quality_metrics.propellant_used
        error_percent = analysis.quality_metrics.propellant_error_percent
        
        comparison['propellant_paper'] = paper_propellant
        comparison['propellant_actual'] = actual_propellant
        comparison['propellant_error_percent'] = error_percent
        comparison['propellant_match'] = error_percent < 5.0  # 5% tolerance
    
    # Flight time comparison
    if 'optimal_flight_time' in paper_reference:
        paper_time = paper_reference['optimal_flight_time']
        actual_time = analysis.quality_metrics.flight_time
        error_percent = analysis.quality_metrics.flight_time_error_percent
        
        comparison['flight_time_paper'] = paper_time
        comparison['flight_time_actual'] = actual_time
        comparison['flight_time_error_percent'] = error_percent
        comparison['flight_time_match'] = error_percent < 5.0  # 5% tolerance
    
    # Switching times comparison
    if 'switching_times' in paper_reference:
        paper_switching = paper_reference['switching_times']
        actual_switching = analysis.thrust_analysis.switching_times
        
        comparison['switching_times_paper'] = paper_switching
        comparison['switching_times_actual'] = actual_switching
        
        if paper_switching and actual_switching:
            # Compare number of switching times
            comparison['switching_count_match'] = len(paper_switching) == len(actual_switching)
            
            # Compare individual switching times
            max_error = 0.0
            for i, (paper_t, actual_t) in enumerate(zip(paper_switching, actual_switching)):
                error = abs(paper_t - actual_t)
                max_error = max(max_error, error)
            
            comparison['max_switching_time_error'] = max_error
            comparison['switching_times_match'] = max_error < 10.0  # 10s tolerance
    
    # Convergence comparison
    if 'expected_iterations' in paper_reference:
        paper_iterations = paper_reference['expected_iterations']
        actual_iterations = analysis.convergence_analysis.iterations
        
        comparison['iterations_paper'] = paper_iterations
        comparison['iterations_actual'] = actual_iterations
        comparison['iterations_error'] = abs(actual_iterations - paper_iterations)
        comparison['convergence_match'] = comparison['iterations_error'] <= 2  # ±2 iterations
    
    # Overall match
    matches = [
        comparison.get('propellant_match', True),
        comparison.get('flight_time_match', True),
        comparison.get('switching_times_match', True),
        comparison.get('convergence_match', True)
    ]
    comparison['overall_match'] = all(matches)
    
    return comparison


def test_trajectory_analysis() -> bool:
    """
    Test the trajectory analysis module.
    
    Returns:
        True if tests pass, False otherwise
    """
    try:
        print("Testing trajectory analysis module...")
        
        # Create mock data for testing
        mock_thrust = np.array([80.0, 80.0, 20.0, 20.0, 80.0, 80.0])
        mock_time = np.array([0.0, 100.0, 200.0, 300.0, 400.0, 500.0])
        
        # Test ThrustProfileAnalysis
        thrust_analysis = ThrustProfileAnalysis(
            thrust_magnitude=mock_thrust,
            time_vector=mock_time,
            dt=100.0
        )
        
        assert len(thrust_analysis.thrust_magnitude) == 6
        assert thrust_analysis.dt == 100.0
        
        # Test analyze_thrust_profile with mock solution
        class MockVehicleParams:
            T_max = 80.0
            T_min = 20.0
        
        class MockSolution:
            def compute_thrust_profile(self):
                return mock_thrust
            
            discretization_params = type('obj', (object,), {'dt': 100.0})()
            vehicle_params = MockVehicleParams()
        
        analysis = analyze_thrust_profile(
            solution=MockSolution(),
            vehicle_params=MockVehicleParams(),
            time_vector=mock_time
        )
        
        assert analysis.thrust_magnitude.shape == (6,)
        assert analysis.time_vector.shape == (6,)
        
        print("✓ All trajectory analysis tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Trajectory analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run tests if module is executed directly
    success = test_trajectory_analysis()
    if success:
        print("\nTrajectory analysis module is ready for use.")
    else:
        print("\nTrajectory analysis module tests failed.")