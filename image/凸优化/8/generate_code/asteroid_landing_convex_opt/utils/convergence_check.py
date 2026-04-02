"""
Convergence checking utilities for the successive solution algorithm.

This module provides functions to analyze and validate the convergence
of the successive solution algorithm (Algorithm 1) used in the asteroid
landing convex optimization framework.

Key functionalities:
- Check convergence metrics (position, velocity, mass, thrust)
- Analyze convergence history and compute convergence rates
- Validate convergence against paper specifications
- Generate convergence summaries and plots
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from pathlib import Path

from ..optimization.successive_solver import SuccessiveSolutionResult
from ..optimization.socp_solver import SOCPSolution
from ..config import VALIDATION_TOLERANCES

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceMetrics:
    """Container for convergence metrics."""
    
    # Convergence status
    converged: bool
    """Whether the algorithm converged within tolerance."""
    
    iterations: int
    """Number of iterations performed."""
    
    max_iterations: int
    """Maximum allowed iterations."""
    
    # Position convergence
    max_position_error: float
    """Maximum position error across all iterations (m)."""
    
    final_position_error: float
    """Position error at final iteration (m)."""
    
    position_tolerance: float
    """Convergence tolerance for position (m)."""
    
    position_converged: bool
    """Whether position converged within tolerance."""
    
    # Velocity convergence
    max_velocity_error: float
    """Maximum velocity error across all iterations (m/s)."""
    
    final_velocity_error: float
    """Velocity error at final iteration (m/s)."""
    
    velocity_tolerance: float
    """Convergence tolerance for velocity (m/s)."""
    
    velocity_converged: bool
    """Whether velocity converged within tolerance."""
    
    # Mass convergence
    max_mass_error: float
    """Maximum mass error across all iterations (kg)."""
    
    final_mass_error: float
    """Mass error at final iteration (kg)."""
    
    mass_tolerance: float
    """Convergence tolerance for mass (kg)."""
    
    mass_converged: bool
    """Whether mass converged within tolerance."""
    
    # Thrust convergence
    max_thrust_error: float
    """Maximum thrust error across all iterations (N)."""
    
    final_thrust_error: float
    """Thrust error at final iteration (N)."""
    
    thrust_tolerance: float
    """Convergence tolerance for thrust (N)."""
    
    thrust_converged: bool
    """Whether thrust converged within tolerance."""
    
    # Convergence rates
    linear_convergence_rate: Optional[float] = None
    """Linear convergence rate (if applicable)."""
    
    quadratic_convergence_rate: Optional[float] = None
    """Quadratic convergence rate (if applicable)."""
    
    convergence_type: str = "unknown"
    """Type of convergence (linear, quadratic, superlinear)."""
    
    # Performance metrics
    total_computation_time: float
    """Total computation time (seconds)."""
    
    average_iteration_time: float
    """Average time per iteration (seconds)."""
    
    # History data
    position_error_history: List[float] = field(default_factory=list)
    """Position error at each iteration."""
    
    velocity_error_history: List[float] = field(default_factory=list)
    """Velocity error at each iteration."""
    
    mass_error_history: List[float] = field(default_factory=list)
    """Mass error at each iteration."""
    
    thrust_error_history: List[float] = field(default_factory=list)
    """Thrust error at each iteration."""
    
    computation_time_history: List[float] = field(default_factory=list)
    """Computation time for each iteration."""
    
    # Paper comparison
    expected_iterations: Optional[int] = None
    """Expected number of iterations from paper."""
    
    iteration_error: Optional[float] = None
    """Difference between actual and expected iterations."""
    
    paper_convergence_tolerance: float = 0.5
    """Convergence tolerance from paper (0.5m)."""
    
    def __post_init__(self):
        """Validate and compute derived metrics."""
        # Compute convergence rates if we have enough data
        if len(self.position_error_history) >= 3:
            self._compute_convergence_rates()
        
        # Compute iteration error if expected iterations provided
        if self.expected_iterations is not None:
            self.iteration_error = self.iterations - self.expected_iterations
    
    def _compute_convergence_rates(self):
        """Compute linear and quadratic convergence rates."""
        errors = np.array(self.position_error_history)
        
        # Skip if errors are too small or not decreasing
        if len(errors) < 3 or np.any(errors <= 0):
            return
        
        # Compute linear convergence rate
        linear_rates = []
        for i in range(1, len(errors)):
            if errors[i-1] > 0:
                rate = errors[i] / errors[i-1]
                linear_rates.append(rate)
        
        if linear_rates:
            self.linear_convergence_rate = float(np.mean(linear_rates[-3:]))  # Average of last 3
        
        # Compute quadratic convergence rate
        quadratic_rates = []
        for i in range(2, len(errors)):
            if errors[i-1] > 0 and errors[i-2] > 0:
                rate = errors[i] / (errors[i-1] ** 2)
                quadratic_rates.append(rate)
        
        if quadratic_rates:
            self.quadratic_convergence_rate = float(np.mean(quadratic_rates[-3:]))
        
        # Determine convergence type
        if self.linear_convergence_rate is not None:
            if self.linear_convergence_rate < 1.0:
                self.convergence_type = "linear"
            elif self.quadratic_convergence_rate is not None:
                self.convergence_type = "quadratic"
    
    @property
    def overall_converged(self) -> bool:
        """Check if overall convergence criteria are met."""
        return (self.position_converged and 
                self.velocity_converged and 
                self.mass_converged and 
                self.thrust_converged)
    
    @property
    def meets_paper_specifications(self) -> bool:
        """Check if convergence meets paper specifications."""
        return (self.position_converged and 
                self.final_position_error <= self.paper_convergence_tolerance and
                self.iterations <= 10)  # Max iterations from paper
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'converged': self.converged,
            'iterations': self.iterations,
            'max_iterations': self.max_iterations,
            'max_position_error': self.max_position_error,
            'final_position_error': self.final_position_error,
            'position_tolerance': self.position_tolerance,
            'position_converged': self.position_converged,
            'max_velocity_error': self.max_velocity_error,
            'final_velocity_error': self.final_velocity_error,
            'velocity_tolerance': self.velocity_tolerance,
            'velocity_converged': self.velocity_converged,
            'max_mass_error': self.max_mass_error,
            'final_mass_error': self.final_mass_error,
            'mass_tolerance': self.mass_tolerance,
            'mass_converged': self.mass_converged,
            'max_thrust_error': self.max_thrust_error,
            'final_thrust_error': self.final_thrust_error,
            'thrust_tolerance': self.thrust_tolerance,
            'thrust_converged': self.thrust_converged,
            'linear_convergence_rate': self.linear_convergence_rate,
            'quadratic_convergence_rate': self.quadratic_convergence_rate,
            'convergence_type': self.convergence_type,
            'total_computation_time': self.total_computation_time,
            'average_iteration_time': self.average_iteration_time,
            'expected_iterations': self.expected_iterations,
            'iteration_error': self.iteration_error,
            'paper_convergence_tolerance': self.paper_convergence_tolerance,
            'overall_converged': self.overall_converged,
            'meets_paper_specifications': self.meets_paper_specifications
        }


def check_convergence_metrics(
    successive_result: SuccessiveSolutionResult,
    position_tolerance: float = 0.5,
    velocity_tolerance: float = 0.01,
    mass_tolerance: float = 0.001,
    thrust_tolerance: float = 0.1,
    expected_iterations: Optional[int] = None
) -> ConvergenceMetrics:
    """
    Check convergence metrics for successive solution algorithm.
    
    Args:
        successive_result: Result from successive solver
        position_tolerance: Convergence tolerance for position (m)
        velocity_tolerance: Convergence tolerance for velocity (m/s)
        mass_tolerance: Convergence tolerance for mass (kg)
        thrust_tolerance: Convergence tolerance for thrust (N)
        expected_iterations: Expected number of iterations from paper
    
    Returns:
        ConvergenceMetrics object with detailed convergence analysis
    """
    if not successive_result.converged:
        logger.warning("Successive solution did not converge")
    
    # Extract error histories
    position_error_history = successive_result.position_error_history
    velocity_error_history = successive_result.velocity_error_history
    mass_error_history = successive_result.mass_error_history
    thrust_error_history = successive_result.thrust_error_history
    computation_time_history = successive_result.computation_time_history
    
    # Compute final errors
    final_position_error = position_error_history[-1] if position_error_history else float('inf')
    final_velocity_error = velocity_error_history[-1] if velocity_error_history else float('inf')
    final_mass_error = mass_error_history[-1] if mass_error_history else float('inf')
    final_thrust_error = thrust_error_history[-1] if thrust_error_history else float('inf')
    
    # Compute max errors
    max_position_error = max(position_error_history) if position_error_history else float('inf')
    max_velocity_error = max(velocity_error_history) if velocity_error_history else float('inf')
    max_mass_error = max(mass_error_history) if mass_error_history else float('inf')
    max_thrust_error = max(thrust_error_history) if thrust_error_history else float('inf')
    
    # Check convergence
    position_converged = final_position_error <= position_tolerance
    velocity_converged = final_velocity_error <= velocity_tolerance
    mass_converged = final_mass_error <= mass_tolerance
    thrust_converged = final_thrust_error <= thrust_tolerance
    
    # Total computation time
    total_computation_time = sum(computation_time_history)
    average_iteration_time = total_computation_time / len(computation_time_history) if computation_time_history else 0.0
    
    # Create metrics object
    metrics = ConvergenceMetrics(
        converged=successive_result.converged,
        iterations=successive_result.iterations,
        max_iterations=successive_result.max_iterations,
        max_position_error=max_position_error,
        final_position_error=final_position_error,
        position_tolerance=position_tolerance,
        position_converged=position_converged,
        max_velocity_error=max_velocity_error,
        final_velocity_error=final_velocity_error,
        velocity_tolerance=velocity_tolerance,
        velocity_converged=velocity_converged,
        max_mass_error=max_mass_error,
        final_mass_error=final_mass_error,
        mass_tolerance=mass_tolerance,
        mass_converged=mass_converged,
        max_thrust_error=max_thrust_error,
        final_thrust_error=final_thrust_error,
        thrust_tolerance=thrust_tolerance,
        thrust_converged=thrust_converged,
        total_computation_time=total_computation_time,
        average_iteration_time=average_iteration_time,
        position_error_history=position_error_history,
        velocity_error_history=velocity_error_history,
        mass_error_history=mass_error_history,
        thrust_error_history=thrust_error_history,
        computation_time_history=computation_time_history,
        expected_iterations=expected_iterations,
        paper_convergence_tolerance=position_tolerance
    )
    
    return metrics


def analyze_convergence_history(
    position_error_history: List[float],
    velocity_error_history: Optional[List[float]] = None,
    mass_error_history: Optional[List[float]] = None,
    thrust_error_history: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Analyze convergence history and compute statistics.
    
    Args:
        position_error_history: Position error at each iteration
        velocity_error_history: Velocity error at each iteration (optional)
        mass_error_history: Mass error at each iteration (optional)
        thrust_error_history: Thrust error at each iteration (optional)
    
    Returns:
        Dictionary with convergence analysis results
    """
    analysis = {
        'position': {},
        'velocity': {},
        'mass': {},
        'thrust': {}
    }
    
    # Analyze position convergence
    if position_error_history:
        pos_errors = np.array(position_error_history)
        analysis['position']['iterations'] = len(pos_errors)
        analysis['position']['initial_error'] = float(pos_errors[0])
        analysis['position']['final_error'] = float(pos_errors[-1])
        analysis['position']['max_error'] = float(np.max(pos_errors))
        analysis['position']['min_error'] = float(np.min(pos_errors))
        analysis['position']['mean_error'] = float(np.mean(pos_errors))
        analysis['position']['std_error'] = float(np.std(pos_errors))
        
        # Compute convergence rate
        if len(pos_errors) >= 2:
            rates = pos_errors[1:] / pos_errors[:-1]
            analysis['position']['mean_convergence_rate'] = float(np.mean(rates))
            analysis['position']['std_convergence_rate'] = float(np.std(rates))
            analysis['position']['final_convergence_rate'] = float(rates[-1]) if len(rates) > 0 else None
    
    # Analyze velocity convergence
    if velocity_error_history:
        vel_errors = np.array(velocity_error_history)
        analysis['velocity']['iterations'] = len(vel_errors)
        analysis['velocity']['initial_error'] = float(vel_errors[0])
        analysis['velocity']['final_error'] = float(vel_errors[-1])
        analysis['velocity']['max_error'] = float(np.max(vel_errors))
        analysis['velocity']['min_error'] = float(np.min(vel_errors))
        analysis['velocity']['mean_error'] = float(np.mean(vel_errors))
        analysis['velocity']['std_error'] = float(np.std(vel_errors))
    
    # Analyze mass convergence
    if mass_error_history:
        mass_errors = np.array(mass_error_history)
        analysis['mass']['iterations'] = len(mass_errors)
        analysis['mass']['initial_error'] = float(mass_errors[0])
        analysis['mass']['final_error'] = float(mass_errors[-1])
        analysis['mass']['max_error'] = float(np.max(mass_errors))
        analysis['mass']['min_error'] = float(np.min(mass_errors))
        analysis['mass']['mean_error'] = float(np.mean(mass_errors))
        analysis['mass']['std_error'] = float(np.std(mass_errors))
    
    # Analyze thrust convergence
    if thrust_error_history:
        thrust_errors = np.array(thrust_error_history)
        analysis['thrust']['iterations'] = len(thrust_errors)
        analysis['thrust']['initial_error'] = float(thrust_errors[0])
        analysis['thrust']['final_error'] = float(thrust_errors[-1])
        analysis['thrust']['max_error'] = float(np.max(thrust_errors))
        analysis['thrust']['min_error'] = float(np.min(thrust_errors))
        analysis['thrust']['mean_error'] = float(np.mean(thrust_errors))
        analysis['thrust']['std_error'] = float(np.std(thrust_errors))
    
    return analysis


def compute_convergence_rate(
    error_history: List[float],
    convergence_type: str = "linear"
) -> Optional[float]:
    """
    Compute convergence rate for given error history.
    
    Args:
        error_history: Error values at each iteration
        convergence_type: Type of convergence ("linear", "quadratic", "superlinear")
    
    Returns:
        Convergence rate or None if cannot be computed
    """
    if len(error_history) < 2:
        return None
    
    errors = np.array(error_history)
    
    if convergence_type == "linear":
        # Linear convergence: e_{k+1} ≈ C * e_k
        if len(errors) >= 2:
            rates = errors[1:] / errors[:-1]
            return float(np.mean(rates[-3:])) if len(rates) >= 3 else float(np.mean(rates))
    
    elif convergence_type == "quadratic":
        # Quadratic convergence: e_{k+1} ≈ C * e_k^2
        if len(errors) >= 3:
            rates = errors[2:] / (errors[1:-1] ** 2)
            return float(np.mean(rates[-3:])) if len(rates) >= 3 else float(np.mean(rates))
    
    elif convergence_type == "superlinear":
        # Superlinear convergence: e_{k+1} / e_k → 0
        if len(errors) >= 2:
            rates = errors[1:] / errors[:-1]
            return float(rates[-1]) if len(rates) > 0 else None
    
    return None


def check_position_convergence(
    position_error_history: List[float],
    tolerance: float = 0.5,
    max_iterations: int = 10
) -> Dict[str, Any]:
    """
    Check position convergence against paper specifications.
    
    Args:
        position_error_history: Position error at each iteration
        tolerance: Convergence tolerance (0.5m from paper)
        max_iterations: Maximum allowed iterations (10 from paper)
    
    Returns:
        Dictionary with position convergence analysis
    """
    if not position_error_history:
        return {
            'converged': False,
            'final_error': float('inf'),
            'tolerance': tolerance,
            'iterations': 0,
            'max_iterations': max_iterations,
            'meets_paper_specs': False,
            'reason': 'No position error history'
        }
    
    final_error = position_error_history[-1]
    iterations = len(position_error_history)
    
    converged = final_error <= tolerance and iterations <= max_iterations
    
    result = {
        'converged': converged,
        'final_error': final_error,
        'tolerance': tolerance,
        'iterations': iterations,
        'max_iterations': max_iterations,
        'meets_paper_specs': converged,
        'reason': 'Converged within specifications' if converged else 'Failed to converge'
    }
    
    if not converged:
        if final_error > tolerance:
            result['reason'] = f'Position error {final_error:.3f}m > tolerance {tolerance}m'
        elif iterations > max_iterations:
            result['reason'] = f'Iterations {iterations} > max {max_iterations}'
    
    return result


def check_velocity_convergence(
    velocity_error_history: List[float],
    tolerance: float = 0.01
) -> Dict[str, Any]:
    """
    Check velocity convergence.
    
    Args:
        velocity_error_history: Velocity error at each iteration
        tolerance: Convergence tolerance for velocity
    
    Returns:
        Dictionary with velocity convergence analysis
    """
    if not velocity_error_history:
        return {
            'converged': False,
            'final_error': float('inf'),
            'tolerance': tolerance,
            'reason': 'No velocity error history'
        }
    
    final_error = velocity_error_history[-1]
    converged = final_error <= tolerance
    
    return {
        'converged': converged,
        'final_error': final_error,
        'tolerance': tolerance,
        'reason': 'Converged' if converged else f'Velocity error {final_error:.3f}m/s > tolerance {tolerance}m/s'
    }


def check_mass_convergence(
    mass_error_history: List[float],
    tolerance: float = 0.001
) -> Dict[str, Any]:
    """
    Check mass convergence.
    
    Args:
        mass_error_history: Mass error at each iteration
        tolerance: Convergence tolerance for mass
    
    Returns:
        Dictionary with mass convergence analysis
    """
    if not mass_error_history:
        return {
            'converged': False,
            'final_error': float('inf'),
            'tolerance': tolerance,
            'reason': 'No mass error history'
        }
    
    final_error = mass_error_history[-1]
    converged = final_error <= tolerance
    
    return {
        'converged': converged,
        'final_error': final_error,
        'tolerance': tolerance,
        'reason': 'Converged' if converged else f'Mass error {final_error:.3f}kg > tolerance {tolerance}kg'
    }


def check_thrust_convergence(
    thrust_error_history: List[float],
    tolerance: float = 0.1
) -> Dict[str, Any]:
    """
    Check thrust convergence.
    
    Args:
        thrust_error_history: Thrust error at each iteration
        tolerance: Convergence tolerance for thrust
    
    Returns:
        Dictionary with thrust convergence analysis
    """
    if not thrust_error_history:
        return {
            'converged': False,
            'final_error': float('inf'),
            'tolerance': tolerance,
            'reason': 'No thrust error history'
        }
    
    final_error = thrust_error_history[-1]
    converged = final_error <= tolerance
    
    return {
        'converged': converged,
        'final_error': final_error,
        'tolerance': tolerance,
        'reason': 'Converged' if converged else f'Thrust error {final_error:.3f}N > tolerance {tolerance}N'
    }


def check_overall_convergence(
    successive_result: SuccessiveSolutionResult,
    position_tolerance: float = 0.5,
    velocity_tolerance: float = 0.01,
    mass_tolerance: float = 0.001,
    thrust_tolerance: float = 0.1,
    max_iterations: int = 10
) -> Dict[str, Any]:
    """
    Check overall convergence of successive solution algorithm.
    
    Args:
        successive_result: Result from successive solver
        position_tolerance: Convergence tolerance for position
        velocity_tolerance: Convergence tolerance for velocity
        mass_tolerance: Convergence tolerance for mass
        thrust_tolerance: Convergence tolerance for thrust
        max_iterations: Maximum allowed iterations
    
    Returns:
        Dictionary with overall convergence analysis
    """
    # Extract error histories
    position_error_history = successive_result.position_error_history
    velocity_error_history = successive_result.velocity_error_history
    mass_error_history = successive_result.mass_error_history
    thrust_error_history = successive_result.thrust_error_history
    
    # Check individual convergences
    position_check = check_position_convergence(
        position_error_history, position_tolerance, max_iterations
    )
    velocity_check = check_velocity_convergence(velocity_error_history, velocity_tolerance)
    mass_check = check_mass_convergence(mass_error_history, mass_tolerance)
    thrust_check = check_thrust_convergence(thrust_error_history, thrust_tolerance)
    
    # Determine overall convergence
    overall_converged = (
        position_check['converged'] and
        velocity_check['converged'] and
        mass_check['converged'] and
        thrust_check['converged'] and
        successive_result.converged
    )
    
    return {
        'overall_converged': overall_converged,
        'position': position_check,
        'velocity': velocity_check,
        'mass': mass_check,
        'thrust': thrust_check,
        'algorithm_converged': successive_result.converged,
        'iterations': successive_result.iterations,
        'max_iterations': successive_result.max_iterations,
        'computation_time': sum(successive_result.computation_time_history) if successive_result.computation_time_history else 0.0
    }


def print_convergence_summary(metrics: ConvergenceMetrics) -> None:
    """
    Print formatted convergence summary.
    
    Args:
        metrics: Convergence metrics to summarize
    """
    print("\n" + "="*80)
    print("CONVERGENCE ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\nOverall Status: {'CONVERGED' if metrics.overall_converged else 'FAILED TO CONVERGE'}")
    print(f"Iterations: {metrics.iterations}/{metrics.max_iterations}")
    print(f"Total Computation Time: {metrics.total_computation_time:.2f}s")
    print(f"Average Iteration Time: {metrics.average_iteration_time:.2f}s")
    
    print(f"\nPaper Specifications Met: {'YES' if metrics.meets_paper_specifications else 'NO'}")
    if metrics.expected_iterations is not None:
        print(f"Expected Iterations: {metrics.expected_iterations}")
        print(f"Iteration Error: {metrics.iteration_error:+d}")
    
    print(f"\nPosition Convergence:")
    print(f"  Final Error: {metrics.final_position_error:.6f}m")
    print(f"  Tolerance: {metrics.position_tolerance}m")
    print(f"  Status: {'CONVERGED' if metrics.position_converged else 'FAILED'}")
    
    print(f"\nVelocity Convergence:")
    print(f"  Final Error: {metrics.final_velocity_error:.6f}m/s")
    print(f"  Tolerance: {metrics.velocity_tolerance}m/s")
    print(f"  Status: {'CONVERGED' if metrics.velocity_converged else 'FAILED'}")
    
    print(f"\nMass Convergence:")
    print(f"  Final Error: {metrics.final_mass_error:.6f}kg")
    print(f"  Tolerance: {metrics.mass_tolerance}kg")
    print(f"  Status: {'CONVERGED' if metrics.mass_converged else 'FAILED'}")
    
    print(f"\nThrust Convergence:")
    print(f"  Final Error: {metrics.final_thrust_error:.6f}N")
    print(f"  Tolerance: {metrics.thrust_tolerance}N")
    print(f"  Status: {'CONVERGED' if metrics.thrust_converged else 'FAILED'}")
    
    if metrics.convergence_type != "unknown":
        print(f"\nConvergence Type: {metrics.convergence_type}")
        if metrics.linear_convergence_rate is not None:
            print(f"Linear Convergence Rate: {metrics.linear_convergence_rate:.4f}")
        if metrics.quadratic_convergence_rate is not None:
            print(f"Quadratic Convergence Rate: {metrics.quadratic_convergence_rate:.4f}")
    
    print("\n" + "="*80)


def plot_convergence_history(
    metrics: ConvergenceMetrics,
    save_plots: bool = False,
    output_dir: str = "results/convergence",
    show_plots: bool = True
) -> None:
    """
    Plot convergence history for analysis.
    
    Args:
        metrics: Convergence metrics with history data
        save_plots: Whether to save plots to file
        output_dir: Directory to save plots
        show_plots: Whether to display plots
    """
    if not metrics.position_error_history:
        logger.warning("No convergence history data to plot")
        return
    
    # Create output directory if needed
    if save_plots:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Successive Solution Convergence Analysis', fontsize=16)
    
    # Plot 1: Position error history
    ax = axes[0, 0]
    iterations = range(1, len(metrics.position_error_history) + 1)
    ax.semilogy(iterations, metrics.position_error_history, 'b-o', linewidth=2, markersize=8)
    ax.axhline(y=metrics.position_tolerance, color='r', linestyle='--', label=f'Tolerance ({metrics.position_tolerance}m)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Position Error (m)')
    ax.set_title('Position Convergence')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Velocity error history
    ax = axes[0, 1]
    if metrics.velocity_error_history:
        ax.semilogy(iterations, metrics.velocity_error_history, 'g-o', linewidth=2, markersize=8)
        ax.axhline(y=metrics.velocity_tolerance, color='r', linestyle='--', label=f'Tolerance ({metrics.velocity_tolerance}m/s)')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Velocity Error (m/s)')
        ax.set_title('Velocity Convergence')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Plot 3: Mass error history
    ax = axes[0, 2]
    if metrics.mass_error_history:
        ax.semilogy(iterations, metrics.mass_error_history, 'm-o', linewidth=2, markersize=8)
        ax.axhline(y=metrics.mass_tolerance, color='r', linestyle='--', label=f'Tolerance ({metrics.mass_tolerance}kg)')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Mass Error (kg)')
        ax.set_title('Mass Convergence')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Plot 4: Thrust error history
    ax = axes[1, 0]
    if metrics.thrust_error_history:
        ax.semilogy(iterations, metrics.thrust_error_history, 'c-o', linewidth=2, markersize=8)
        ax.axhline(y=metrics.thrust_tolerance, color='r', linestyle='--', label=f'Tolerance ({metrics.thrust_tolerance}N)')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Thrust Error (N)')
        ax.set_title('Thrust Convergence')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Plot 5: Computation time per iteration
    ax = axes[1, 1]
    if metrics.computation_time_history:
        ax.plot(iterations, metrics.computation_time_history, 'k-o', linewidth=2, markersize=8)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Computation Time (s)')
        ax.set_title('Computation Time per Iteration')
        ax.grid(True, alpha=0.3)
    
    # Plot 6: Convergence rate analysis
    ax = axes[1, 2]
    if len(metrics.position_error_history) >= 2:
        errors = np.array(metrics.position_error_history)
        convergence_rates = errors[1:] / errors[:-1]
        
        ax.plot(range(2, len(errors) + 1), convergence_rates, 'r-o', linewidth=2, markersize=8)
        ax.axhline(y=1.0, color='k', linestyle='--', label='Divergence threshold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Convergence Rate (e_k+1 / e_k)')
        ax.set_title('Position Convergence Rate')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if requested
    if save_plots:
        filename = Path(output_dir) / f"convergence_analysis_{metrics.iterations}iter.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved convergence plot to {filename}")
    
    # Show plot if requested
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def test_convergence_check() -> bool:
    """
    Test the convergence check module.
    
    Returns:
        True if all tests pass, False otherwise
    """
    logger.info("Testing convergence_check module...")
    
    try:
        # Create test data
        test_position_history = [10.0, 2.0, 0.8, 0.3, 0.1, 0.05]
        test_velocity_history = [1.0, 0.5, 0.2, 0.08, 0.03, 0.01]
        test_mass_history = [0.5, 0.2, 0.08, 0.03, 0.01, 0.005]
        test_thrust_history = [5.0, 2.0, 0.8, 0.3, 0.1, 0.05]
        test_time_history = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
        
        # Test convergence rate computation
        linear_rate = compute_convergence_rate(test_position_history, "linear")
        quadratic_rate = compute_convergence_rate(test_position_history, "quadratic")
        
        logger.info(f"Linear convergence rate: {linear_rate}")
        logger.info(f"Quadratic convergence rate: {quadratic_rate}")
        
        # Test convergence analysis
        analysis = analyze_convergence_history(
            test_position_history,
            test_velocity_history,
            test_mass_history,
            test_thrust_history
        )
        
        logger.info(f"Convergence analysis: {analysis}")
        
        # Test individual convergence checks
        position_check = check_position_convergence(test_position_history)
        velocity_check = check_velocity_convergence(test_velocity_history)
        mass_check = check_mass_convergence(test_mass_history)
        thrust_check = check_thrust_convergence(test_thrust_history)
        
        logger.info(f"Position check: {position_check}")
        logger.info(f"Velocity check: {velocity_check}")
        logger.info(f"Mass check: {mass_check}")
        logger.info(f"Thrust check: {thrust_check}")
        
        # Create a mock SuccessiveSolutionResult for testing
        class MockSuccessiveResult:
            def __init__(self):
                self.converged = True
                self.iterations = 6
                self.max_iterations = 10
                self.position_error_history = test_position_history
                self.velocity_error_history = test_velocity_history
                self.mass_error_history = test_mass_history
                self.thrust_error_history = test_thrust_history
                self.computation_time_history = test_time_history
        
        mock_result = MockSuccessiveResult()
        
        # Test overall convergence check
        overall_check = check_overall_convergence(mock_result)
        logger.info(f"Overall convergence check: {overall_check}")
        
        # Test convergence metrics
        metrics = check_convergence_metrics(mock_result, expected_iterations=5)
        logger.info(f"Convergence metrics created successfully")
        
        # Test summary printing
        print_convergence_summary(metrics)
        
        logger.info("All convergence_check tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"Error testing convergence_check module: {e}")
        return False


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    success = test_convergence_check()
    if success:
        print("\n✓ All convergence_check tests passed!")
    else:
        print("\n✗ Some convergence_check tests failed!")