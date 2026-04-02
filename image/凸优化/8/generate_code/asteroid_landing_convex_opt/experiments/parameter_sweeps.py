"""
Parameter sweeps for asteroid landing convex optimization.

This module reproduces Figures 5-7 from the paper:
- Figure 5: Propellant vs flight time for triaxial ellipsoid A1
- Figure 6: Propellant vs flight time for triaxial ellipsoid A2  
- Figure 7: Propellant vs flight time for triaxial ellipsoid A3

Also includes comparison with Mars landing (constant gravity) baseline.
"""

import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from pathlib import Path

from ..config import (
    ASTEROID_A1, ASTEROID_A2, ASTEROID_A3,
    FULL_THRUST_VEHICLE, QUARTER_THRUST_VEHICLE,
    ExperimentConfig, create_triaxial_experiment,
    get_asteroid_by_name, get_vehicle_by_thrust,
    G
)
from ..optimization.successive_solver import (
    solve_successive_solution, SuccessiveSolutionParameters
)
from ..optimization.socp_solver import SOCPSolution
from ..dynamics.state_equations import StateVector
from ..utils.metrics import compute_propellant_used

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ParameterSweepResult:
    """Container for parameter sweep results."""
    asteroid_name: str
    vehicle_thrust: str
    flight_times: np.ndarray
    propellant_used: np.ndarray
    solutions: List[Optional[SOCPSolution]]
    computation_times: np.ndarray
    converged: np.ndarray
    sweep_params: Dict[str, Any]
    
    def __post_init__(self):
        """Validate arrays have same length."""
        n_points = len(self.flight_times)
        assert len(self.propellant_used) == n_points
        assert len(self.solutions) == n_points
        assert len(self.computation_times) == n_points
        assert len(self.converged) == n_points
    
    @property
    def optimal_flight_time(self) -> Optional[float]:
        """Find flight time with minimum propellant usage."""
        if len(self.propellant_used) == 0:
            return None
        valid_idx = np.where(self.converged)[0]
        if len(valid_idx) == 0:
            return None
        min_idx = valid_idx[np.argmin(self.propellant_used[valid_idx])]
        return float(self.flight_times[min_idx])
    
    @property
    def min_propellant(self) -> Optional[float]:
        """Minimum propellant usage among converged solutions."""
        if len(self.propellant_used) == 0:
            return None
        valid_propellant = self.propellant_used[self.converged]
        if len(valid_propellant) == 0:
            return None
        return float(np.min(valid_propellant))
    
    @property
    def success_rate(self) -> float:
        """Percentage of converged solutions."""
        if len(self.converged) == 0:
            return 0.0
        return float(np.sum(self.converged) / len(self.converged) * 100)


@dataclass
class MarsBaselineResult:
    """Container for Mars landing baseline results."""
    flight_times: np.ndarray
    propellant_used: np.ndarray
    gravity_acceleration: float  # m/s²
    vehicle_mass: float  # kg
    max_thrust: float  # N
    min_thrust: float  # N
    I_sp: float  # s
    g_0: float  # m/s²


def compute_mars_baseline(
    flight_times: np.ndarray,
    vehicle_params: Any,
    initial_altitude: float = 2000.0,
    final_velocity: float = 0.1,
    mars_gravity: float = 3.711  # m/s² (Mars surface gravity)
) -> MarsBaselineResult:
    """
    Compute propellant usage for Mars landing with constant gravity.
    
    Uses analytical solution for constant gravity, constant mass approximation.
    This provides a baseline for comparison with asteroid results.
    
    Args:
        flight_times: Array of flight times to evaluate (s)
        vehicle_params: Vehicle parameters object
        initial_altitude: Initial altitude above landing site (m)
        final_velocity: Desired final velocity (m/s)
        mars_gravity: Mars surface gravity (m/s²)
        
    Returns:
        MarsBaselineResult with propellant usage for each flight time
    """
    # Extract vehicle parameters
    m_wet = vehicle_params.m_wet
    T_max = vehicle_params.T_max
    T_min = vehicle_params.T_min
    I_sp = vehicle_params.I_sp
    g_0 = vehicle_params.g_0
    
    # For constant gravity, we can compute required acceleration analytically
    # Using constant acceleration approximation (bang-bang control)
    propellant_list = []
    
    for t_f in flight_times:
        if t_f <= 0:
            propellant_list.append(np.inf)
            continue
            
        # Minimum acceleration needed to land from given altitude
        # Solve: h = v0*t + 0.5*a*t² with v0=0.1 m/s (initial velocity)
        # and final velocity vf = 0.1 m/s
        v0 = 0.1  # initial velocity
        vf = final_velocity
        
        # Required acceleration to satisfy position and velocity constraints
        # Using constant acceleration model
        a_req = 2 * (initial_altitude - v0 * t_f) / (t_f ** 2)
        
        # Add gravity compensation
        a_total = a_req + mars_gravity
        
        # Check if acceleration is within thrust bounds
        max_accel = T_max / m_wet
        min_accel = T_min / m_wet
        
        if a_total > max_accel or a_total < min_accel:
            # Not feasible with constant acceleration
            propellant_list.append(np.inf)
            continue
        
        # Compute propellant usage (Tsiolkovsky rocket equation)
        # For constant acceleration, thrust = m * a_total
        # Mass flow rate = thrust / (I_sp * g_0)
        # Integrate over time: Δm = ∫ (m * a_total) / (I_sp * g_0) dt
        # With constant a_total and approximately constant m ≈ m_wet
        delta_m = (m_wet * a_total * t_f) / (I_sp * g_0)
        propellant_list.append(delta_m)
    
    return MarsBaselineResult(
        flight_times=flight_times,
        propellant_used=np.array(propellant_list),
        gravity_acceleration=mars_gravity,
        vehicle_mass=m_wet,
        max_thrust=T_max,
        min_thrust=T_min,
        I_sp=I_sp,
        g_0=g_0
    )


def run_parameter_sweep(
    asteroid_name: str,
    vehicle_thrust: str = "full",
    flight_time_range: Tuple[float, float] = (100.0, 1500.0),
    n_points: int = 20,
    discretization_dt: float = 2.0,
    solver: str = "ECOS",
    use_scaling: bool = True,
    verbose: bool = False
) -> ParameterSweepResult:
    """
    Run parameter sweep over flight times for a given asteroid.
    
    Args:
        asteroid_name: Name of asteroid ("A1", "A2", "A3", or "Castalia")
        vehicle_thrust: "full" or "quarter" thrust configuration
        flight_time_range: (min, max) flight time range in seconds
        n_points: Number of flight times to evaluate
        discretization_dt: Time step for discretization (s)
        solver: SOCP solver to use
        use_scaling: Whether to use numerical scaling
        verbose: Print progress information
        
    Returns:
        ParameterSweepResult with sweep results
    """
    logger.info(f"Starting parameter sweep for {asteroid_name} with {vehicle_thrust} thrust")
    
    # Get asteroid and vehicle parameters
    asteroid = get_asteroid_by_name(asteroid_name)
    vehicle = get_vehicle_by_thrust(vehicle_thrust)
    
    # Create experiment configuration
    experiment_config = create_triaxial_experiment(asteroid, vehicle)
    
    # Generate flight times (logarithmically spaced for better coverage)
    t_min, t_max = flight_time_range
    flight_times = np.logspace(np.log10(t_min), np.log10(t_max), n_points)
    
    # Configure successive solver parameters
    successive_params = SuccessiveSolutionParameters(
        max_iterations=10,
        position_tolerance=0.5,  # 0.5m convergence tolerance
        adaptive_linearization=True,
        verbose=False
    )
    
    # Run sweep
    propellant_list = []
    solutions = []
    computation_times = []
    converged_list = []
    
    for i, t_f in enumerate(flight_times):
        if verbose:
            logger.info(f"  Evaluating t_f = {t_f:.1f}s ({i+1}/{n_points})")
        
        start_time = time.time()
        
        try:
            # Solve for this flight time
            result = solve_successive_solution(
                experiment_config=experiment_config,
                t_f=t_f,
                discretization_dt=discretization_dt,
                solver=solver,
                use_scaling=use_scaling,
                params=successive_params
            )
            
            comp_time = time.time() - start_time
            
            if result.converged:
                propellant = compute_propellant_used(
                    result.solution.initial_mass,
                    result.solution.final_mass
                )
                propellant_list.append(propellant)
                solutions.append(result.solution)
                converged_list.append(True)
            else:
                # Failed to converge
                propellant_list.append(np.inf)
                solutions.append(None)
                converged_list.append(False)
            
            computation_times.append(comp_time)
            
            if verbose:
                status = "converged" if result.converged else "failed"
                logger.info(f"    {status} in {comp_time:.2f}s, propellant = {propellant_list[-1]:.3f}kg")
                
        except Exception as e:
            # Handle any solver errors
            comp_time = time.time() - start_time
            propellant_list.append(np.inf)
            solutions.append(None)
            computation_times.append(comp_time)
            converged_list.append(False)
            
            if verbose:
                logger.warning(f"    Error: {e}")
    
    # Create result object
    sweep_params = {
        "flight_time_range": flight_time_range,
        "n_points": n_points,
        "discretization_dt": discretization_dt,
        "solver": solver,
        "use_scaling": use_scaling
    }
    
    return ParameterSweepResult(
        asteroid_name=asteroid_name,
        vehicle_thrust=vehicle_thrust,
        flight_times=flight_times,
        propellant_used=np.array(propellant_list),
        solutions=solutions,
        computation_times=np.array(computation_times),
        converged=np.array(converged_list),
        sweep_params=sweep_params
    )


def run_triaxial_ellipsoid_sweeps(
    vehicle_thrust: str = "full",
    flight_time_range: Tuple[float, float] = (100.0, 1500.0),
    n_points: int = 20,
    discretization_dt: float = 2.0,
    include_mars_baseline: bool = True,
    save_plots: bool = False,
    output_dir: str = "results/parameter_sweeps"
) -> Dict[str, ParameterSweepResult]:
    """
    Run parameter sweeps for all three triaxial ellipsoids (A1, A2, A3).
    
    Args:
        vehicle_thrust: "full" or "quarter" thrust configuration
        flight_time_range: (min, max) flight time range in seconds
        n_points: Number of flight times to evaluate
        discretization_dt: Time step for discretization (s)
        include_mars_baseline: Whether to compute Mars baseline
        save_plots: Whether to save plots to disk
        output_dir: Directory to save plots and results
        
    Returns:
        Dictionary mapping asteroid names to sweep results
    """
    logger.info("Running triaxial ellipsoid parameter sweeps")
    
    # Create output directory
    if save_plots:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run sweeps for each asteroid
    asteroids = ["A1", "A2", "A3"]
    results = {}
    
    for asteroid_name in asteroids:
        logger.info(f"Processing asteroid {asteroid_name}")
        
        result = run_parameter_sweep(
            asteroid_name=asteroid_name,
            vehicle_thrust=vehicle_thrust,
            flight_time_range=flight_time_range,
            n_points=n_points,
            discretization_dt=discretization_dt,
            verbose=True
        )
        
        results[asteroid_name] = result
        
        # Print summary
        logger.info(f"  Success rate: {result.success_rate:.1f}%")
        logger.info(f"  Optimal flight time: {result.optimal_flight_time:.1f}s")
        logger.info(f"  Minimum propellant: {result.min_propellant:.3f}kg")
        logger.info(f"  Average computation time: {np.mean(result.computation_times):.2f}s")
    
    # Compute Mars baseline if requested
    mars_result = None
    if include_mars_baseline:
        logger.info("Computing Mars landing baseline")
        vehicle = get_vehicle_by_thrust(vehicle_thrust)
        flight_times = np.logspace(
            np.log10(flight_time_range[0]),
            np.log10(flight_time_range[1]),
            n_points
        )
        mars_result = compute_mars_baseline(flight_times, vehicle)
    
    # Create plots
    if save_plots or True:  # Always create plots for now
        plot_triaxial_sweeps(
            results=results,
            mars_result=mars_result,
            vehicle_thrust=vehicle_thrust,
            save_plots=save_plots,
            output_dir=output_dir
        )
    
    return results


def plot_triaxial_sweeps(
    results: Dict[str, ParameterSweepResult],
    mars_result: Optional[MarsBaselineResult] = None,
    vehicle_thrust: str = "full",
    save_plots: bool = False,
    output_dir: str = "results/parameter_sweeps"
):
    """
    Create plots for triaxial ellipsoid parameter sweeps.
    
    Args:
        results: Dictionary of sweep results for A1, A2, A3
        mars_result: Mars baseline results (optional)
        vehicle_thrust: Thrust configuration for plot title
        save_plots: Whether to save plots to disk
        output_dir: Directory to save plots
    """
    # Create figure with subplots (one for each asteroid)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = {'A1': 'blue', 'A2': 'green', 'A3': 'red'}
    markers = {'A1': 'o', 'A2': 's', 'A3': '^'}
    
    for idx, (asteroid_name, result) in enumerate(results.items()):
        ax = axes[idx]
        color = colors[asteroid_name]
        marker = markers[asteroid_name]
        
        # Plot converged points
        converged_mask = result.converged
        finite_mask = np.isfinite(result.propellant_used)
        valid_mask = converged_mask & finite_mask
        
        if np.any(valid_mask):
            ax.plot(
                result.flight_times[valid_mask],
                result.propellant_used[valid_mask],
                marker=marker, color=color, linestyle='-',
                linewidth=2, markersize=6,
                label=f'{asteroid_name} (converged)'
            )
        
        # Plot non-converged points
        non_converged_mask = ~converged_mask
        if np.any(non_converged_mask):
            ax.plot(
                result.flight_times[non_converged_mask],
                result.propellant_used[non_converged_mask],
                marker='x', color='gray', linestyle='none',
                markersize=8, label='Failed'
            )
        
        # Plot Mars baseline if available
        if mars_result is not None:
            ax.plot(
                mars_result.flight_times,
                mars_result.propellant_used,
                color='black', linestyle='--',
                linewidth=2, label='Mars baseline'
            )
        
        # Mark optimal point
        optimal_tf = result.optimal_flight_time
        if optimal_tf is not None:
            # Find index of optimal point
            optimal_idx = np.argmin(np.abs(result.flight_times - optimal_tf))
            if result.converged[optimal_idx]:
                ax.plot(
                    optimal_tf,
                    result.propellant_used[optimal_idx],
                    marker='*', color='gold', markersize=15,
                    markeredgecolor='black', markeredgewidth=1,
                    label=f'Optimal ({optimal_tf:.0f}s)'
                )
        
        # Configure subplot
        ax.set_xlabel('Flight Time (s)', fontsize=12)
        ax.set_ylabel('Propellant Used (kg)', fontsize=12)
        ax.set_title(f'Asteroid {asteroid_name}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Set reasonable axis limits
        if np.any(valid_mask):
            valid_propellant = result.propellant_used[valid_mask]
            if len(valid_propellant) > 0:
                prop_min = np.min(valid_propellant)
                prop_max = np.max(valid_propellant)
                prop_range = prop_max - prop_min
                ax.set_ylim([
                    max(0, prop_min - 0.1 * prop_range),
                    prop_max + 0.1 * prop_range
                ])
    
    # Add overall title
    thrust_label = "Full (80N)" if vehicle_thrust == "full" else "Quarter (20N)"
    fig.suptitle(
        f'Propellant vs Flight Time for Triaxial Ellipsoids\n{thrust_label} Thrust',
        fontsize=16, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    
    if save_plots:
        filename = f"triaxial_sweeps_{vehicle_thrust}_thrust.png"
        filepath = Path(output_dir) / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {filepath}")
    
    plt.show()
    
    # Create convergence statistics plot
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot success rates
    asteroid_names = list(results.keys())
    success_rates = [results[name].success_rate for name in asteroid_names]
    
    bars = ax1.bar(asteroid_names, success_rates, color=[colors[name] for name in asteroid_names])
    ax1.set_xlabel('Asteroid', fontsize=12)
    ax1.set_ylabel('Success Rate (%)', fontsize=12)
    ax1.set_title('Convergence Success Rate', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 105])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Plot computation times
    comp_times = [np.mean(results[name].computation_times) for name in asteroid_names]
    comp_std = [np.std(results[name].computation_times) for name in asteroid_names]
    
    bars2 = ax2.bar(asteroid_names, comp_times, yerr=comp_std,
                   color=[colors[name] for name in asteroid_names],
                   capsize=5, alpha=0.7)
    ax2.set_xlabel('Asteroid', fontsize=12)
    ax2.set_ylabel('Average Computation Time (s)', fontsize=12)
    ax2.set_title('Computation Time per Flight Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, time_val in zip(bars2, comp_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{time_val:.1f}s', ha='center', va='bottom', fontsize=10)
    
    fig2.suptitle(
        f'Algorithm Performance - {thrust_label} Thrust',
        fontsize=16, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    
    if save_plots:
        filename = f"performance_stats_{vehicle_thrust}_thrust.png"
        filepath = Path(output_dir) / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Saved performance plot to {filepath}")
    
    plt.show()


def analyze_unimodality(results: Dict[str, ParameterSweepResult]) -> Dict[str, Dict[str, Any]]:
    """
    Analyze unimodality of propellant vs flight time curves.
    
    Args:
        results: Dictionary of sweep results
        
    Returns:
        Dictionary with unimodality analysis for each asteroid
    """
    analysis = {}
    
    for asteroid_name, result in results.items():
        # Get valid (converged, finite) data points
        valid_mask = result.converged & np.isfinite(result.propellant_used)
        if np.sum(valid_mask) < 3:
            analysis[asteroid_name] = {
                "is_unimodal": False,
                "reason": "Insufficient valid data points",
                "minima_count": 0,
                "optimal_flight_time": None
            }
            continue
        
        valid_times = result.flight_times[valid_mask]
        valid_propellant = result.propellant_used[valid_mask]
        
        # Find local minima
        minima_indices = []
        n = len(valid_propellant)
        
        for i in range(1, n-1):
            if (valid_propellant[i] < valid_propellant[i-1] and 
                valid_propellant[i] < valid_propellant[i+1]):
                minima_indices.append(i)
        
        # Check if unimodal (should have exactly one minimum)
        is_unimodal = len(minima_indices) == 1
        
        analysis[asteroid_name] = {
            "is_unimodal": is_unimodal,
            "minima_count": len(minima_indices),
            "minima_locations": [valid_times[i] for i in minima_indices],
            "minima_values": [valid_propellant[i] for i in minima_indices],
            "optimal_flight_time": result.optimal_flight_time,
            "n_valid_points": np.sum(valid_mask),
            "propellant_range": (np.min(valid_propellant), np.max(valid_propellant))
        }
    
    return analysis


def print_sweep_summary(results: Dict[str, ParameterSweepResult]):
    """
    Print summary of parameter sweep results.
    
    Args:
        results: Dictionary of sweep results
    """
    print("\n" + "="*80)
    print("PARAMETER SWEEP SUMMARY")
    print("="*80)
    
    for asteroid_name, result in results.items():
        print(f"\nAsteroid: {asteroid_name}")
        print(f"  Thrust configuration: {result.vehicle_thrust}")
        print(f"  Flight time range: {result.sweep_params['flight_time_range'][0]:.0f}-"
              f"{result.sweep_params['flight_time_range'][1]:.0f}s")
        print(f"  Number of points: {result.sweep_params['n_points']}")
        print(f"  Success rate: {result.success_rate:.1f}%")
        print(f"  Optimal flight time: {result.optimal_flight_time:.1f}s")
        print(f"  Minimum propellant: {result.min_propellant:.3f}kg")
        print(f"  Average computation time: {np.mean(result.computation_times):.2f}s")
    
    # Analyze unimodality
    print("\n" + "-"*80)
    print("UNIMODALITY ANALYSIS")
    print("-"*80)
    
    unimodality_analysis = analyze_unimodality(results)
    for asteroid_name, analysis in unimodality_analysis.items():
        status = "✓ UNIMODAL" if analysis["is_unimodal"] else "✗ NOT UNIMODAL"
        print(f"\n{asteroid_name}: {status}")
        if analysis["minima_count"] > 0:
            print(f"  Found {analysis['minima_count']} local minima at:")
            for t, m in zip(analysis["minima_locations"], analysis["minima_values"]):
                print(f"    t = {t:.1f}s, m = {m:.3f}kg")
        else:
            print("  No local minima found")
        print(f"  Valid data points: {analysis['n_valid_points']}")
        print(f"  Propellant range: {analysis['propellant_range'][0]:.3f}-"
              f"{analysis['propellant_range'][1]:.3f}kg")
    
    print("\n" + "="*80)


def test_parameter_sweeps():
    """Test function for parameter sweeps module."""
    print("Testing parameter sweeps module...")
    
    try:
        # Test with a small sweep (few points, short times) for speed
        results = run_triaxial_ellipsoid_sweeps(
            vehicle_thrust="full",
            flight_time_range=(200.0, 800.0),
            n_points=5,
            discretization_dt=5.0,  # Larger dt for faster testing
            include_mars_baseline=True,
            save_plots=False
        )
        
        # Check that we got results for all asteroids
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"
        assert "A1" in results
        assert "A2" in results
        assert "A3" in results
        
        # Check result structure
        for asteroid_name, result in results.items():
            assert isinstance(result, ParameterSweepResult)
            assert len(result.flight_times) == 5
            assert len(result.propellant_used) == 5
            assert len(result.converged) == 5
            
            # At least some points should converge
            assert np.sum(result.converged) > 0, f"No converged points for {asteroid_name}"
        
        print("✓ Parameter sweeps test passed")
        return True
        
    except Exception as e:
        print(f"✗ Parameter sweeps test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to run parameter sweeps from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run parameter sweeps for asteroid landing optimization"
    )
    parser.add_argument(
        "--thrust", type=str, default="full",
        choices=["full", "quarter"],
        help="Thrust configuration (full=80N, quarter=20N)"
    )
    parser.add_argument(
        "--t-min", type=float, default=100.0,
        help="Minimum flight time (s)"
    )
    parser.add_argument(
        "--t-max", type=float, default=1500.0,
        help="Maximum flight time (s)"
    )
    parser.add_argument(
        "--n-points", type=int, default=20,
        help="Number of flight times to evaluate"
    )
    parser.add_argument(
        "--dt", type=float, default=2.0,
        help="Discretization time step (s)"
    )
    parser.add_argument(
        "--save-plots", action="store_true",
        help="Save plots to disk"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/parameter_sweeps",
        help="Output directory for plots and results"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run test instead of full sweep"
    )
    
    args = parser.parse_args()
    
    if args.test:
        # Run test
        success = test_parameter_sweeps()
        return 0 if success else 1
    
    # Run full parameter sweeps
    results = run_triaxial_ellipsoid_sweeps(
        vehicle_thrust=args.thrust,
        flight_time_range=(args.t_min, args.t_max),
        n_points=args.n_points,
        discretization_dt=args.dt,
        include_mars_baseline=True,
        save_plots=args.save_plots,
        output_dir=args.output_dir
    )
    
    # Print summary
    print_sweep_summary(results)
    
    return 0


if __name__ == "__main__":
    main()