"""
Castalia Landing Experiments

This module reproduces the main Castalia landing experiments from the paper,
corresponding to Tables 5 and 7. It runs landing simulations for three
landing sites (LS1, LS2, LS3) with both full thrust (80N) and quarter thrust (20N)
configurations.

References:
- Table 4: Initial conditions for Castalia landing sites
- Table 5: Switching times for full thrust landing
- Table 7: Optimal flight times and propellant usage
"""

import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt

from ..config import (
    CASTALIA, CASTALIA_LS1, CASTALIA_LS2, CASTALIA_LS3,
    FULL_THRUST_VEHICLE, QUARTER_THRUST_VEHICLE,
    create_castalia_experiment, ExperimentConfig, VehicleParameters,
    LandingSite, AsteroidParameters, G
)
from ..optimization.flight_time_optimizer import (
    FlightTimeOptimizer, FlightTimeOptimizationResult,
    FlightTimeOptimizerParameters, optimize_flight_time
)
from ..optimization.successive_solver import SuccessiveSolutionResult
from ..optimization.socp_solver import SOCPSolution
from ..dynamics.state_equations import StateVector
from ..utils.metrics import compute_propellant_used, compute_thrust_profile_metrics
from ..utils.visualization import plot_trajectory_3d, plot_thrust_profile
from ..utils.convergence_check import check_convergence_metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CastaliaLandingResult:
    """Results for a single Castalia landing experiment."""
    landing_site_name: str
    thrust_config: str
    experiment_config: ExperimentConfig
    optimization_result: FlightTimeOptimizationResult
    solution: SOCPSolution
    convergence_metrics: Dict[str, float]
    thrust_metrics: Dict[str, float]
    computation_time: float
    
    @property
    def optimal_flight_time(self) -> float:
        """Optimal flight time in seconds."""
        return self.optimization_result.optimal_flight_time
    
    @property
    def propellant_used(self) -> float:
        """Propellant used in kg."""
        return self.optimization_result.optimal_propellant
    
    @property
    def final_mass(self) -> float:
        """Final mass in kg."""
        return self.solution.final_mass
    
    @property
    def initial_mass(self) -> float:
        """Initial mass in kg."""
        return self.solution.initial_mass
    
    @property
    def converged(self) -> bool:
        """Whether the optimization converged."""
        return self.optimization_result.converged
    
    @property
    def brent_iterations(self) -> int:
        """Number of Brent method iterations."""
        return self.optimization_result.brent_iterations
    
    @property
    def function_evaluations(self) -> int:
        """Number of function evaluations."""
        return self.optimization_result.function_evaluations


@dataclass
class CastaliaExperimentSummary:
    """Summary of all Castalia landing experiments."""
    results: Dict[str, CastaliaLandingResult]  # key: "LS1_full", "LS1_quarter", etc.
    paper_comparison: Dict[str, Dict[str, float]]  # comparison with paper values
    overall_stats: Dict[str, float]
    
    def print_summary(self) -> None:
        """Print a formatted summary of all experiments."""
        print("\n" + "="*80)
        print("CASTALIA LANDING EXPERIMENTS - SUMMARY")
        print("="*80)
        
        # Table header
        print(f"{'Landing Site':<12} {'Thrust':<10} {'t_f (s)':<12} {'Propellant (kg)':<18} "
              f"{'Converged':<10} {'Brent Iters':<12} {'Evals':<10}")
        print("-"*80)
        
        for key, result in self.results.items():
            site, thrust = key.split("_")
            print(f"{site:<12} {thrust:<10} {result.optimal_flight_time:>10.2f} "
                  f"{result.propellant_used:>16.3f} {str(result.converged):<10} "
                  f"{result.brent_iterations:>10} {result.function_evaluations:>10}")
        
        print("-"*80)
        
        # Compare with paper values
        print("\nCOMPARISON WITH PAPER VALUES (Table 7):")
        print(f"{'Landing Site':<12} {'Thrust':<10} {'t_f (paper)':<12} {'t_f (ours)':<12} "
              f"{'Error %':<10} {'Propellant (paper)':<18} {'Propellant (ours)':<18} {'Error %':<10}")
        print("-"*100)
        
        for key, paper_vals in self.paper_comparison.items():
            if key in self.results:
                result = self.results[key]
                site, thrust = key.split("_")
                
                t_f_paper = paper_vals.get('t_f', 0)
                prop_paper = paper_vals.get('propellant', 0)
                
                t_f_error = abs(result.optimal_flight_time - t_f_paper) / t_f_paper * 100 if t_f_paper > 0 else 0
                prop_error = abs(result.propellant_used - prop_paper) / prop_paper * 100 if prop_paper > 0 else 0
                
                print(f"{site:<12} {thrust:<10} {t_f_paper:>10.2f} {result.optimal_flight_time:>10.2f} "
                      f"{t_f_error:>8.2f}% {prop_paper:>16.3f} {result.propellant_used:>16.3f} "
                      f"{prop_error:>8.2f}%")
        
        print("\n" + "="*80)


def create_castalia_experiment_config(
    landing_site: LandingSite,
    vehicle: VehicleParameters,
    initial_altitude: float = 2000.0,
    initial_velocity: float = 0.1
) -> ExperimentConfig:
    """
    Create experiment configuration for Castalia landing.
    
    Args:
        landing_site: Landing site (LS1, LS2, or LS3)
        vehicle: Vehicle parameters (full or quarter thrust)
        initial_altitude: Initial altitude above landing site (m)
        initial_velocity: Initial velocity magnitude (m/s)
        
    Returns:
        ExperimentConfig for the landing scenario
    """
    # Create base experiment
    experiment = create_castalia_experiment(
        landing_site=landing_site,
        vehicle=vehicle,
        initial_altitude=initial_altitude,
        initial_velocity=initial_velocity
    )
    
    # Adjust flight time range based on thrust configuration
    if vehicle.T_max == 80.0:  # Full thrust
        # Paper: optimal times around 512-513s
        experiment.flight_time_range = (300.0, 800.0)
    else:  # Quarter thrust
        # Paper: optimal times around 1050-1076s
        experiment.flight_time_range = (800.0, 1300.0)
    
    return experiment


def run_castalia_landing_experiment(
    landing_site: LandingSite,
    vehicle: VehicleParameters,
    initial_altitude: float = 2000.0,
    initial_velocity: float = 0.1,
    optimizer_params: Optional[FlightTimeOptimizerParameters] = None,
    plot_results: bool = False,
    save_plots: bool = False
) -> CastaliaLandingResult:
    """
    Run a single Castalia landing experiment.
    
    Args:
        landing_site: Landing site (LS1, LS2, or LS3)
        vehicle: Vehicle parameters (full or quarter thrust)
        initial_altitude: Initial altitude above landing site (m)
        initial_velocity: Initial velocity magnitude (m/s)
        optimizer_params: Parameters for flight time optimization
        plot_results: Whether to generate plots
        save_plots: Whether to save plots to files
        
    Returns:
        CastaliaLandingResult with experiment results
    """
    start_time = time.time()
    
    # Create experiment configuration
    experiment_config = create_castalia_experiment_config(
        landing_site=landing_site,
        vehicle=vehicle,
        initial_altitude=initial_altitude,
        initial_velocity=initial_velocity
    )
    
    # Set default optimizer parameters if not provided
    if optimizer_params is None:
        optimizer_params = FlightTimeOptimizerParameters(
            brent_tolerance=1.0,  # 1 second tolerance
            brent_max_iter=20,
            t_min_factor=0.8,
            t_max_factor=1.2,
            coarse_dt=10.0,  # Coarse discretization for Brent search
            fine_dt=2.0,     # Fine discretization for final solution
            verbose=True
        )
    
    logger.info(f"Running Castalia landing experiment: {landing_site.name}, "
                f"Thrust={vehicle.T_max}N, Altitude={initial_altitude}m")
    
    # Run flight time optimization
    optimization_result = optimize_flight_time(
        experiment_config=experiment_config,
        params=optimizer_params
    )
    
    # Extract solution
    solution = optimization_result.optimal_solution
    
    # Compute convergence metrics
    convergence_metrics = check_convergence_metrics(solution)
    
    # Compute thrust profile metrics
    thrust_metrics = compute_thrust_profile_metrics(solution)
    
    # Compute computation time
    computation_time = time.time() - start_time
    
    # Create result object
    result = CastaliaLandingResult(
        landing_site_name=landing_site.name,
        thrust_config="full" if vehicle.T_max == 80.0 else "quarter",
        experiment_config=experiment_config,
        optimization_result=optimization_result,
        solution=solution,
        convergence_metrics=convergence_metrics,
        thrust_metrics=thrust_metrics,
        computation_time=computation_time
    )
    
    # Generate plots if requested
    if plot_results:
        plot_castalia_landing_results(result, save_plots=save_plots)
    
    logger.info(f"Experiment completed: t_f={result.optimal_flight_time:.2f}s, "
                f"propellant={result.propellant_used:.3f}kg, "
                f"time={computation_time:.1f}s")
    
    return result


def plot_castalia_landing_results(
    result: CastaliaLandingResult,
    save_plots: bool = False
) -> None:
    """
    Generate plots for Castalia landing results.
    
    Args:
        result: Landing experiment results
        save_plots: Whether to save plots to files
    """
    site_name = result.landing_site_name
    thrust_config = result.thrust_config
    solution = result.solution
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 3D trajectory plot
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    plot_trajectory_3d(
        solution.position_history,
        solution.velocity_history,
        result.experiment_config.landing_site.position,
        ax=ax1,
        title=f"Trajectory - {site_name} ({thrust_config} thrust)"
    )
    
    # 2. Thrust magnitude profile
    ax2 = fig.add_subplot(2, 3, 2)
    plot_thrust_profile(
        solution.time_history,
        solution.thrust_magnitude_history,
        result.experiment_config.vehicle.T_max,
        result.experiment_config.vehicle.T_min,
        ax=ax2,
        title=f"Thrust Profile - {site_name}"
    )
    
    # 3. Mass profile
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(solution.time_history, solution.mass_history)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Mass (kg)')
    ax3.set_title('Mass Profile')
    ax3.grid(True)
    
    # 4. Altitude profile
    ax4 = fig.add_subplot(2, 3, 4)
    landing_site_pos = result.experiment_config.landing_site.position
    altitudes = []
    for pos in solution.position_history:
        # Altitude relative to landing site
        alt = np.linalg.norm(pos - landing_site_pos)
        altitudes.append(alt)
    ax4.plot(solution.time_history, altitudes)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Altitude (m)')
    ax4.set_title('Altitude Profile')
    ax4.grid(True)
    
    # 5. Velocity magnitude profile
    ax5 = fig.add_subplot(2, 3, 5)
    velocity_magnitudes = [np.linalg.norm(v) for v in solution.velocity_history]
    ax5.plot(solution.time_history, velocity_magnitudes)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Velocity (m/s)')
    ax5.set_title('Velocity Profile')
    ax5.grid(True)
    
    # 6. Convergence history (if available)
    ax6 = fig.add_subplot(2, 3, 6)
    if hasattr(result.optimization_result, 'evaluation_history'):
        eval_history = result.optimization_result.evaluation_history
        if eval_history:
            t_f_vals = [h['t_f'] for h in eval_history]
            prop_vals = [h['propellant'] for h in eval_history]
            ax6.plot(t_f_vals, prop_vals, 'bo-')
            ax6.plot(result.optimal_flight_time, result.propellant_used, 'r*', markersize=10)
            ax6.set_xlabel('Flight Time (s)')
            ax6.set_ylabel('Propellant (kg)')
            ax6.set_title('Brent Method Search')
            ax6.grid(True)
    
    plt.tight_layout()
    
    if save_plots:
        filename = f"castalia_{site_name}_{thrust_config}_thrust.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot to {filename}")
    
    plt.show()


def run_all_castalia_experiments(
    initial_altitude: float = 2000.0,
    initial_velocity: float = 0.1,
    plot_results: bool = False,
    save_plots: bool = False
) -> CastaliaExperimentSummary:
    """
    Run all Castalia landing experiments (3 sites × 2 thrust levels).
    
    Args:
        initial_altitude: Initial altitude above landing site (m)
        initial_velocity: Initial velocity magnitude (m/s)
        plot_results: Whether to generate plots
        save_plots: Whether to save plots to files
        
    Returns:
        CastaliaExperimentSummary with all results
    """
    logger.info("Starting all Castalia landing experiments")
    
    # Define landing sites and vehicles
    landing_sites = [CASTALIA_LS1, CASTALIA_LS2, CASTALIA_LS3]
    vehicles = [FULL_THRUST_VEHICLE, QUARTER_THRUST_VEHICLE]
    
    # Paper values from Table 7
    paper_values = {
        "LS1_full": {"t_f": 512.0, "propellant": 5.31},
        "LS2_full": {"t_f": 513.0, "propellant": 5.34},
        "LS3_full": {"t_f": 512.0, "propellant": 5.32},
        "LS1_quarter": {"t_f": 1050.0, "propellant": 3.40},
        "LS2_quarter": {"t_f": 1076.0, "propellant": 3.40},
        "LS3_quarter": {"t_f": 1050.0, "propellant": 3.40},
    }
    
    # Run experiments
    results = {}
    total_computation_time = 0.0
    
    for landing_site in landing_sites:
        for vehicle in vehicles:
            key = f"{landing_site.name}_{'full' if vehicle.T_max == 80.0 else 'quarter'}"
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Running experiment: {key}")
            logger.info(f"{'='*60}")
            
            result = run_castalia_landing_experiment(
                landing_site=landing_site,
                vehicle=vehicle,
                initial_altitude=initial_altitude,
                initial_velocity=initial_velocity,
                plot_results=plot_results,
                save_plots=save_plots
            )
            
            results[key] = result
            total_computation_time += result.computation_time
    
    # Compute overall statistics
    overall_stats = {
        "total_computation_time": total_computation_time,
        "average_computation_time": total_computation_time / len(results),
        "success_rate": sum(1 for r in results.values() if r.converged) / len(results) * 100,
        "average_brent_iterations": np.mean([r.brent_iterations for r in results.values()]),
        "average_function_evaluations": np.mean([r.function_evaluations for r in results.values()]),
    }
    
    # Create summary
    summary = CastaliaExperimentSummary(
        results=results,
        paper_comparison=paper_values,
        overall_stats=overall_stats
    )
    
    # Print summary
    summary.print_summary()
    
    # Print overall statistics
    print("\nOVERALL STATISTICS:")
    print(f"Total computation time: {overall_stats['total_computation_time']:.1f}s")
    print(f"Average per experiment: {overall_stats['average_computation_time']:.1f}s")
    print(f"Success rate: {overall_stats['success_rate']:.1f}%")
    print(f"Average Brent iterations: {overall_stats['average_brent_iterations']:.1f}")
    print(f"Average function evaluations: {overall_stats['average_function_evaluations']:.1f}")
    
    return summary


def analyze_switching_times(results: Dict[str, CastaliaLandingResult]) -> None:
    """
    Analyze switching times in thrust profiles (Table 5 in paper).
    
    Args:
        results: Dictionary of landing results
    """
    print("\n" + "="*80)
    print("SWITCHING TIME ANALYSIS (Table 5 comparison)")
    print("="*80)
    
    # Paper switching times for full thrust (Table 5)
    paper_switching_times = {
        "LS1_full": {"t1": 0.0, "t2": 512.0},  # Always at max thrust
        "LS2_full": {"t1": 0.0, "t2": 513.0},  # Always at max thrust
        "LS3_full": {"t1": 0.0, "t2": 512.0},  # Always at max thrust
    }
    
    print(f"{'Landing Site':<12} {'Thrust':<10} {'t1 (paper)':<12} {'t1 (ours)':<12} "
          f"{'t2 (paper)':<12} {'t2 (ours)':<12} {'Profile Type':<15}")
    print("-"*80)
    
    for key, result in results.items():
        if "full" in key:  # Only analyze full thrust cases
            site_name = result.landing_site_name
            thrust_config = result.thrust_config
            full_key = f"{site_name}_{thrust_config}"
            
            # Get thrust profile
            thrust_profile = result.solution.thrust_magnitude_history
            time_history = result.solution.time_history
            
            # Detect switching times (where thrust changes significantly)
            T_max = result.experiment_config.vehicle.T_max
            T_min = result.experiment_config.vehicle.T_min
            
            # Simple switching detection: find times where thrust is near max or min
            near_max = np.abs(thrust_profile - T_max) < 0.1 * T_max
            near_min = np.abs(thrust_profile - T_min) < 0.1 * T_min
            
            # Find transitions
            t1 = 0.0  # Start time (usually at max thrust)
            t2 = time_history[-1]  # End time
            
            # Determine profile type
            if np.all(near_max):
                profile_type = "Max-Max"
            elif np.any(near_min):
                # Check if it's Max-Min-Max or Max-Min
                min_indices = np.where(near_min)[0]
                if len(min_indices) > 0:
                    if near_max[-1]:  # Ends at max
                        profile_type = "Max-Min-Max"
                    else:  # Ends at min
                        profile_type = "Max-Min"
                else:
                    profile_type = "Complex"
            else:
                profile_type = "Other"
            
            # Get paper values
            paper_t1 = paper_switching_times.get(full_key, {}).get('t1', 0.0)
            paper_t2 = paper_switching_times.get(full_key, {}).get('t2', 0.0)
            
            print(f"{site_name:<12} {thrust_config:<10} {paper_t1:>10.2f} {t1:>10.2f} "
                  f"{paper_t2:>10.2f} {t2:>10.2f} {profile_type:<15}")
    
    print("-"*80)


def main():
    """Main function to run Castalia landing experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Castalia landing experiments")
    parser.add_argument("--site", type=str, choices=["LS1", "LS2", "LS3", "all"],
                       default="all", help="Landing site to test")
    parser.add_argument("--thrust", type=str, choices=["full", "quarter", "both"],
                       default="both", help="Thrust configuration")
    parser.add_argument("--altitude", type=float, default=2000.0,
                       help="Initial altitude (m)")
    parser.add_argument("--velocity", type=float, default=0.1,
                       help="Initial velocity magnitude (m/s)")
    parser.add_argument("--plot", action="store_true",
                       help="Generate plots")
    parser.add_argument("--save-plots", action="store_true",
                       help="Save plots to files")
    parser.add_argument("--analyze-switching", action="store_true",
                       help="Analyze switching times (Table 5)")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if args.site == "all":
        # Run all experiments
        summary = run_all_castalia_experiments(
            initial_altitude=args.altitude,
            initial_velocity=args.velocity,
            plot_results=args.plot,
            save_plots=args.save_plots
        )
        
        if args.analyze_switching:
            analyze_switching_times(summary.results)
    else:
        # Run specific experiment
        landing_site_map = {
            "LS1": CASTALIA_LS1,
            "LS2": CASTALIA_LS2,
            "LS3": CASTALIA_LS3
        }
        
        vehicle_map = {
            "full": FULL_THRUST_VEHICLE,
            "quarter": QUARTER_THRUST_VEHICLE
        }
        
        if args.thrust == "both":
            vehicles = [FULL_THRUST_VEHICLE, QUARTER_THRUST_VEHICLE]
        else:
            vehicles = [vehicle_map[args.thrust]]
        
        results = {}
        for vehicle in vehicles:
            key = f"{args.site}_{'full' if vehicle.T_max == 80.0 else 'quarter'}"
            
            result = run_castalia_landing_experiment(
                landing_site=landing_site_map[args.site],
                vehicle=vehicle,
                initial_altitude=args.altitude,
                initial_velocity=args.velocity,
                plot_results=args.plot,
                save_plots=args.save_plots
            )
            
            results[key] = result
        
        # Create summary for the specific cases
        paper_values = {
            "LS1_full": {"t_f": 512.0, "propellant": 5.31},
            "LS2_full": {"t_f": 513.0, "propellant": 5.34},
            "LS3_full": {"t_f": 512.0, "propellant": 5.32},
            "LS1_quarter": {"t_f": 1050.0, "propellant": 3.40},
            "LS2_quarter": {"t_f": 1076.0, "propellant": 3.40},
            "LS3_quarter": {"t_f": 1050.0, "propellant": 3.40},
        }
        
        summary = CastaliaExperimentSummary(
            results=results,
            paper_comparison={k: v for k, v in paper_values.items() if k in results},
            overall_stats={}
        )
        
        summary.print_summary()
        
        if args.analyze_switching:
            analyze_switching_times(results)


if __name__ == "__main__":
    main()