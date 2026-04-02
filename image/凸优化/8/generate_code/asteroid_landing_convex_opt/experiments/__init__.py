"""
Experiments module for asteroid landing convex optimization.

This module provides all experiment scripts for reproducing the paper's results,
including Castalia landing experiments, parameter sweeps, trajectory analysis,
and validation tests.

Public API:
    - Castalia landing experiments: run_castalia_landing_experiment, run_all_castalia_experiments
    - Parameter sweeps: run_parameter_sweep, run_triaxial_ellipsoid_sweeps
    - Trajectory analysis: analyze_complete_trajectory, print_trajectory_analysis_summary
    - Validation: run_validation_tests, validate_lossless_convexification
"""

from .castalia_landing import (
    CastaliaLandingResult,
    CastaliaExperimentSummary,
    create_castalia_experiment_config,
    run_castalia_landing_experiment,
    run_all_castalia_experiments,
    plot_castalia_landing_results,
    analyze_switching_times,
    main as castalia_main,
)

from .parameter_sweeps import (
    ParameterSweepResult,
    MarsBaselineResult,
    compute_mars_baseline,
    run_parameter_sweep,
    run_triaxial_ellipsoid_sweeps,
    plot_triaxial_sweeps,
    analyze_unimodality,
    print_sweep_summary,
    main as parameter_sweeps_main,
)

from .trajectory_analysis import (
    ThrustProfileAnalysis,
    TrajectoryConvergenceAnalysis,
    ConstraintViolationAnalysis,
    TrajectoryQualityMetrics,
    CompleteTrajectoryAnalysis,
    analyze_thrust_profile,
    analyze_convergence,
    analyze_constraint_violations,
    analyze_trajectory_quality,
    analyze_complete_trajectory,
    print_trajectory_analysis_summary,
    plot_trajectory_analysis,
    compare_with_paper_results,
    test_trajectory_analysis,
)

from .validation import (
    ValidationResult,
    ValidationTest,
    run_validation_tests,
    validate_lossless_convexification,
    validate_gravity_models,
    validate_constraint_satisfaction,
    validate_convergence_behavior,
    compare_with_paper_results as validate_compare_with_paper,
    main as validation_main,
)

# Convenience functions for running all experiments
def run_all_experiments(
    castalia_experiments: bool = True,
    parameter_sweeps: bool = True,
    validation_tests: bool = True,
    plot_results: bool = False,
    save_plots: bool = False,
    output_dir: str = "results",
) -> dict:
    """
    Run all experiments from the paper.
    
    Args:
        castalia_experiments: Whether to run Castalia landing experiments
        parameter_sweeps: Whether to run parameter sweep experiments
        validation_tests: Whether to run validation tests
        plot_results: Whether to generate plots
        save_plots: Whether to save plots to files
        output_dir: Directory to save results and plots
        
    Returns:
        Dictionary containing all experiment results
    """
    import os
    import logging
    from pathlib import Path
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Run Castalia landing experiments
    if castalia_experiments:
        logging.info("Running Castalia landing experiments...")
        castalia_results = run_all_castalia_experiments(
            plot_results=plot_results,
            save_plots=save_plots,
        )
        results["castalia"] = castalia_results
        
        # Save summary
        castalia_summary_file = output_path / "castalia_summary.txt"
        with open(castalia_summary_file, "w") as f:
            f.write(str(castalia_results))
        
        logging.info(f"Castalia experiments completed. Summary saved to {castalia_summary_file}")
    
    # Run parameter sweeps
    if parameter_sweeps:
        logging.info("Running parameter sweep experiments...")
        sweep_results = run_triaxial_ellipsoid_sweeps(
            save_plots=save_plots,
            output_dir=str(output_path / "parameter_sweeps"),
        )
        results["parameter_sweeps"] = sweep_results
        
        # Save summary
        sweep_summary_file = output_path / "parameter_sweeps_summary.txt"
        with open(sweep_summary_file, "w") as f:
            f.write(str(sweep_results))
        
        logging.info(f"Parameter sweep experiments completed. Summary saved to {sweep_summary_file}")
    
    # Run validation tests
    if validation_tests:
        logging.info("Running validation tests...")
        validation_results = run_validation_tests(
            output_dir=str(output_path / "validation"),
        )
        results["validation"] = validation_results
        
        # Save summary
        validation_summary_file = output_path / "validation_summary.txt"
        with open(validation_summary_file, "w") as f:
            f.write(str(validation_results))
        
        logging.info(f"Validation tests completed. Summary saved to {validation_summary_file}")
    
    return results

def print_experiment_summary(results: dict) -> None:
    """
    Print a comprehensive summary of all experiment results.
    
    Args:
        results: Dictionary of experiment results from run_all_experiments
    """
    print("=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    
    if "castalia" in results:
        print("\nCASTALIA LANDING EXPERIMENTS:")
        print("-" * 40)
        castalia_summary = results["castalia"]
        if hasattr(castalia_summary, 'print_summary'):
            castalia_summary.print_summary()
        else:
            print(str(castalia_summary))
    
    if "parameter_sweeps" in results:
        print("\nPARAMETER SWEEP EXPERIMENTS:")
        print("-" * 40)
        sweep_results = results["parameter_sweeps"]
        print_sweep_summary(sweep_results)
    
    if "validation" in results:
        print("\nVALIDATION TESTS:")
        print("-" * 40)
        validation_results = results["validation"]
        if hasattr(validation_results, 'print_summary'):
            validation_results.print_summary()
        else:
            print(str(validation_results))
    
    print("\n" + "=" * 80)
    print("END OF EXPERIMENT SUMMARY")
    print("=" * 80)

# Define public API
__all__ = [
    # Castalia landing
    "CastaliaLandingResult",
    "CastaliaExperimentSummary",
    "create_castalia_experiment_config",
    "run_castalia_landing_experiment",
    "run_all_castalia_experiments",
    "plot_castalia_landing_results",
    "analyze_switching_times",
    "castalia_main",
    
    # Parameter sweeps
    "ParameterSweepResult",
    "MarsBaselineResult",
    "compute_mars_baseline",
    "run_parameter_sweep",
    "run_triaxial_ellipsoid_sweeps",
    "plot_triaxial_sweeps",
    "analyze_unimodality",
    "print_sweep_summary",
    "parameter_sweeps_main",
    
    # Trajectory analysis
    "ThrustProfileAnalysis",
    "TrajectoryConvergenceAnalysis",
    "ConstraintViolationAnalysis",
    "TrajectoryQualityMetrics",
    "CompleteTrajectoryAnalysis",
    "analyze_thrust_profile",
    "analyze_convergence",
    "analyze_constraint_violations",
    "analyze_trajectory_quality",
    "analyze_complete_trajectory",
    "print_trajectory_analysis_summary",
    "plot_trajectory_analysis",
    "compare_with_paper_results",
    "test_trajectory_analysis",
    
    # Validation
    "ValidationResult",
    "ValidationTest",
    "run_validation_tests",
    "validate_lossless_convexification",
    "validate_gravity_models",
    "validate_constraint_satisfaction",
    "validate_convergence_behavior",
    "validate_compare_with_paper",
    "validation_main",
    
    # Convenience functions
    "run_all_experiments",
    "print_experiment_summary",
]

# Test function for the module
def test_experiments_module() -> bool:
    """
    Run basic tests for the experiments module.
    
    Returns:
        True if all tests pass, False otherwise
    """
    import logging
    
    logging.info("Testing experiments module...")
    
    # Test that all imports work
    try:
        # Test Castalia imports
        from .castalia_landing import run_all_castalia_experiments as test_castalia
        logging.info("✓ Castalia landing imports successful")
        
        # Test parameter sweep imports
        from .parameter_sweeps import run_triaxial_ellipsoid_sweeps as test_sweeps
        logging.info("✓ Parameter sweep imports successful")
        
        # Test trajectory analysis imports
        from .trajectory_analysis import analyze_complete_trajectory as test_analysis
        logging.info("✓ Trajectory analysis imports successful")
        
        # Test validation imports
        from .validation import run_validation_tests as test_validation
        logging.info("✓ Validation imports successful")
        
        logging.info("All experiments module imports successful")
        return True
        
    except ImportError as e:
        logging.error(f"Import error in experiments module: {e}")
        return False
    except Exception as e:
        logging.error(f"Error testing experiments module: {e}")
        return False

if __name__ == "__main__":
    # Run module tests when executed directly
    import sys
    success = test_experiments_module()
    sys.exit(0 if success else 1)