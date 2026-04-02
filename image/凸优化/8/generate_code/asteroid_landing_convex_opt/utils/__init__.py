"""
Utilities module for asteroid landing convex optimization.

This module provides helper functions for visualization, metrics computation,
convergence checking, and coordinate transformations used throughout the
optimization framework.
"""

from .visualization import (
    plot_trajectory_3d,
    plot_thrust_profile,
    plot_mass_profile,
    plot_altitude_profile,
    plot_velocity_profile,
    plot_brent_search_history,
    create_trajectory_animation,
    save_plot,
    plot_parameter_sweep_results,
    plot_convergence_history,
    plot_constraint_violations,
    plot_gravity_field,
    plot_asteroid_shape,
    create_comprehensive_plot,
    test_visualization
)

from .metrics import (
    compute_propellant_used,
    compute_thrust_profile_metrics,
    compute_trajectory_metrics,
    compute_constraint_violations,
    compute_convergence_metrics,
    compute_gravity_error,
    compute_paper_comparison_metrics,
    validate_solution,
    test_metrics
)

from .convergence_check import (
    check_convergence_metrics,
    analyze_convergence_history,
    compute_convergence_rate,
    check_position_convergence,
    check_velocity_convergence,
    check_mass_convergence,
    check_thrust_convergence,
    check_overall_convergence,
    print_convergence_summary,
    plot_convergence_history,
    test_convergence_check
)

from .coordinate_transforms import (
    cartesian_to_spherical,
    spherical_to_cartesian,
    rotate_to_asteroid_frame,
    rotate_from_asteroid_frame,
    compute_surface_normal,
    compute_glide_slope_cone,
    compute_vertical_motion_constraint,
    test_coordinate_transforms
)

# Convenience functions
from .visualization import create_comprehensive_plot as plot_trajectory_summary
from .metrics import compute_propellant_used as calculate_propellant
from .convergence_check import check_overall_convergence as check_convergence

__all__ = [
    # Visualization
    'plot_trajectory_3d',
    'plot_thrust_profile',
    'plot_mass_profile',
    'plot_altitude_profile',
    'plot_velocity_profile',
    'plot_brent_search_history',
    'create_trajectory_animation',
    'save_plot',
    'plot_parameter_sweep_results',
    'plot_convergence_history',
    'plot_constraint_violations',
    'plot_gravity_field',
    'plot_asteroid_shape',
    'create_comprehensive_plot',
    'plot_trajectory_summary',
    'test_visualization',
    
    # Metrics
    'compute_propellant_used',
    'compute_thrust_profile_metrics',
    'compute_trajectory_metrics',
    'compute_constraint_violations',
    'compute_convergence_metrics',
    'compute_gravity_error',
    'compute_paper_comparison_metrics',
    'validate_solution',
    'calculate_propellant',
    'test_metrics',
    
    # Convergence check
    'check_convergence_metrics',
    'analyze_convergence_history',
    'compute_convergence_rate',
    'check_position_convergence',
    'check_velocity_convergence',
    'check_mass_convergence',
    'check_thrust_convergence',
    'check_overall_convergence',
    'check_convergence',
    'print_convergence_summary',
    'plot_convergence_history',
    'test_convergence_check',
    
    # Coordinate transforms
    'cartesian_to_spherical',
    'spherical_to_cartesian',
    'rotate_to_asteroid_frame',
    'rotate_from_asteroid_frame',
    'compute_surface_normal',
    'compute_glide_slope_cone',
    'compute_vertical_motion_constraint',
    'test_coordinate_transforms',
]

def test_utils_module() -> bool:
    """
    Run comprehensive tests for the entire utils module.
    
    Returns:
        bool: True if all tests pass, False otherwise
    """
    import logging
    logger = logging.getLogger(__name__)
    
    test_results = []
    
    try:
        # Test visualization
        logger.info("Testing visualization module...")
        test_results.append(test_visualization())
    except Exception as e:
        logger.error(f"Visualization test failed: {e}")
        test_results.append(False)
    
    try:
        # Test metrics
        logger.info("Testing metrics module...")
        test_results.append(test_metrics())
    except Exception as e:
        logger.error(f"Metrics test failed: {e}")
        test_results.append(False)
    
    try:
        # Test convergence check
        logger.info("Testing convergence check module...")
        test_results.append(test_convergence_check())
    except Exception as e:
        logger.error(f"Convergence check test failed: {e}")
        test_results.append(False)
    
    try:
        # Test coordinate transforms
        logger.info("Testing coordinate transforms module...")
        test_results.append(test_coordinate_transforms())
    except Exception as e:
        logger.error(f"Coordinate transforms test failed: {e}")
        test_results.append(False)
    
    all_passed = all(test_results)
    if all_passed:
        logger.info("All utils module tests passed!")
    else:
        logger.warning(f"Some utils module tests failed: {test_results}")
    
    return all_passed