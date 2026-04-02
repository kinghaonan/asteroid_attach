"""
Dynamics module for asteroid landing convex optimization.

This module provides the equations of motion, discretization methods, and scaling
systems for the asteroid landing trajectory optimization problem.

The module implements:
1. State equations with gravitational, Coriolis, centrifugal, and Euler terms
2. Trapezoidal discretization for convex optimization constraints
3. Numerical scaling for improved solver conditioning
"""

from .state_equations import (
    StateVector,
    ControlVector,
    StateEquations,
    test_state_equations,
)

from .discretization import (
    DiscretizationParameters,
    TrapezoidalDiscretizer,
    test_discretization,
)

from .scaling import (
    ScalingFactors,
    ScalingSystem,
    create_default_scaling_system,
    test_scaling_system,
)

__all__ = [
    # State equations
    "StateVector",
    "ControlVector",
    "StateEquations",
    "test_state_equations",
    
    # Discretization
    "DiscretizationParameters",
    "TrapezoidalDiscretizer",
    "test_discretization",
    
    # Scaling
    "ScalingFactors",
    "ScalingSystem",
    "create_default_scaling_system",
    "test_scaling_system",
]

# Version information
__version__ = "1.0.0"
__author__ = "Asteroid Landing Convex Optimization Team"
__description__ = "Dynamics module for asteroid landing trajectory optimization"