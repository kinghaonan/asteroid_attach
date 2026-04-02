"""
Vehicle module for asteroid landing convex optimization.

This module provides spacecraft and engine models for the optimization problem.
It includes the full-thrust (80N max) and quarter-thrust (20N max) configurations
from the paper, as well as utility functions for creating and managing vehicle models.

Classes:
    Spacecraft: Complete spacecraft model with mass, thrust, and engine parameters.
    EngineParameters: Engine-specific parameters for thrust and mass flow calculations.

Functions:
    create_full_thrust_spacecraft: Creates the full-thrust configuration (80N max).
    create_quarter_thrust_spacecraft: Creates the quarter-thrust configuration (20N max).
    create_mars_lander_spacecraft: Creates a Mars lander for baseline comparison.
    get_spacecraft_by_name: Retrieves a predefined spacecraft by name.
    create_full_thrust_engine: Creates engine parameters for full-thrust configuration.
    create_quarter_thrust_engine: Creates engine parameters for quarter-thrust configuration.
    create_mars_lander_engine: Creates engine parameters for Mars lander baseline.
    get_engine_by_name: Retrieves predefined engine configuration by name.
"""

from .spacecraft import (
    Spacecraft,
    create_full_thrust_spacecraft,
    create_quarter_thrust_spacecraft,
    create_mars_lander_spacecraft,
    get_spacecraft_by_name,
)

from .engine import (
    EngineParameters,
    create_full_thrust_engine,
    create_quarter_thrust_engine,
    create_mars_lander_engine,
    get_engine_by_name,
)

__all__ = [
    # Spacecraft classes and functions
    "Spacecraft",
    "create_full_thrust_spacecraft",
    "create_quarter_thrust_spacecraft",
    "create_mars_lander_spacecraft",
    "get_spacecraft_by_name",
    
    # Engine classes and functions
    "EngineParameters",
    "create_full_thrust_engine",
    "create_quarter_thrust_engine",
    "create_mars_lander_engine",
    "get_engine_by_name",
]

# Convenience constants for common configurations
FULL_THRUST_SPACECRAFT = create_full_thrust_spacecraft()
QUARTER_THRUST_SPACECRAFT = create_quarter_thrust_spacecraft()
MARS_LANDER_SPACECRAFT = create_mars_lander_spacecraft()

FULL_THRUST_ENGINE = create_full_thrust_engine()
QUARTER_THRUST_ENGINE = create_quarter_thrust_engine()
MARS_LANDER_ENGINE = create_mars_lander_engine()