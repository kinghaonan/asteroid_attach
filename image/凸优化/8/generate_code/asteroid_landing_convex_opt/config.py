"""
Configuration parameters for asteroid landing convex optimization.

This module defines all physical parameters, asteroid properties, vehicle specifications,
and experiment configurations used throughout the project. Parameters are based on
the paper "Trajectory Design Employing Convex Optimization for Landing on Irregularly
Shaped Asteroids" and associated references.

Author: Implementation Team
Date: 2026-01-15
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class AsteroidParameters:
    """Physical parameters for an asteroid."""
    name: str
    mu: float  # gravitational parameter [m^3/s^2]
    R_b: float  # Brillouin sphere radius [m]
    rotation_rate: float  # angular velocity [rad/s]
    rotation_axis: np.ndarray  # unit vector of rotation axis
    shape_params: Dict  # shape-specific parameters
    gravity_model: str  # 'spherical_harmonics' or 'bessel'
    
    # For triaxial ellipsoids
    semi_axes: Optional[Tuple[float, float, float]] = None  # (a, b, c) [m]
    
    # For Castalia
    castalia_coeffs_file: Optional[str] = None


@dataclass
class VehicleParameters:
    """Spacecraft physical parameters."""
    m_wet: float  # initial mass [kg]
    m_dry: float  # dry mass [kg]
    I_sp: float  # specific impulse [s]
    g_0: float  # Earth gravity [m/s^2]
    T_max: float  # maximum thrust [N]
    T_min: float  # minimum thrust [N]
    
    @property
    def mass_ratio(self) -> float:
        """Mass ratio m_wet/m_dry."""
        return self.m_wet / self.m_dry
    
    @property
    def max_acceleration(self) -> float:
        """Maximum acceleration T_max/m_wet [m/s^2]."""
        return self.T_max / self.m_wet
    
    @property
    def min_acceleration(self) -> float:
        """Minimum acceleration T_min/m_wet [m/s^2]."""
        return self.T_min / self.m_wet


@dataclass
class LandingSite:
    """Landing site definition."""
    name: str
    position: np.ndarray  # [x, y, z] in asteroid-fixed frame [m]
    surface_normal: np.ndarray  # unit normal vector
    glide_slope_angle: float  # maximum approach angle [rad]


@dataclass
class ExperimentConfig:
    """Configuration for a specific experiment."""
    asteroid: AsteroidParameters
    vehicle: VehicleParameters
    landing_site: LandingSite
    initial_state: Dict  # r0, v0
    final_state: Dict  # rf, vf
    flight_time_range: Tuple[float, float]  # (t_min, t_max) [s]
    discretization_dt: float  # time step for discretization [s]
    convergence_tolerance: float  # position tolerance for successive solution [m]
    max_iterations: int  # maximum iterations for successive solution


# ============================================================================
# ASTEROID DEFINITIONS
# ============================================================================

# Triaxial ellipsoids from paper (Table 1)
ASTEROID_A1 = AsteroidParameters(
    name="A1",
    mu=2.0e9,  # [m^3/s^2]
    R_b=1000.0,  # [m]
    rotation_rate=2.0e-4,  # [rad/s]
    rotation_axis=np.array([0, 0, 1]),  # z-axis rotation
    shape_params={"type": "triaxial_ellipsoid"},
    semi_axes=(1000.0, 800.0, 600.0),  # (a, b, c) [m]
    gravity_model="spherical_harmonics"
)

ASTEROID_A2 = AsteroidParameters(
    name="A2",
    mu=2.0e9,  # [m^3/s^2]
    R_b=1000.0,  # [m]
    rotation_rate=2.0e-4,  # [rad/s]
    rotation_axis=np.array([0, 0, 1]),  # z-axis rotation
    shape_params={"type": "triaxial_ellipsoid"},
    semi_axes=(1000.0, 700.0, 500.0),  # (a, b, c) [m]
    gravity_model="spherical_harmonics"
)

ASTEROID_A3 = AsteroidParameters(
    name="A3",
    mu=2.0e9,  # [m^3/s^2]
    R_b=1000.0,  # [m]
    rotation_rate=2.0e-4,  # [rad/s]
    rotation_axis=np.array([0, 0, 1]),  # z-axis rotation
    shape_params={"type": "triaxial_ellipsoid"},
    semi_axes=(1000.0, 600.0, 400.0),  # (a, b, c) [m]
    gravity_model="spherical_harmonics"
)

# Castalia asteroid (Table 2)
CASTALIA = AsteroidParameters(
    name="Castalia",
    mu=4.46e6,  # [m^3/s^2] - from paper
    R_b=850.0,  # [m] - Brillouin sphere radius
    rotation_rate=4.041e-4,  # [rad/s] - rotation period ~4.3 hours
    rotation_axis=np.array([0, 0, 1]),  # assumed z-axis rotation
    shape_params={
        "type": "castalia",
        "polyhedron_file": None,  # would reference shape model
        "coeffs_file": "data/castalia_coeffs.npy"  # spherical harmonic coefficients
    },
    gravity_model="mixed",  # uses both spherical harmonics and Bessel
    castalia_coeffs_file="data/castalia_coeffs.npy"
)


# ============================================================================
# VEHICLE DEFINITIONS
# ============================================================================

# Full thrust configuration (Table 3)
FULL_THRUST_VEHICLE = VehicleParameters(
    m_wet=500.0,  # [kg]
    m_dry=494.66,  # [kg] - from paper results
    I_sp=225.0,  # [s]
    g_0=9.80665,  # [m/s^2]
    T_max=80.0,  # [N]
    T_min=5.0  # [N] - minimum thrust for control authority
)

# Quarter thrust configuration (Table 3)
QUARTER_THRUST_VEHICLE = VehicleParameters(
    m_wet=500.0,  # [kg]
    m_dry=496.6,  # [kg] - from paper results (approx)
    I_sp=225.0,  # [s]
    g_0=9.80665,  # [m/s^2]
    T_max=20.0,  # [N]
    T_min=1.25  # [N] - quarter of minimum thrust
)


# ============================================================================
# LANDING SITES
# ============================================================================

# Castalia landing sites (Table 4)
CASTALIA_LS1 = LandingSite(
    name="LS1",
    position=np.array([0.0, 0.0, 850.0]),  # [m] - on surface at z=850
    surface_normal=np.array([0.0, 0.0, 1.0]),  # pointing outward
    glide_slope_angle=np.deg2rad(30.0)  # [rad]
)

CASTALIA_LS2 = LandingSite(
    name="LS2",
    position=np.array([600.0, 0.0, 600.0]),  # [m]
    surface_normal=np.array([0.707, 0.0, 0.707]),  # 45-degree slope
    glide_slope_angle=np.deg2rad(30.0)  # [rad]
)

CASTALIA_LS3 = LandingSite(
    name="LS3",
    position=np.array([0.0, 600.0, 600.0]),  # [m]
    surface_normal=np.array([0.0, 0.707, 0.707]),  # 45-degree slope
    glide_slope_angle=np.deg2rad(30.0)  # [rad]
)


# ============================================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================================

def create_castalia_experiment(
    landing_site: LandingSite,
    vehicle: VehicleParameters,
    initial_altitude: float = 2000.0,
    initial_velocity: float = 0.1
) -> ExperimentConfig:
    """
    Create experiment configuration for Castalia landing.
    
    Args:
        landing_site: Landing site (LS1, LS2, or LS3)
        vehicle: Vehicle configuration (full or quarter thrust)
        initial_altitude: Initial altitude above landing site [m]
        initial_velocity: Initial velocity magnitude [m/s]
        
    Returns:
        ExperimentConfig for the specified landing scenario
    """
    # Initial position: directly above landing site at specified altitude
    r0 = landing_site.position + landing_site.surface_normal * initial_altitude
    
    # Initial velocity: small downward velocity toward landing site
    v0 = -landing_site.surface_normal * initial_velocity
    
    # Final state: at landing site with zero velocity
    rf = landing_site.position
    vf = np.array([0.0, 0.0, 0.0])
    
    # Flight time range based on paper results
    if vehicle.T_max == 80.0:  # full thrust
        t_range = (400.0, 600.0)  # [s] - optimal around 512-513s
    else:  # quarter thrust
        t_range = (800.0, 1200.0)  # [s] - optimal around 1050-1076s
    
    return ExperimentConfig(
        asteroid=CASTALIA,
        vehicle=vehicle,
        landing_site=landing_site,
        initial_state={"r0": r0, "v0": v0},
        final_state={"rf": rf, "vf": vf},
        flight_time_range=t_range,
        discretization_dt=2.0,  # [s] - from paper
        convergence_tolerance=0.5,  # [m] - from paper
        max_iterations=10  # maximum iterations for successive solution
    )


def create_triaxial_experiment(
    asteroid: AsteroidParameters,
    vehicle: VehicleParameters,
    landing_site_position: np.ndarray = None
) -> ExperimentConfig:
    """
    Create experiment configuration for triaxial ellipsoid landing.
    
    Args:
        asteroid: Asteroid (A1, A2, or A3)
        vehicle: Vehicle configuration
        landing_site_position: Optional custom landing site position
        
    Returns:
        ExperimentConfig for the specified scenario
    """
    # Default landing site: at pole along z-axis
    if landing_site_position is None:
        landing_site_position = np.array([0.0, 0.0, asteroid.semi_axes[2]])
    
    landing_site = LandingSite(
        name=f"{asteroid.name}_pole",
        position=landing_site_position,
        surface_normal=np.array([0.0, 0.0, 1.0]),
        glide_slope_angle=np.deg2rad(30.0)
    )
    
    # Initial position: 2 km above landing site
    r0 = landing_site_position + np.array([0.0, 0.0, 2000.0])
    
    # Initial velocity: small downward velocity
    v0 = np.array([0.0, 0.0, -0.1])
    
    # Final state
    rf = landing_site_position
    vf = np.array([0.0, 0.0, 0.0])
    
    return ExperimentConfig(
        asteroid=asteroid,
        vehicle=vehicle,
        landing_site=landing_site,
        initial_state={"r0": r0, "v0": v0},
        final_state={"rf": rf, "vf": vf},
        flight_time_range=(200.0, 1000.0),  # broad range for parameter sweeps
        discretization_dt=2.0,  # [s]
        convergence_tolerance=0.5,  # [m]
        max_iterations=10
    )


# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================

# Universal constants
G = 6.67430e-11  # gravitational constant [m^3/kg/s^2]

# Numerical parameters
DEFAULT_SOLVER = "MOSEK"  # or "SCS" for open-source alternative
SOLVER_SETTINGS = {
    "MOSEK": {
        "verbose": False,
        "max_iters": 200,
        "feastol": 1e-8,
        "reltol": 1e-8,
        "abstol": 1e-8
    },
    "SCS": {
        "verbose": False,
        "max_iters": 5000,
        "eps": 1e-6,
        "alpha": 1.5,
        "scale": 5.0,
        "normalize": True
    }
}

# Scaling parameters (Eq. 59)
SCALING_CONFIG = {
    "distance": "semi_minor_axis",  # or "brillouin_radius"
    "velocity": "sqrt_g_distance",
    "time": "sqrt_distance_over_g",
    "mass": "logarithmic"  # use q = ln(m) for mass scaling
}

# Validation tolerances
VALIDATION_TOLERANCES = {
    "position_convergence": 0.5,  # [m]
    "thrust_equality": 1e-6,  # relative tolerance for ∥a_t∥ = a_tm
    "constraint_violation": 1e-4,  # [m] or [m/s]
    "boundary_conditions": 1e-3,  # [m] for position, [m/s] for velocity
    "mass_limit": 1e-6  # [kg]
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_asteroid_by_name(name: str) -> AsteroidParameters:
    """Get asteroid parameters by name."""
    asteroids = {
        "A1": ASTEROID_A1,
        "A2": ASTEROID_A2,
        "A3": ASTEROID_A3,
        "Castalia": CASTALIA
    }
    
    if name not in asteroids:
        raise ValueError(f"Unknown asteroid: {name}. Available: {list(asteroids.keys())}")
    
    return asteroids[name]


def get_vehicle_by_thrust(thrust_level: str) -> VehicleParameters:
    """Get vehicle parameters by thrust level."""
    vehicles = {
        "full": FULL_THRUST_VEHICLE,
        "quarter": QUARTER_THRUST_VEHICLE
    }
    
    if thrust_level not in vehicles:
        raise ValueError(f"Unknown thrust level: {thrust_level}. Available: {list(vehicles.keys())}")
    
    return vehicles[thrust_level]


def get_landing_site_by_name(name: str) -> LandingSite:
    """Get landing site by name."""
    sites = {
        "LS1": CASTALIA_LS1,
        "LS2": CASTALIA_LS2,
        "LS3": CASTALIA_LS3
    }
    
    if name not in sites:
        raise ValueError(f"Unknown landing site: {name}. Available: {list(sites.keys())}")
    
    return sites[name]


# Export commonly used configurations
__all__ = [
    # Asteroid parameters
    "ASTEROID_A1", "ASTEROID_A2", "ASTEROID_A3", "CASTALIA",
    
    # Vehicle parameters
    "FULL_THRUST_VEHICLE", "QUARTER_THRUST_VEHICLE",
    
    # Landing sites
    "CASTALIA_LS1", "CASTALIA_LS2", "CASTALIA_LS3",
    
    # Experiment creation functions
    "create_castalia_experiment", "create_triaxial_experiment",
    
    # Helper functions
    "get_asteroid_by_name", "get_vehicle_by_thrust", "get_landing_site_by_name",
    
    # Constants and settings
    "G", "DEFAULT_SOLVER", "SOLVER_SETTINGS", "SCALING_CONFIG", "VALIDATION_TOLERANCES",
    
    # Data classes
    "AsteroidParameters", "VehicleParameters", "LandingSite", "ExperimentConfig"
]