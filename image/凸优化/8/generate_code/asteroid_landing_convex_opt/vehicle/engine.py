"""
Engine and thrust model for spacecraft propulsion system.

This module implements the engine model for the spacecraft, including thrust
characteristics, mass flow rate calculations, and engine-specific parameters
that are used in the optimization problem formulation.

Based on the paper's vehicle parameters:
- Full thrust: 80N max, 20N min
- Quarter thrust: 20N max, 5N min
- Specific impulse (I_sp): 225s for both configurations
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class EngineParameters:
    """
    Parameters for a rocket engine model.
    
    Attributes:
        name: Engine name/identifier
        I_sp: Specific impulse (seconds)
        g_0: Standard gravity (m/s²)
        max_thrust: Maximum thrust (N)
        min_thrust: Minimum thrust (N)
        max_mass_flow_rate: Maximum mass flow rate (kg/s)
        min_mass_flow_rate: Minimum mass flow rate (kg/s)
        throttle_range: Throttle range as (min, max) fraction
        engine_efficiency: Engine efficiency factor (0-1)
        structural_factor: Structural mass factor (dry/wet mass ratio)
        description: Optional description of the engine
    """
    name: str
    I_sp: float
    g_0: float = 9.80665
    max_thrust: float = 80.0
    min_thrust: float = 20.0
    max_mass_flow_rate: Optional[float] = None
    min_mass_flow_rate: Optional[float] = None
    throttle_range: Tuple[float, float] = (0.25, 1.0)  # 25% to 100%
    engine_efficiency: float = 1.0
    structural_factor: Optional[float] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        """Validate and compute derived parameters."""
        # Validate thrust bounds
        if self.max_thrust <= 0:
            raise ValueError(f"max_thrust must be positive, got {self.max_thrust}")
        if self.min_thrust <= 0:
            raise ValueError(f"min_thrust must be positive, got {self.min_thrust}")
        if self.min_thrust >= self.max_thrust:
            raise ValueError(f"min_thrust ({self.min_thrust}) must be less than max_thrust ({self.max_thrust})")
        
        # Validate specific impulse
        if self.I_sp <= 0:
            raise ValueError(f"I_sp must be positive, got {self.I_sp}")
        
        # Validate throttle range
        if self.throttle_range[0] <= 0 or self.throttle_range[0] >= 1:
            raise ValueError(f"Minimum throttle must be between 0 and 1, got {self.throttle_range[0]}")
        if self.throttle_range[1] <= 0 or self.throttle_range[1] > 1:
            raise ValueError(f"Maximum throttle must be between 0 and 1, got {self.throttle_range[1]}")
        if self.throttle_range[0] >= self.throttle_range[1]:
            raise ValueError(f"Minimum throttle ({self.throttle_range[0]}) must be less than maximum throttle ({self.throttle_range[1]})")
        
        # Compute mass flow rates if not provided
        if self.max_mass_flow_rate is None:
            self.max_mass_flow_rate = self.max_thrust / (self.I_sp * self.g_0)
        
        if self.min_mass_flow_rate is None:
            self.min_mass_flow_rate = self.min_thrust / (self.I_sp * self.g_0)
        
        # Validate mass flow rates
        if self.max_mass_flow_rate <= 0:
            raise ValueError(f"max_mass_flow_rate must be positive, got {self.max_mass_flow_rate}")
        if self.min_mass_flow_rate <= 0:
            raise ValueError(f"min_mass_flow_rate must be positive, got {self.min_mass_flow_rate}")
        if self.min_mass_flow_rate >= self.max_mass_flow_rate:
            raise ValueError(f"min_mass_flow_rate ({self.min_mass_flow_rate}) must be less than max_mass_flow_rate ({self.max_mass_flow_rate})")
        
        # Validate efficiency
        if self.engine_efficiency <= 0 or self.engine_efficiency > 1:
            raise ValueError(f"engine_efficiency must be between 0 and 1, got {self.engine_efficiency}")
    
    @property
    def effective_exhaust_velocity(self) -> float:
        """Effective exhaust velocity (m/s)."""
        return self.I_sp * self.g_0
    
    @property
    def thrust_to_weight_ratio(self) -> float:
        """Thrust-to-weight ratio (using max thrust and g_0)."""
        return self.max_thrust / (1.0 * self.g_0)  # For 1kg mass
    
    @property
    def mass_flow_rate_range(self) -> Tuple[float, float]:
        """Mass flow rate range (kg/s)."""
        return (self.min_mass_flow_rate, self.max_mass_flow_rate)
    
    @property
    def thrust_range(self) -> Tuple[float, float]:
        """Thrust range (N)."""
        return (self.min_thrust, self.max_thrust)
    
    def compute_mass_flow_rate(self, thrust: float) -> float:
        """
        Compute mass flow rate for a given thrust.
        
        Args:
            thrust: Thrust magnitude (N)
            
        Returns:
            Mass flow rate (kg/s)
            
        Raises:
            ValueError: If thrust is outside valid range
        """
        if thrust < self.min_thrust or thrust > self.max_thrust:
            raise ValueError(f"Thrust {thrust}N is outside valid range [{self.min_thrust}, {self.max_thrust}]N")
        
        # Linear interpolation between min and max mass flow rates
        thrust_fraction = (thrust - self.min_thrust) / (self.max_thrust - self.min_thrust)
        mass_flow = self.min_mass_flow_rate + thrust_fraction * (self.max_mass_flow_rate - self.min_mass_flow_rate)
        
        return mass_flow * self.engine_efficiency
    
    def compute_thrust_from_mass_flow(self, mass_flow: float) -> float:
        """
        Compute thrust for a given mass flow rate.
        
        Args:
            mass_flow: Mass flow rate (kg/s)
            
        Returns:
            Thrust magnitude (N)
            
        Raises:
            ValueError: If mass flow is outside valid range
        """
        effective_mass_flow = mass_flow / self.engine_efficiency
        
        if effective_mass_flow < self.min_mass_flow_rate or effective_mass_flow > self.max_mass_flow_rate:
            raise ValueError(f"Mass flow {mass_flow}kg/s is outside valid range [{self.min_mass_flow_rate}, {self.max_mass_flow_rate}]kg/s")
        
        # Linear interpolation between min and max thrust
        mass_flow_fraction = (effective_mass_flow - self.min_mass_flow_rate) / (self.max_mass_flow_rate - self.min_mass_flow_rate)
        thrust = self.min_thrust + mass_flow_fraction * (self.max_thrust - self.min_thrust)
        
        return thrust
    
    def compute_thrust_from_throttle(self, throttle: float) -> float:
        """
        Compute thrust for a given throttle setting.
        
        Args:
            throttle: Throttle setting (0-1)
            
        Returns:
            Thrust magnitude (N)
            
        Raises:
            ValueError: If throttle is outside valid range
        """
        if throttle < self.throttle_range[0] or throttle > self.throttle_range[1]:
            raise ValueError(f"Throttle {throttle} is outside valid range {self.throttle_range}")
        
        # Linear interpolation between min and max thrust based on throttle
        throttle_fraction = (throttle - self.throttle_range[0]) / (self.throttle_range[1] - self.throttle_range[0])
        thrust = self.min_thrust + throttle_fraction * (self.max_thrust - self.min_thrust)
        
        return thrust
    
    def compute_throttle_from_thrust(self, thrust: float) -> float:
        """
        Compute throttle setting for a given thrust.
        
        Args:
            thrust: Thrust magnitude (N)
            
        Returns:
            Throttle setting (0-1)
            
        Raises:
            ValueError: If thrust is outside valid range
        """
        if thrust < self.min_thrust or thrust > self.max_thrust:
            raise ValueError(f"Thrust {thrust}N is outside valid range [{self.min_thrust}, {self.max_thrust}]N")
        
        # Linear interpolation between min and max throttle
        thrust_fraction = (thrust - self.min_thrust) / (self.max_thrust - self.min_thrust)
        throttle = self.throttle_range[0] + thrust_fraction * (self.throttle_range[1] - self.throttle_range[0])
        
        return throttle
    
    def validate_thrust_profile(self, thrust_profile: np.ndarray, time_vector: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Validate a thrust profile against engine constraints.
        
        Args:
            thrust_profile: Array of thrust magnitudes (N)
            time_vector: Optional time vector for rate of change checks
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'valid': True,
            'max_thrust_violation': 0.0,
            'min_thrust_violation': 0.0,
            'max_thrust_exceeded': False,
            'min_thrust_violated': False,
            'thrust_rate_of_change': None,
            'max_thrust_rate_of_change': None
        }
        
        # Check thrust bounds
        max_thrust = np.max(thrust_profile)
        min_thrust = np.min(thrust_profile)
        
        if max_thrust > self.max_thrust:
            results['max_thrust_violation'] = max_thrust - self.max_thrust
            results['max_thrust_exceeded'] = True
            results['valid'] = False
        
        if min_thrust < self.min_thrust:
            results['min_thrust_violation'] = self.min_thrust - min_thrust
            results['min_thrust_violated'] = True
            results['valid'] = False
        
        # Check rate of change if time vector is provided
        if time_vector is not None and len(time_vector) == len(thrust_profile):
            dt = np.diff(time_vector)
            if len(dt) > 0 and np.all(dt > 0):  # Valid time vector
                thrust_rate = np.diff(thrust_profile) / dt
                results['thrust_rate_of_change'] = thrust_rate
                results['max_thrust_rate_of_change'] = np.max(np.abs(thrust_rate))
        
        return results
    
    def compute_propellant_consumption(self, thrust_profile: np.ndarray, time_vector: np.ndarray) -> float:
        """
        Compute total propellant consumption for a thrust profile.
        
        Args:
            thrust_profile: Array of thrust magnitudes (N)
            time_vector: Time vector (s)
            
        Returns:
            Total propellant mass consumed (kg)
        """
        if len(thrust_profile) != len(time_vector):
            raise ValueError(f"thrust_profile length ({len(thrust_profile)}) must match time_vector length ({len(time_vector)})")
        
        # Compute mass flow rate at each point
        mass_flow_rates = np.array([self.compute_mass_flow_rate(thrust) for thrust in thrust_profile])
        
        # Integrate using trapezoidal rule
        dt = np.diff(time_vector)
        avg_mass_flow = 0.5 * (mass_flow_rates[:-1] + mass_flow_rates[1:])
        propellant_used = np.sum(avg_mass_flow * dt)
        
        return propellant_used
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert engine parameters to dictionary."""
        return {
            'name': self.name,
            'I_sp': self.I_sp,
            'g_0': self.g_0,
            'max_thrust': self.max_thrust,
            'min_thrust': self.min_thrust,
            'max_mass_flow_rate': self.max_mass_flow_rate,
            'min_mass_flow_rate': self.min_mass_flow_rate,
            'throttle_range': list(self.throttle_range),
            'engine_efficiency': self.engine_efficiency,
            'structural_factor': self.structural_factor,
            'description': self.description,
            'effective_exhaust_velocity': self.effective_exhaust_velocity,
            'thrust_to_weight_ratio': self.thrust_to_weight_ratio
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EngineParameters':
        """Create EngineParameters from dictionary."""
        # Handle throttle_range conversion
        if 'throttle_range' in data and isinstance(data['throttle_range'], list):
            data['throttle_range'] = tuple(data['throttle_range'])
        
        return cls(**data)
    
    def __str__(self) -> str:
        """String representation of engine parameters."""
        desc = f"Engine: {self.name}\n"
        desc += f"  Thrust range: {self.min_thrust:.1f} - {self.max_thrust:.1f} N\n"
        desc += f"  I_sp: {self.I_sp:.1f} s\n"
        desc += f"  Mass flow: {self.min_mass_flow_rate:.6f} - {self.max_mass_flow_rate:.6f} kg/s\n"
        desc += f"  Exhaust velocity: {self.effective_exhaust_velocity:.1f} m/s\n"
        desc += f"  Throttle range: {self.throttle_range[0]:.2f} - {self.throttle_range[1]:.2f}\n"
        if self.description:
            desc += f"  Description: {self.description}"
        return desc


def create_full_thrust_engine() -> EngineParameters:
    """
    Create engine parameters for the full-thrust configuration from the paper.
    
    Returns:
        EngineParameters for full-thrust configuration (80N max, 20N min)
    """
    return EngineParameters(
        name="FullThrustEngine",
        I_sp=225.0,
        max_thrust=80.0,
        min_thrust=20.0,
        throttle_range=(0.25, 1.0),  # 25% to 100%
        description="Full thrust configuration from paper: 80N max, 20N min, I_sp=225s"
    )


def create_quarter_thrust_engine() -> EngineParameters:
    """
    Create engine parameters for the quarter-thrust configuration from the paper.
    
    Returns:
        EngineParameters for quarter-thrust configuration (20N max, 5N min)
    """
    return EngineParameters(
        name="QuarterThrustEngine",
        I_sp=225.0,
        max_thrust=20.0,
        min_thrust=5.0,
        throttle_range=(0.25, 1.0),  # 25% to 100%
        description="Quarter thrust configuration from paper: 20N max, 5N min, I_sp=225s"
    )


def create_mars_lander_engine() -> EngineParameters:
    """
    Create engine parameters for a Mars lander baseline.
    
    Returns:
        EngineParameters for Mars lander
    """
    return EngineParameters(
        name="MarsLanderEngine",
        I_sp=300.0,  # Higher I_sp for Mars lander
        max_thrust=100.0,
        min_thrust=25.0,
        throttle_range=(0.25, 1.0),
        description="Mars lander engine for baseline comparison"
    )


def get_engine_by_name(name: str) -> EngineParameters:
    """
    Get a predefined engine configuration by name.
    
    Args:
        name: Engine name ('full', 'quarter', 'mars')
        
    Returns:
        EngineParameters instance
        
    Raises:
        ValueError: If engine name is not recognized
    """
    engines = {
        'full': create_full_thrust_engine,
        'quarter': create_quarter_thrust_engine,
        'mars': create_mars_lander_engine,
        'full_thrust': create_full_thrust_engine,
        'quarter_thrust': create_quarter_thrust_engine,
        'mars_lander': create_mars_lander_engine,
    }
    
    if name.lower() not in engines:
        raise ValueError(f"Unknown engine name: {name}. Available: {list(engines.keys())}")
    
    return engines[name.lower()]()


def test_engine_module() -> bool:
    """
    Test the engine module.
    
    Returns:
        True if all tests pass
    """
    logger.info("Testing engine module...")
    
    try:
        # Test full thrust engine
        full_engine = create_full_thrust_engine()
        assert full_engine.name == "FullThrustEngine"
        assert full_engine.max_thrust == 80.0
        assert full_engine.min_thrust == 20.0
        assert full_engine.I_sp == 225.0
        
        # Test quarter thrust engine
        quarter_engine = create_quarter_thrust_engine()
        assert quarter_engine.name == "QuarterThrustEngine"
        assert quarter_engine.max_thrust == 20.0
        assert quarter_engine.min_thrust == 5.0
        
        # Test mass flow rate computation
        mass_flow = full_engine.compute_mass_flow_rate(50.0)
        assert mass_flow > 0
        
        # Test thrust from mass flow
        thrust = full_engine.compute_thrust_from_mass_flow(mass_flow)
        assert abs(thrust - 50.0) < 1e-6
        
        # Test throttle computations
        throttle = full_engine.compute_throttle_from_thrust(50.0)
        assert 0.25 <= throttle <= 1.0
        
        thrust_from_throttle = full_engine.compute_thrust_from_throttle(throttle)
        assert abs(thrust_from_throttle - 50.0) < 1e-6
        
        # Test validation
        thrust_profile = np.array([30.0, 40.0, 50.0, 60.0, 70.0])
        time_vector = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        validation = full_engine.validate_thrust_profile(thrust_profile, time_vector)
        assert validation['valid'] == True
        
        # Test propellant consumption
        propellant = full_engine.compute_propellant_consumption(thrust_profile, time_vector)
        assert propellant > 0
        
        # Test dictionary conversion
        engine_dict = full_engine.to_dict()
        engine_from_dict = EngineParameters.from_dict(engine_dict)
        assert engine_from_dict.name == full_engine.name
        assert engine_from_dict.max_thrust == full_engine.max_thrust
        
        logger.info("All engine module tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"Engine module test failed: {e}")
        return False


if __name__ == "__main__":
    # Run tests if module is executed directly
    import sys
    logging.basicConfig(level=logging.INFO)
    
    if test_engine_module():
        print("Engine module tests passed!")
        sys.exit(0)
    else:
        print("Engine module tests failed!")
        sys.exit(1)