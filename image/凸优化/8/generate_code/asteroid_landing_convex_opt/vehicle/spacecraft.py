"""
Spacecraft model for asteroid landing optimization.

This module defines the spacecraft parameters including mass, specific impulse,
thrust bounds, and other vehicle-specific properties used in the optimization
problem formulation.

Based on the paper: "Trajectory Design Employing Convex Optimization for 
Landing on Irregularly Shaped Asteroids"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class Spacecraft:
    """
    Spacecraft model containing all vehicle-specific parameters.
    
    This class encapsulates the spacecraft properties needed for the
    optimization problem, including mass, thrust bounds, and specific impulse.
    
    Attributes:
        name: Name of the spacecraft configuration
        m_wet: Wet mass (initial mass with propellant) [kg]
        m_dry: Dry mass (mass without propellant) [kg]
        I_sp: Specific impulse of the engine [s]
        g_0: Standard gravity [m/s²] (default: 9.80665)
        T_max: Maximum thrust magnitude [N]
        T_min: Minimum thrust magnitude [N]
        max_mass_flow_rate: Maximum mass flow rate [kg/s]
        min_mass_flow_rate: Minimum mass flow rate [kg/s]
        engine_efficiency: Engine efficiency factor (0-1)
        structural_factor: Structural mass factor (dry/wet mass ratio)
        description: Optional description of the spacecraft
    """
    
    name: str
    m_wet: float  # kg
    m_dry: float  # kg
    I_sp: float   # s
    g_0: float = 9.80665  # m/s²
    T_max: float  # N
    T_min: float  # N
    max_mass_flow_rate: Optional[float] = None
    min_mass_flow_rate: Optional[float] = None
    engine_efficiency: float = 1.0
    structural_factor: Optional[float] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        """Validate spacecraft parameters and compute derived values."""
        # Validate basic parameters
        if self.m_wet <= 0:
            raise ValueError(f"Wet mass must be positive, got {self.m_wet}")
        if self.m_dry <= 0:
            raise ValueError(f"Dry mass must be positive, got {self.m_dry}")
        if self.m_dry >= self.m_wet:
            raise ValueError(f"Dry mass ({self.m_dry}) must be less than wet mass ({self.m_wet})")
        if self.I_sp <= 0:
            raise ValueError(f"Specific impulse must be positive, got {self.I_sp}")
        if self.T_max <= 0:
            raise ValueError(f"Maximum thrust must be positive, got {self.T_max}")
        if self.T_min <= 0:
            raise ValueError(f"Minimum thrust must be positive, got {self.T_min}")
        if self.T_min > self.T_max:
            raise ValueError(f"Minimum thrust ({self.T_min}) cannot exceed maximum thrust ({self.T_max})")
        if self.g_0 <= 0:
            raise ValueError(f"Standard gravity must be positive, got {self.g_0}")
        if self.engine_efficiency <= 0 or self.engine_efficiency > 1:
            raise ValueError(f"Engine efficiency must be in (0, 1], got {self.engine_efficiency}")
        
        # Compute structural factor if not provided
        if self.structural_factor is None:
            self.structural_factor = self.m_dry / self.m_wet
        
        # Compute mass flow rates if not provided
        if self.max_mass_flow_rate is None:
            self.max_mass_flow_rate = self.T_max / (self.I_sp * self.g_0)
        if self.min_mass_flow_rate is None:
            self.min_mass_flow_rate = self.T_min / (self.I_sp * self.g_0)
        
        # Validate mass flow rates
        if self.max_mass_flow_rate <= 0:
            raise ValueError(f"Maximum mass flow rate must be positive, got {self.max_mass_flow_rate}")
        if self.min_mass_flow_rate <= 0:
            raise ValueError(f"Minimum mass flow rate must be positive, got {self.min_mass_flow_rate}")
        if self.min_mass_flow_rate > self.max_mass_flow_rate:
            raise ValueError(f"Minimum mass flow rate ({self.min_mass_flow_rate}) cannot exceed maximum ({self.max_mass_flow_rate})")
    
    @property
    def propellant_mass(self) -> float:
        """Propellant mass [kg]."""
        return self.m_wet - self.m_dry
    
    @property
    def mass_ratio(self) -> float:
        """Mass ratio (wet/dry)."""
        return self.m_wet / self.m_dry
    
    @property
    def max_acceleration(self) -> float:
        """Maximum acceleration [m/s²] at initial mass."""
        return self.T_max / self.m_wet
    
    @property
    def min_acceleration(self) -> float:
        """Minimum acceleration [m/s²] at initial mass."""
        return self.T_min / self.m_wet
    
    @property
    def max_acceleration_dry(self) -> float:
        """Maximum acceleration [m/s²] at dry mass."""
        return self.T_max / self.m_dry
    
    @property
    def min_acceleration_dry(self) -> float:
        """Minimum acceleration [m/s²] at dry mass."""
        return self.T_min / self.m_dry
    
    @property
    def exhaust_velocity(self) -> float:
        """Exhaust velocity [m/s]."""
        return self.I_sp * self.g_0
    
    @property
    def characteristic_velocity(self) -> float:
        """Characteristic velocity (Δv) for full propellant burn [m/s]."""
        return self.exhaust_velocity * np.log(self.mass_ratio)
    
    def compute_mass_flow_rate(self, thrust_magnitude: float) -> float:
        """
        Compute mass flow rate for a given thrust magnitude.
        
        Args:
            thrust_magnitude: Thrust magnitude [N]
            
        Returns:
            Mass flow rate [kg/s]
        """
        if thrust_magnitude < self.T_min or thrust_magnitude > self.T_max:
            logger.warning(f"Thrust magnitude {thrust_magnitude} outside bounds [{self.T_min}, {self.T_max}]")
        
        return thrust_magnitude / (self.I_sp * self.g_0)
    
    def compute_thrust_from_mass_flow(self, mass_flow_rate: float) -> float:
        """
        Compute thrust magnitude from mass flow rate.
        
        Args:
            mass_flow_rate: Mass flow rate [kg/s]
            
        Returns:
            Thrust magnitude [N]
        """
        if mass_flow_rate < self.min_mass_flow_rate or mass_flow_rate > self.max_mass_flow_rate:
            logger.warning(f"Mass flow rate {mass_flow_rate} outside bounds [{self.min_mass_flow_rate}, {self.max_mass_flow_rate}]")
        
        return mass_flow_rate * self.I_sp * self.g_0
    
    def compute_mass_at_time(self, t: float, initial_mass: Optional[float] = None, 
                           thrust_profile: Optional[np.ndarray] = None) -> float:
        """
        Compute mass at time t given a thrust profile.
        
        Args:
            t: Time [s]
            initial_mass: Initial mass [kg] (default: wet mass)
            thrust_profile: Array of thrust magnitudes at discrete times [N]
            
        Returns:
            Mass at time t [kg]
            
        Note:
            If thrust_profile is not provided, assumes constant thrust at T_min.
        """
        if initial_mass is None:
            initial_mass = self.m_wet
        
        if thrust_profile is None:
            # Assume constant minimum thrust
            mass_flow_rate = self.min_mass_flow_rate
            mass = initial_mass - mass_flow_rate * t
        else:
            # Integrate mass flow over time
            # This is a simplified version - actual implementation would need time vector
            avg_thrust = np.mean(thrust_profile) if len(thrust_profile) > 0 else self.T_min
            mass_flow_rate = avg_thrust / (self.I_sp * self.g_0)
            mass = initial_mass - mass_flow_rate * t
        
        return max(mass, self.m_dry)
    
    def validate_thrust_profile(self, thrust_magnitudes: np.ndarray, 
                              time_vector: Optional[np.ndarray] = None,
                              tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Validate a thrust profile against spacecraft constraints.
        
        Args:
            thrust_magnitudes: Array of thrust magnitudes [N]
            time_vector: Optional time vector [s] (for mass constraint validation)
            tolerance: Numerical tolerance for constraint violations
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'thrust_bounds_satisfied': True,
            'max_thrust_violation': 0.0,
            'min_thrust_violation': 0.0,
            'mass_constraint_satisfied': True,
            'min_mass_violation': 0.0,
            'mass_flow_bounds_satisfied': True,
            'max_mass_flow_violation': 0.0,
            'min_mass_flow_violation': 0.0,
        }
        
        # Check thrust bounds
        if len(thrust_magnitudes) > 0:
            max_thrust = np.max(thrust_magnitudes)
            min_thrust = np.min(thrust_magnitudes)
            
            results['max_thrust_violation'] = max(0.0, max_thrust - self.T_max)
            results['min_thrust_violation'] = max(0.0, self.T_min - min_thrust)
            results['thrust_bounds_satisfied'] = (
                results['max_thrust_violation'] <= tolerance and 
                results['min_thrust_violation'] <= tolerance
            )
        
        # Check mass flow bounds
        if len(thrust_magnitudes) > 0:
            mass_flow_rates = thrust_magnitudes / (self.I_sp * self.g_0)
            max_mass_flow = np.max(mass_flow_rates)
            min_mass_flow = np.min(mass_flow_rates)
            
            results['max_mass_flow_violation'] = max(0.0, max_mass_flow - self.max_mass_flow_rate)
            results['min_mass_flow_violation'] = max(0.0, self.min_mass_flow_rate - min_mass_flow)
            results['mass_flow_bounds_satisfied'] = (
                results['max_mass_flow_violation'] <= tolerance and 
                results['min_mass_flow_violation'] <= tolerance
            )
        
        # Check mass constraint if time vector is provided
        if time_vector is not None and len(thrust_magnitudes) > 0:
            # Simple trapezoidal integration for mass
            dt = time_vector[1] - time_vector[0] if len(time_vector) > 1 else 1.0
            total_mass_flow = np.sum(mass_flow_rates) * dt
            final_mass = self.m_wet - total_mass_flow
            results['min_mass_violation'] = max(0.0, self.m_dry - final_mass)
            results['mass_constraint_satisfied'] = results['min_mass_violation'] <= tolerance
        
        return results
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert spacecraft parameters to dictionary."""
        return {
            'name': self.name,
            'm_wet': self.m_wet,
            'm_dry': self.m_dry,
            'I_sp': self.I_sp,
            'g_0': self.g_0,
            'T_max': self.T_max,
            'T_min': self.T_min,
            'max_mass_flow_rate': self.max_mass_flow_rate,
            'min_mass_flow_rate': self.min_mass_flow_rate,
            'engine_efficiency': self.engine_efficiency,
            'structural_factor': self.structural_factor,
            'description': self.description,
            'propellant_mass': self.propellant_mass,
            'mass_ratio': self.mass_ratio,
            'exhaust_velocity': self.exhaust_velocity,
            'characteristic_velocity': self.characteristic_velocity,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Spacecraft':
        """Create spacecraft from dictionary."""
        # Extract required fields
        required = ['name', 'm_wet', 'm_dry', 'I_sp', 'T_max', 'T_min']
        for field in required:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        # Create spacecraft instance
        return cls(
            name=data['name'],
            m_wet=data['m_wet'],
            m_dry=data['m_dry'],
            I_sp=data['I_sp'],
            g_0=data.get('g_0', 9.80665),
            T_max=data['T_max'],
            T_min=data['T_min'],
            max_mass_flow_rate=data.get('max_mass_flow_rate'),
            min_mass_flow_rate=data.get('min_mass_flow_rate'),
            engine_efficiency=data.get('engine_efficiency', 1.0),
            structural_factor=data.get('structural_factor'),
            description=data.get('description'),
        )
    
    def __str__(self) -> str:
        """String representation of spacecraft."""
        lines = [
            f"Spacecraft: {self.name}",
            f"  Mass: wet={self.m_wet:.2f} kg, dry={self.m_dry:.2f} kg, propellant={self.propellant_mass:.2f} kg",
            f"  Thrust: T_max={self.T_max:.1f} N, T_min={self.T_min:.1f} N",
            f"  Engine: I_sp={self.I_sp:.1f} s, exhaust velocity={self.exhaust_velocity:.1f} m/s",
            f"  Acceleration: max={self.max_acceleration:.4f} m/s², min={self.min_acceleration:.4f} m/s²",
            f"  Mass flow: max={self.max_mass_flow_rate:.6f} kg/s, min={self.min_mass_flow_rate:.6f} kg/s",
        ]
        if self.description:
            lines.append(f"  Description: {self.description}")
        return "\n".join(lines)


# Predefined spacecraft configurations from the paper
def create_full_thrust_spacecraft() -> Spacecraft:
    """
    Create the full-thrust spacecraft configuration from the paper.
    
    From Table 4: m_wet = 500 kg, m_dry = 494.66 kg, I_sp = 225 s,
    T_max = 80 N, T_min = 20 N.
    
    Returns:
        Spacecraft instance for full-thrust configuration
    """
    return Spacecraft(
        name="FullThrust",
        m_wet=500.0,      # kg
        m_dry=494.66,     # kg
        I_sp=225.0,       # s
        T_max=80.0,       # N
        T_min=20.0,       # N
        description="Full thrust configuration from paper (80N max, 20N min)"
    )


def create_quarter_thrust_spacecraft() -> Spacecraft:
    """
    Create the quarter-thrust spacecraft configuration from the paper.
    
    From Table 4: m_wet = 500 kg, m_dry = 496.6 kg, I_sp = 225 s,
    T_max = 20 N, T_min = 5 N.
    
    Returns:
        Spacecraft instance for quarter-thrust configuration
    """
    return Spacecraft(
        name="QuarterThrust",
        m_wet=500.0,      # kg
        m_dry=496.6,      # kg
        I_sp=225.0,       # s
        T_max=20.0,       # N
        T_min=5.0,        # N
        description="Quarter thrust configuration from paper (20N max, 5N min)"
    )


def create_mars_lander_spacecraft() -> Spacecraft:
    """
    Create a Mars lander spacecraft for baseline comparison.
    
    Based on typical Mars lander parameters for comparison with asteroid landing.
    
    Returns:
        Spacecraft instance for Mars lander
    """
    return Spacecraft(
        name="MarsLander",
        m_wet=1000.0,     # kg
        m_dry=800.0,      # kg
        I_sp=300.0,       # s
        T_max=5000.0,     # N
        T_min=1000.0,     # N
        description="Typical Mars lander for baseline comparison"
    )


def get_spacecraft_by_name(name: str) -> Spacecraft:
    """
    Get a predefined spacecraft configuration by name.
    
    Args:
        name: Spacecraft name ("FullThrust", "QuarterThrust", "MarsLander")
        
    Returns:
        Spacecraft instance
        
    Raises:
        ValueError: If spacecraft name is not recognized
    """
    spacecraft_map = {
        "FullThrust": create_full_thrust_spacecraft,
        "QuarterThrust": create_quarter_thrust_spacecraft,
        "MarsLander": create_mars_lander_spacecraft,
    }
    
    if name not in spacecraft_map:
        raise ValueError(f"Unknown spacecraft name: {name}. Available: {list(spacecraft_map.keys())}")
    
    return spacecraft_map[name]()


def test_spacecraft_module() -> bool:
    """
    Test the spacecraft module.
    
    Returns:
        True if all tests pass
    """
    try:
        # Test full thrust spacecraft
        full = create_full_thrust_spacecraft()
        assert full.name == "FullThrust"
        assert abs(full.m_wet - 500.0) < 1e-6
        assert abs(full.m_dry - 494.66) < 1e-6
        assert abs(full.T_max - 80.0) < 1e-6
        assert abs(full.T_min - 20.0) < 1e-6
        assert abs(full.I_sp - 225.0) < 1e-6
        
        # Test quarter thrust spacecraft
        quarter = create_quarter_thrust_spacecraft()
        assert quarter.name == "QuarterThrust"
        assert abs(quarter.T_max - 20.0) < 1e-6
        assert abs(quarter.T_min - 5.0) < 1e-6
        
        # Test property calculations
        assert full.propellant_mass > 0
        assert full.mass_ratio > 1.0
        assert full.max_acceleration > 0
        assert full.min_acceleration > 0
        assert full.exhaust_velocity > 0
        
        # Test validation
        thrust_profile = np.array([30.0, 40.0, 50.0, 60.0, 70.0])
        validation = full.validate_thrust_profile(thrust_profile)
        assert validation['thrust_bounds_satisfied'] == True
        
        # Test invalid thrust (should still work with warning)
        invalid_thrust = np.array([10.0, 90.0])  # Outside bounds
        validation = full.validate_thrust_profile(invalid_thrust)
        assert validation['thrust_bounds_satisfied'] == False
        
        # Test dictionary conversion
        data = full.to_dict()
        full2 = Spacecraft.from_dict(data)
        assert abs(full2.m_wet - full.m_wet) < 1e-6
        assert abs(full2.T_max - full.T_max) < 1e-6
        
        # Test get by name
        spacecraft = get_spacecraft_by_name("FullThrust")
        assert spacecraft.name == "FullThrust"
        
        print("All spacecraft module tests passed!")
        return True
        
    except Exception as e:
        print(f"Spacecraft module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run tests if module is executed directly
    test_spacecraft_module()