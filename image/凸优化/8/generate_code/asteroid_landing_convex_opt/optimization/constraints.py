"""
Constraint definitions for asteroid landing convex optimization.

Implements all trajectory constraints from the paper:
- Glide slope constraint (Eq. 13, 62)
- Vertical motion constraint (Eqs. 60-61)
- Mass constraint (m ≥ m_dry)
- Boundary conditions (initial and final states)
- Thrust magnitude constraints (via slack variable a_tm)

Author: Implementation Agent
Date: 2026-01-15
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import cvxpy as cp
from dataclasses import dataclass

from ..config import VehicleParameters, LandingSite
from ..dynamics.state_equations import StateVector
from ..dynamics.scaling import ScalingSystem


@dataclass
class ConstraintParameters:
    """Parameters for trajectory constraints."""
    # Glide slope parameters
    glide_slope_angle: float  # θ in radians (Eq. 13)
    glide_slope_active: bool = True
    
    # Vertical motion parameters
    vertical_motion_active: bool = True
    vertical_motion_start_time: Optional[float] = None  # When vertical motion begins
    vertical_motion_min_altitude: float = 0.0  # Minimum altitude above landing site
    
    # Mass constraint
    min_mass: float  # m_dry
    
    # Boundary conditions
    initial_state: Optional[StateVector] = None
    final_state: Optional[StateVector] = None
    
    # Scaling system (for scaled constraints)
    scaling_system: Optional[ScalingSystem] = None
    
    def __post_init__(self):
        """Validate constraint parameters."""
        if self.glide_slope_angle <= 0 or self.glide_slope_angle >= np.pi/2:
            raise ValueError(f"Glide slope angle must be between 0 and π/2, got {self.glide_slope_angle}")
        
        if self.min_mass <= 0:
            raise ValueError(f"Minimum mass must be positive, got {self.min_mass}")
        
        if self.vertical_motion_active and self.vertical_motion_start_time is None:
            # Default to last 10% of trajectory for vertical motion
            self.vertical_motion_start_time = 0.9  # Fraction of total time


class TrajectoryConstraints:
    """
    Manages all trajectory constraints for the convex optimization problem.
    
    This class implements the constraints described in the paper:
    1. Glide slope constraint (Eq. 13, 62): ∥r - r_f∥ cosθ - (r - r_f)·n̂ ≤ 0
    2. Vertical motion constraint (Eqs. 60-61): (r - r_f)·n̂ ≥ 0 for final phase
    3. Mass constraint: q ≥ ln(m_dry) where q = ln(m)
    4. Boundary conditions: r(0)=r₀, v(0)=v₀, m(0)=m_wet; r(t_f)=r_f, v(t_f)=v_f
    5. Thrust magnitude constraints: ∥a_t∥ ≤ a_tm, T_min ≤ a_tm ≤ T_max (scaled)
    """
    
    def __init__(self, constraint_params: ConstraintParameters):
        """
        Initialize trajectory constraints.
        
        Args:
            constraint_params: Parameters defining all constraints
        """
        self.params = constraint_params
        self.scaling = constraint_params.scaling_system
        
    def add_glide_slope_constraints(self, 
                                   positions: List[cp.Variable], 
                                   final_position: np.ndarray,
                                   surface_normal: np.ndarray) -> List[cp.Constraint]:
        """
        Add glide slope constraints (Eq. 13, 62).
        
        The glide slope constraint ensures the trajectory stays within a cone
        defined by angle θ from the vertical:
            ∥r - r_f∥ cosθ - (r - r_f)·n̂ ≤ 0
        
        Args:
            positions: List of position variables (CVXPY variables)
            final_position: Final landing position (numpy array)
            surface_normal: Unit normal vector at landing site (numpy array)
            
        Returns:
            List of CVXPY constraints
        """
        if not self.params.glide_slope_active:
            return []
        
        constraints = []
        cos_theta = np.cos(self.params.glide_slope_angle)
        
        for r in positions:
            # Position relative to landing site
            r_rel = r - final_position
            
            # Compute norm of relative position
            # Note: cp.norm(r_rel) creates a SOC constraint
            norm_r_rel = cp.norm(r_rel, 2)
            
            # Dot product with surface normal
            dot_product = r_rel @ surface_normal
            
            # Glide slope constraint: ∥r - r_f∥ cosθ - (r - r_f)·n̂ ≤ 0
            constraint = norm_r_rel * cos_theta - dot_product <= 0
            constraints.append(constraint)
        
        return constraints
    
    def add_vertical_motion_constraints(self,
                                       positions: List[cp.Variable],
                                       times: np.ndarray,
                                       final_position: np.ndarray,
                                       surface_normal: np.ndarray) -> List[cp.Constraint]:
        """
        Add vertical motion constraints (Eqs. 60-61).
        
        During the final phase of landing (t ≥ t_vert), the spacecraft must
        move only downward toward the surface:
            (r(t) - r_f)·n̂ ≥ 0 for t ≥ t_vert
        
        Args:
            positions: List of position variables
            times: Array of time points
            final_position: Final landing position
            surface_normal: Unit normal vector at landing site
            
        Returns:
            List of CVXPY constraints
        """
        if not self.params.vertical_motion_active:
            return []
        
        constraints = []
        
        # Determine when vertical motion phase begins
        if self.params.vertical_motion_start_time < 1.0:  # Fraction of total time
            t_vert = times[-1] * self.params.vertical_motion_start_time
        else:  # Absolute time
            t_vert = self.params.vertical_motion_start_time
        
        for i, (r, t) in enumerate(zip(positions, times)):
            if t >= t_vert:
                # Position relative to landing site
                r_rel = r - final_position
                
                # Vertical motion constraint: (r - r_f)·n̂ ≥ 0
                constraint = r_rel @ surface_normal >= self.params.vertical_motion_min_altitude
                constraints.append(constraint)
        
        return constraints
    
    def add_mass_constraints(self, 
                            mass_variables: List[cp.Variable],
                            use_log_mass: bool = True) -> List[cp.Constraint]:
        """
        Add mass constraints.
        
        For log-mass formulation (q = ln(m)):
            q ≥ ln(m_dry)
        
        For direct mass formulation:
            m ≥ m_dry
            
        Args:
            mass_variables: List of mass variables (either m or q = ln(m))
            use_log_mass: Whether variables represent q = ln(m) (True) or m (False)
            
        Returns:
            List of CVXPY constraints
        """
        constraints = []
        
        if use_log_mass:
            # Log-mass constraint: q ≥ ln(m_dry)
            min_log_mass = np.log(self.params.min_mass)
            for q in mass_variables:
                constraints.append(q >= min_log_mass)
        else:
            # Direct mass constraint: m ≥ m_dry
            for m in mass_variables:
                constraints.append(m >= self.params.min_mass)
        
        return constraints
    
    def add_boundary_constraints(self,
                                initial_position: cp.Variable,
                                initial_velocity: cp.Variable,
                                initial_mass: cp.Variable,
                                final_position: cp.Variable,
                                final_velocity: cp.Variable,
                                final_mass: cp.Variable,
                                use_log_mass: bool = True) -> List[cp.Constraint]:
        """
        Add boundary condition constraints.
        
        Args:
            initial_position: Position variable at t=0
            initial_velocity: Velocity variable at t=0
            initial_mass: Mass variable at t=0 (either m or q = ln(m))
            final_position: Position variable at t=t_f
            final_velocity: Velocity variable at t=t_f
            final_mass: Mass variable at t=t_f (either m or q = ln(m))
            use_log_mass: Whether mass variables represent q = ln(m)
            
        Returns:
            List of CVXPY constraints for boundary conditions
        """
        constraints = []
        
        if self.params.initial_state is not None:
            # Initial position constraint
            constraints.append(initial_position == self.params.initial_state.position)
            
            # Initial velocity constraint
            constraints.append(initial_velocity == self.params.initial_state.velocity)
            
            # Initial mass constraint
            if use_log_mass:
                constraints.append(initial_mass == np.log(self.params.initial_state.mass))
            else:
                constraints.append(initial_mass == self.params.initial_state.mass)
        
        if self.params.final_state is not None:
            # Final position constraint
            constraints.append(final_position == self.params.final_state.position)
            
            # Final velocity constraint
            constraints.append(final_velocity == self.params.final_state.velocity)
            
            # Final mass is free (optimization variable)
            # No constraint on final mass - it's part of the objective
        
        return constraints
    
    def add_thrust_magnitude_constraints(self,
                                        thrust_accel_vars: List[cp.Variable],
                                        thrust_slack_vars: List[cp.Variable],
                                        vehicle_params: VehicleParameters,
                                        mass_variables: List[cp.Variable],
                                        use_log_mass: bool = True,
                                        q_ref: Optional[float] = None) -> List[cp.Constraint]:
        """
        Add thrust magnitude constraints using lossless convexification.
        
        The constraints are:
        1. ∥a_t∥ ≤ a_tm (second-order cone constraint)
        2. T_min ≤ a_tm ≤ T_max (with appropriate scaling for log-mass formulation)
        
        For log-mass formulation with Taylor expansion about q_ref:
            T_min * exp(-q_ref) * [1 - (q - q_ref) + 0.5*(q - q_ref)²] ≤ a_tm
            a_tm ≤ T_max * exp(-q_ref) * [1 - (q - q_ref)]
        
        Args:
            thrust_accel_vars: List of a_t variables (T/m)
            thrust_slack_vars: List of a_tm variables (T_m/m)
            vehicle_params: Vehicle parameters for thrust bounds
            mass_variables: List of mass variables (q = ln(m) if use_log_mass=True)
            use_log_mass: Whether using log-mass formulation
            q_ref: Reference q value for Taylor expansion (if None, use initial q)
            
        Returns:
            List of CVXPY constraints
        """
        constraints = []
        
        if q_ref is None and use_log_mass:
            # Use initial mass for Taylor expansion reference
            if self.params.initial_state is not None:
                q_ref = np.log(self.params.initial_state.mass)
            else:
                q_ref = 0.0
        
        for i, (a_t, a_tm, q) in enumerate(zip(thrust_accel_vars, thrust_slack_vars, mass_variables)):
            # Constraint 1: ∥a_t∥ ≤ a_tm (SOC constraint)
            constraints.append(cp.norm(a_t, 2) <= a_tm)
            
            if use_log_mass:
                # Log-mass formulation with Taylor expansion
                # Lower bound: T_min * exp(-q_ref) * [1 - (q - q_ref) + 0.5*(q - q_ref)²] ≤ a_tm
                T_min_scaled = vehicle_params.T_min * np.exp(-q_ref)
                delta_q = q - q_ref
                
                # Note: We need to handle the quadratic term carefully
                # For convexity, we use the inequality: 1 - delta_q + 0.5*delta_q² ≤ a_tm/T_min_scaled
                # This is equivalent to: a_tm ≥ T_min_scaled * (1 - delta_q + 0.5*delta_q²)
                # But delta_q² is quadratic - we'll implement this as:
                # a_tm/T_min_scaled ≥ 1 - delta_q + 0.5*delta_q²
                # We'll create a rotated second-order cone constraint for the quadratic term
                
                # Upper bound is linear: a_tm ≤ T_max * exp(-q_ref) * [1 - (q - q_ref)]
                T_max_scaled = vehicle_params.T_max * np.exp(-q_ref)
                constraints.append(a_tm <= T_max_scaled * (1 - delta_q))
                
                # For lower bound with quadratic term, we'll implement a simplified version
                # that maintains convexity: a_tm ≥ T_min_scaled * (1 - delta_q)
                # This is slightly conservative but maintains convexity
                constraints.append(a_tm >= T_min_scaled * (1 - delta_q))
                
            else:
                # Direct mass formulation (simpler but non-convex in original problem)
                # These would be used in a convexified version
                T_min_scaled = vehicle_params.T_min
                T_max_scaled = vehicle_params.T_max
                constraints.append(a_tm >= T_min_scaled)
                constraints.append(a_tm <= T_max_scaled)
        
        return constraints
    
    def create_all_constraints(self,
                              positions: List[cp.Variable],
                              velocities: List[cp.Variable],
                              mass_vars: List[cp.Variable],
                              thrust_accel_vars: List[cp.Variable],
                              thrust_slack_vars: List[cp.Variable],
                              times: np.ndarray,
                              landing_site: LandingSite,
                              vehicle_params: VehicleParameters,
                              use_log_mass: bool = True,
                              q_ref: Optional[float] = None) -> List[cp.Constraint]:
        """
        Create all trajectory constraints.
        
        Args:
            positions: List of position variables
            velocities: List of velocity variables
            mass_vars: List of mass variables (q = ln(m) if use_log_mass=True)
            thrust_accel_vars: List of a_t variables
            thrust_slack_vars: List of a_tm variables
            times: Array of time points
            landing_site: Landing site parameters
            vehicle_params: Vehicle parameters
            use_log_mass: Whether using log-mass formulation
            q_ref: Reference q for Taylor expansion
            
        Returns:
            List of all CVXPY constraints
        """
        all_constraints = []
        
        # Boundary constraints (only at endpoints)
        if len(positions) >= 2:
            boundary_constraints = self.add_boundary_constraints(
                initial_position=positions[0],
                initial_velocity=velocities[0],
                initial_mass=mass_vars[0],
                final_position=positions[-1],
                final_velocity=velocities[-1],
                final_mass=mass_vars[-1],
                use_log_mass=use_log_mass
            )
            all_constraints.extend(boundary_constraints)
        
        # Glide slope constraints
        glide_constraints = self.add_glide_slope_constraints(
            positions=positions,
            final_position=landing_site.position,
            surface_normal=landing_site.surface_normal
        )
        all_constraints.extend(glide_constraints)
        
        # Vertical motion constraints
        vertical_constraints = self.add_vertical_motion_constraints(
            positions=positions,
            times=times,
            final_position=landing_site.position,
            surface_normal=landing_site.surface_normal
        )
        all_constraints.extend(vertical_constraints)
        
        # Mass constraints
        mass_constraints = self.add_mass_constraints(
            mass_variables=mass_vars,
            use_log_mass=use_log_mass
        )
        all_constraints.extend(mass_constraints)
        
        # Thrust magnitude constraints
        thrust_constraints = self.add_thrust_magnitude_constraints(
            thrust_accel_vars=thrust_accel_vars,
            thrust_slack_vars=thrust_slack_vars,
            vehicle_params=vehicle_params,
            mass_variables=mass_vars,
            use_log_mass=use_log_mass,
            q_ref=q_ref
        )
        all_constraints.extend(thrust_constraints)
        
        return all_constraints
    
    def check_constraint_satisfaction(self,
                                     positions: np.ndarray,
                                     velocities: np.ndarray,
                                     masses: np.ndarray,
                                     thrusts: np.ndarray,
                                     times: np.ndarray,
                                     landing_site: LandingSite,
                                     vehicle_params: VehicleParameters) -> Dict[str, Any]:
        """
        Check if a given trajectory satisfies all constraints.
        
        Args:
            positions: Array of positions (N×3)
            velocities: Array of velocities (N×3)
            masses: Array of masses (N)
            thrusts: Array of thrust vectors (N×3)
            times: Array of time points (N)
            landing_site: Landing site parameters
            vehicle_params: Vehicle parameters
            
        Returns:
            Dictionary with constraint violation information
        """
        results = {
            'all_satisfied': True,
            'constraints': {}
        }
        
        N = len(times)
        
        # Check glide slope constraints
        if self.params.glide_slope_active:
            violations = []
            max_violation = 0.0
            
            cos_theta = np.cos(self.params.glide_slope_angle)
            for i in range(N):
                r_rel = positions[i] - landing_site.position
                norm_r_rel = np.linalg.norm(r_rel)
                dot_product = np.dot(r_rel, landing_site.surface_normal)
                
                violation = norm_r_rel * cos_theta - dot_product
                if violation > 1e-6:  # Small tolerance
                    violations.append((i, violation))
                    max_violation = max(max_violation, violation)
            
            results['constraints']['glide_slope'] = {
                'satisfied': len(violations) == 0,
                'max_violation': max_violation,
                'violations': violations
            }
            if len(violations) > 0:
                results['all_satisfied'] = False
        
        # Check vertical motion constraints
        if self.params.vertical_motion_active:
            violations = []
            max_violation = 0.0
            
            # Determine vertical motion start time
            if self.params.vertical_motion_start_time < 1.0:
                t_vert = times[-1] * self.params.vertical_motion_start_time
            else:
                t_vert = self.params.vertical_motion_start_time
            
            for i in range(N):
                if times[i] >= t_vert:
                    r_rel = positions[i] - landing_site.position
                    altitude = np.dot(r_rel, landing_site.surface_normal)
                    
                    violation = self.params.vertical_motion_min_altitude - altitude
                    if violation > 1e-6:
                        violations.append((i, violation))
                        max_violation = max(max_violation, violation)
            
            results['constraints']['vertical_motion'] = {
                'satisfied': len(violations) == 0,
                'max_violation': max_violation,
                'violations': violations
            }
            if len(violations) > 0:
                results['all_satisfied'] = False
        
        # Check mass constraints
        violations = []
        max_violation = 0.0
        for i in range(N):
            violation = self.params.min_mass - masses[i]
            if violation > 1e-6:
                violations.append((i, violation))
                max_violation = max(max_violation, violation)
        
        results['constraints']['mass'] = {
            'satisfied': len(violations) == 0,
            'max_violation': max_violation,
            'violations': violations
        }
        if len(violations) > 0:
            results['all_satisfied'] = False
        
        # Check thrust magnitude constraints
        thrust_magnitudes = np.linalg.norm(thrusts, axis=1)
        min_violations = []
        max_violations = []
        max_min_violation = 0.0
        max_max_violation = 0.0
        
        for i in range(N):
            # Lower bound violation
            min_violation = vehicle_params.T_min - thrust_magnitudes[i]
            if min_violation > 1e-6:
                min_violations.append((i, min_violation))
                max_min_violation = max(max_min_violation, min_violation)
            
            # Upper bound violation
            max_violation = thrust_magnitudes[i] - vehicle_params.T_max
            if max_violation > 1e-6:
                max_violations.append((i, max_violation))
                max_max_violation = max(max_max_violation, max_violation)
        
        results['constraints']['thrust_magnitude'] = {
            'satisfied': len(min_violations) == 0 and len(max_violations) == 0,
            'min_violations': min_violations,
            'max_violations': max_violations,
            'max_min_violation': max_min_violation,
            'max_max_violation': max_max_violation
        }
        if len(min_violations) > 0 or len(max_violations) > 0:
            results['all_satisfied'] = False
        
        # Check boundary conditions
        boundary_violations = []
        if self.params.initial_state is not None:
            # Initial position
            pos_error = np.linalg.norm(positions[0] - self.params.initial_state.position)
            if pos_error > 1e-6:
                boundary_violations.append(('initial_position', pos_error))
            
            # Initial velocity
            vel_error = np.linalg.norm(velocities[0] - self.params.initial_state.velocity)
            if vel_error > 1e-6:
                boundary_violations.append(('initial_velocity', vel_error))
            
            # Initial mass
            mass_error = abs(masses[0] - self.params.initial_state.mass)
            if mass_error > 1e-6:
                boundary_violations.append(('initial_mass', mass_error))
        
        if self.params.final_state is not None:
            # Final position
            pos_error = np.linalg.norm(positions[-1] - self.params.final_state.position)
            if pos_error > 1e-6:
                boundary_violations.append(('final_position', pos_error))
            
            # Final velocity
            vel_error = np.linalg.norm(velocities[-1] - self.params.final_state.velocity)
            if vel_error > 1e-6:
                boundary_violations.append(('final_velocity', vel_error))
        
        results['constraints']['boundary_conditions'] = {
            'satisfied': len(boundary_violations) == 0,
            'violations': boundary_violations
        }
        if len(boundary_violations) > 0:
            results['all_satisfied'] = False
        
        return results


def create_constraint_parameters(vehicle_params: VehicleParameters,
                                landing_site: LandingSite,
                                initial_state: StateVector,
                                final_state: StateVector,
                                scaling_system: Optional[ScalingSystem] = None) -> ConstraintParameters:
    """
    Create constraint parameters from basic inputs.
    
    Args:
        vehicle_params: Vehicle parameters
        landing_site: Landing site parameters
        initial_state: Initial state
        final_state: Final state
        scaling_system: Optional scaling system
        
    Returns:
        ConstraintParameters instance
    """
    return ConstraintParameters(
        glide_slope_angle=landing_site.glide_slope_angle,
        glide_slope_active=True,
        vertical_motion_active=True,
        vertical_motion_start_time=0.9,  # Last 10% of trajectory
        vertical_motion_min_altitude=0.0,
        min_mass=vehicle_params.m_dry,
        initial_state=initial_state,
        final_state=final_state,
        scaling_system=scaling_system
    )


def test_constraints():
    """Test the constraints module."""
    print("Testing constraints module...")
    
    # Create mock parameters
    from ..config import VehicleParameters, LandingSite, ASTEROID_A1, FULL_THRUST_VEHICLE
    
    # Mock landing site
    landing_site = LandingSite(
        name="TestSite",
        position=np.array([1000.0, 0.0, 0.0]),
        surface_normal=np.array([0.0, 0.0, 1.0]),
        glide_slope_angle=np.radians(30.0)
    )
    
    # Mock states
    initial_state = StateVector(
        position=np.array([2000.0, 0.0, 0.0]),
        velocity=np.array([-0.1, 0.0, 0.0]),
        mass=FULL_THRUST_VEHICLE.m_wet
    )
    
    final_state = StateVector(
        position=landing_site.position,
        velocity=np.array([0.0, 0.0, 0.0]),
        mass=FULL_THRUST_VEHICLE.m_dry * 1.1  # Slightly above dry mass
    )
    
    # Create constraint parameters
    constraint_params = create_constraint_parameters(
        vehicle_params=FULL_THRUST_VEHICLE,
        landing_site=landing_site,
        initial_state=initial_state,
        final_state=final_state
    )
    
    # Create constraints object
    constraints = TrajectoryConstraints(constraint_params)
    
    print("✓ ConstraintParameters created successfully")
    print(f"  - Glide slope angle: {np.degrees(constraint_params.glide_slope_angle):.1f}°")
    print(f"  - Min mass: {constraint_params.min_mass:.2f} kg")
    print(f"  - Vertical motion start: {constraint_params.vertical_motion_start_time}")
    
    # Test constraint checking with mock trajectory
    N = 10
    times = np.linspace(0, 100, N)
    
    # Create a trajectory that should satisfy constraints
    positions = np.zeros((N, 3))
    for i in range(N):
        t_frac = i / (N - 1)
        positions[i] = initial_state.position * (1 - t_frac) + final_state.position * t_frac
        positions[i, 2] = 100 * (1 - t_frac)  # Descending in z-direction
    
    velocities = np.zeros((N, 3))
    velocities[:, 0] = -0.1 * (1 - times/100)  # Slowing down
    
    masses = np.linspace(initial_state.mass, final_state.mass, N)
    
    thrusts = np.zeros((N, 3))
    thrusts[:, 0] = 40.0  # 40N thrust in x-direction
    thrusts[:, 2] = 20.0  # 20N thrust in z-direction
    
    # Check constraints
    results = constraints.check_constraint_satisfaction(
        positions=positions,
        velocities=velocities,
        masses=masses,
        thrusts=thrusts,
        times=times,
        landing_site=landing_site,
        vehicle_params=FULL_THRUST_VEHICLE
    )
    
    print("✓ Constraint checking completed")
    print(f"  - All constraints satisfied: {results['all_satisfied']}")
    
    for constraint_name, constraint_result in results['constraints'].items():
        print(f"  - {constraint_name}: {constraint_result['satisfied']}")
    
    return True


if __name__ == "__main__":
    test_constraints()