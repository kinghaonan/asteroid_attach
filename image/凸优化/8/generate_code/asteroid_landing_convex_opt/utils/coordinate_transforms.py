"""
Coordinate transformation utilities for asteroid landing convex optimization.

This module provides functions for converting between coordinate systems,
rotating between asteroid-fixed and inertial frames, and computing geometric
quantities needed for constraints (glide slope, vertical motion).

Functions:
    cartesian_to_spherical: Convert Cartesian coordinates to spherical (r, δ, λ)
    spherical_to_cartesian: Convert spherical coordinates to Cartesian
    rotate_to_asteroid_frame: Rotate from inertial to asteroid-fixed frame
    rotate_from_asteroid_frame: Rotate from asteroid-fixed to inertial frame
    compute_surface_normal: Compute surface normal at a point on asteroid
    compute_glide_slope_cone: Compute glide slope cone constraint parameters
    compute_vertical_motion_constraint: Compute vertical motion constraint
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def cartesian_to_spherical(position: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert Cartesian coordinates (x, y, z) to spherical coordinates (r, δ, λ).
    
    Spherical coordinates:
        r: radial distance from origin
        δ: declination (latitude-like, measured from equatorial plane)
        λ: right ascension (longitude-like, measured from x-axis in equatorial plane)
    
    Args:
        position: Cartesian position vector [x, y, z] in meters
        
    Returns:
        Tuple of (r, δ, λ) where:
            r: radial distance (m)
            δ: declination (radians, -π/2 to π/2)
            λ: right ascension (radians, 0 to 2π)
            
    Raises:
        ValueError: If position is not a 3-element vector
    """
    if position.shape != (3,):
        raise ValueError(f"Position must be a 3-element vector, got shape {position.shape}")
    
    x, y, z = position
    r = np.linalg.norm(position)
    
    if r == 0:
        # At origin, declination and right ascension are undefined
        return 0.0, 0.0, 0.0
    
    # Declination (δ): angle from equatorial plane
    δ = np.arcsin(z / r)
    
    # Right ascension (λ): angle in equatorial plane from x-axis
    λ = np.arctan2(y, x)
    if λ < 0:
        λ += 2 * np.pi  # Normalize to [0, 2π)
    
    return float(r), float(δ), float(λ)


def spherical_to_cartesian(r: float, δ: float, λ: float) -> np.ndarray:
    """
    Convert spherical coordinates (r, δ, λ) to Cartesian coordinates (x, y, z).
    
    Args:
        r: Radial distance from origin (m)
        δ: Declination (radians, -π/2 to π/2)
        λ: Right ascension (radians, 0 to 2π)
        
    Returns:
        Cartesian position vector [x, y, z] in meters
        
    Raises:
        ValueError: If r is negative
    """
    if r < 0:
        raise ValueError(f"Radial distance r must be non-negative, got {r}")
    
    # Handle r = 0 case
    if r == 0:
        return np.zeros(3)
    
    # Compute Cartesian coordinates
    cos_δ = np.cos(δ)
    x = r * cos_δ * np.cos(λ)
    y = r * cos_δ * np.sin(λ)
    z = r * np.sin(δ)
    
    return np.array([x, y, z], dtype=np.float64)


def rotate_to_asteroid_frame(
    position: np.ndarray,
    rotation_axis: np.ndarray,
    rotation_angle: float
) -> np.ndarray:
    """
    Rotate a position vector from inertial frame to asteroid-fixed frame.
    
    The asteroid-fixed frame rotates with the asteroid. This function applies
    a rotation about the asteroid's rotation axis by the given angle.
    
    Args:
        position: Position vector in inertial frame [x, y, z]
        rotation_axis: Unit vector along asteroid rotation axis
        rotation_angle: Rotation angle (radians, positive for right-hand rule)
        
    Returns:
        Position vector in asteroid-fixed frame
        
    Raises:
        ValueError: If rotation_axis is not a unit vector
    """
    if position.shape != (3,):
        raise ValueError(f"Position must be a 3-element vector, got shape {position.shape}")
    
    if rotation_axis.shape != (3,):
        raise ValueError(f"Rotation axis must be a 3-element vector, got shape {rotation_axis.shape}")
    
    # Normalize rotation axis
    axis_norm = np.linalg.norm(rotation_axis)
    if abs(axis_norm - 1.0) > 1e-10:
        rotation_axis = rotation_axis / axis_norm
        logger.warning(f"Rotation axis normalized from norm {axis_norm} to 1.0")
    
    # Rodrigues' rotation formula
    # R = I + sinθ * K + (1 - cosθ) * K²
    # where K is the cross-product matrix of rotation_axis
    
    # Identity matrix
    I = np.eye(3)
    
    # Cross-product matrix K
    K = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
    ])
    
    # K²
    K2 = K @ K
    
    # Rotation matrix
    sinθ = np.sin(rotation_angle)
    cosθ = np.cos(rotation_angle)
    R = I + sinθ * K + (1 - cosθ) * K2
    
    # Apply rotation
    return R @ position


def rotate_from_asteroid_frame(
    position: np.ndarray,
    rotation_axis: np.ndarray,
    rotation_angle: float
) -> np.ndarray:
    """
    Rotate a position vector from asteroid-fixed frame to inertial frame.
    
    This is the inverse of rotate_to_asteroid_frame.
    
    Args:
        position: Position vector in asteroid-fixed frame [x, y, z]
        rotation_axis: Unit vector along asteroid rotation axis
        rotation_angle: Rotation angle (radians, positive for right-hand rule)
        
    Returns:
        Position vector in inertial frame
    """
    # Rotating from asteroid frame to inertial frame is the inverse rotation
    # Inverse rotation is rotation by negative angle
    return rotate_to_asteroid_frame(position, rotation_axis, -rotation_angle)


def compute_surface_normal(
    position: np.ndarray,
    asteroid_params: Any,
    shape_model: Optional[Any] = None
) -> np.ndarray:
    """
    Compute surface normal vector at a given point on asteroid surface.
    
    For a triaxial ellipsoid, the surface normal is the gradient of the
    ellipsoid equation. For Castalia, a shape model is required.
    
    Args:
        position: Position vector in asteroid-fixed frame [x, y, z] (m)
        asteroid_params: Asteroid parameters (must have semi_axes attribute)
        shape_model: Optional shape model for irregular asteroids like Castalia
        
    Returns:
        Unit normal vector pointing outward from asteroid surface
        
    Raises:
        ValueError: If position is not on or near asteroid surface
        NotImplementedError: For irregular asteroids without shape model
    """
    if position.shape != (3,):
        raise ValueError(f"Position must be a 3-element vector, got shape {position.shape}")
    
    # Check if we have a shape model for irregular asteroids
    if hasattr(asteroid_params, 'name') and 'castalia' in asteroid_params.name.lower():
        if shape_model is None:
            raise NotImplementedError(
                "Shape model required for Castalia surface normal computation. "
                "Use get_castalia_shape_model() from asteroid_data.castalia."
            )
        
        # Use shape model to compute surface normal
        try:
            return shape_model.compute_surface_normal(position)
        except AttributeError:
            raise NotImplementedError(
                "Shape model must implement compute_surface_normal() method"
            )
    
    # For triaxial ellipsoids, compute normal analytically
    if not hasattr(asteroid_params, 'semi_axes'):
        raise ValueError("Asteroid parameters must have semi_axes attribute")
    
    a, b, c = asteroid_params.semi_axes
    
    # For ellipsoid (x/a)² + (y/b)² + (z/c)² = 1
    # Normal vector is gradient: [2x/a², 2y/b², 2z/c²]
    x, y, z = position
    normal = np.array([2*x/(a**2), 2*y/(b**2), 2*z/(c**2)], dtype=np.float64)
    
    # Normalize
    norm = np.linalg.norm(normal)
    if norm == 0:
        raise ValueError(f"Position {position} results in zero normal vector")
    
    return normal / norm


def compute_glide_slope_cone(
    landing_site_position: np.ndarray,
    surface_normal: np.ndarray,
    glide_slope_angle: float
) -> Tuple[np.ndarray, float]:
    """
    Compute parameters for glide slope cone constraint.
    
    The glide slope constraint ensures the trajectory stays within a cone
    defined by the glide slope angle from the vertical at the landing site.
    
    Eq. 13: ∥r - r_f∥ cosθ - (r - r_f)·n̂ ≤ 0
    
    Args:
        landing_site_position: Position of landing site [x, y, z] (m)
        surface_normal: Unit normal vector at landing site
        glide_slope_angle: Maximum allowed angle from vertical (radians)
        
    Returns:
        Tuple of (cone_axis, cone_angle) where:
            cone_axis: Unit vector along cone axis (opposite to surface normal)
            cone_angle: Cone half-angle (radians)
            
    Raises:
        ValueError: If inputs have incorrect dimensions
    """
    if landing_site_position.shape != (3,):
        raise ValueError(f"Landing site position must be 3-element vector, got {landing_site_position.shape}")
    
    if surface_normal.shape != (3,):
        raise ValueError(f"Surface normal must be 3-element vector, got {surface_normal.shape}")
    
    # Cone axis is opposite to surface normal (pointing upward)
    cone_axis = -surface_normal
    
    # Normalize cone axis
    axis_norm = np.linalg.norm(cone_axis)
    if axis_norm == 0:
        raise ValueError("Surface normal has zero magnitude")
    cone_axis = cone_axis / axis_norm
    
    # Cone half-angle is the glide slope angle
    cone_angle = float(glide_slope_angle)
    
    return cone_axis, cone_angle


def compute_vertical_motion_constraint(
    position: np.ndarray,
    landing_site_position: np.ndarray,
    surface_normal: np.ndarray,
    min_altitude: float = 0.0
) -> float:
    """
    Compute vertical motion constraint value.
    
    The vertical motion constraint ensures the vehicle maintains positive
    altitude relative to the landing site during the final phase.
    
    Eq. 60-61: (r - r_f)·n̂ ≥ 0 (or ≥ min_altitude)
    
    Args:
        position: Current position [x, y, z] (m)
        landing_site_position: Landing site position [x, y, z] (m)
        surface_normal: Unit normal vector at landing site
        min_altitude: Minimum allowed altitude (m), default 0
        
    Returns:
        Constraint value: (r - r_f)·n̂ - min_altitude
        Should be ≥ 0 for constraint satisfaction
        
    Raises:
        ValueError: If inputs have incorrect dimensions
    """
    if position.shape != (3,):
        raise ValueError(f"Position must be 3-element vector, got {position.shape}")
    
    if landing_site_position.shape != (3,):
        raise ValueError(f"Landing site position must be 3-element vector, got {landing_site_position.shape}")
    
    if surface_normal.shape != (3,):
        raise ValueError(f"Surface normal must be 3-element vector, got {surface_normal.shape}")
    
    # Compute relative position
    relative_position = position - landing_site_position
    
    # Compute dot product with surface normal
    altitude_component = np.dot(relative_position, surface_normal)
    
    # Apply minimum altitude requirement
    return altitude_component - min_altitude


def test_coordinate_transforms() -> bool:
    """
    Test the coordinate transforms module.
    
    Returns:
        True if all tests pass, False otherwise
    """
    logger.info("Testing coordinate transforms module...")
    
    test_results = []
    
    try:
        # Test 1: Cartesian to spherical and back
        test_position = np.array([1000.0, 2000.0, 3000.0])
        r, δ, λ = cartesian_to_spherical(test_position)
        recovered_position = spherical_to_cartesian(r, δ, λ)
        
        error = np.linalg.norm(test_position - recovered_position)
        test_passed = error < 1e-10
        test_results.append(("Cartesian ↔ spherical round-trip", test_passed, error))
        
        # Test 2: Spherical to Cartesian edge cases
        # Test at origin
        origin_position = spherical_to_cartesian(0.0, 0.0, 0.0)
        test_passed = np.allclose(origin_position, [0.0, 0.0, 0.0])
        test_results.append(("Spherical origin to Cartesian", test_passed, np.linalg.norm(origin_position)))
        
        # Test 3: Rotation consistency
        rotation_axis = np.array([0.0, 0.0, 1.0])  # z-axis
        rotation_angle = np.pi / 4  # 45 degrees
        
        # Create a test vector
        test_vector = np.array([1.0, 0.0, 0.0])
        
        # Rotate to asteroid frame and back
        asteroid_frame = rotate_to_asteroid_frame(test_vector, rotation_axis, rotation_angle)
        inertial_frame = rotate_from_asteroid_frame(asteroid_frame, rotation_axis, rotation_angle)
        
        error = np.linalg.norm(test_vector - inertial_frame)
        test_passed = error < 1e-10
        test_results.append(("Rotation round-trip", test_passed, error))
        
        # Test 4: Rotation by 2π should return original
        full_rotation = rotate_to_asteroid_frame(test_vector, rotation_axis, 2*np.pi)
        error = np.linalg.norm(test_vector - full_rotation)
        test_passed = error < 1e-10
        test_results.append(("2π rotation identity", test_passed, error))
        
        # Test 5: Surface normal for ellipsoid
        class MockAsteroidParams:
            def __init__(self):
                self.semi_axes = (1000.0, 800.0, 600.0)
        
        asteroid = MockAsteroidParams()
        surface_point = np.array([1000.0, 0.0, 0.0])  # On x-axis
        normal = compute_surface_normal(surface_point, asteroid)
        
        # Expected normal at (a, 0, 0) is [1, 0, 0]
        expected_normal = np.array([1.0, 0.0, 0.0])
        error = np.linalg.norm(normal - expected_normal)
        test_passed = error < 1e-10
        test_results.append(("Ellipsoid surface normal", test_passed, error))
        
        # Test 6: Glide slope cone computation
        landing_site = np.array([0.0, 0.0, 600.0])  # On z-axis
        surface_normal = np.array([0.0, 0.0, 1.0])  # Pointing upward
        glide_angle = np.radians(30.0)  # 30 degrees
        
        cone_axis, cone_angle = compute_glide_slope_cone(
            landing_site, surface_normal, glide_angle
        )
        
        # Cone axis should be opposite to surface normal
        expected_axis = np.array([0.0, 0.0, -1.0])
        axis_error = np.linalg.norm(cone_axis - expected_axis)
        angle_error = abs(cone_angle - glide_angle)
        
        test_passed = axis_error < 1e-10 and angle_error < 1e-10
        test_results.append(("Glide slope cone", test_passed, max(axis_error, angle_error)))
        
        # Test 7: Vertical motion constraint
        position_above = np.array([0.0, 0.0, 800.0])  # 200m above landing site
        constraint_value = compute_vertical_motion_constraint(
            position_above, landing_site, surface_normal, min_altitude=100.0
        )
        
        # Expected: (800-600)*1 - 100 = 100
        expected_value = 100.0
        error = abs(constraint_value - expected_value)
        test_passed = error < 1e-10
        test_results.append(("Vertical motion constraint", test_passed, error))
        
        # Print test results
        all_passed = True
        for test_name, passed, error_val in test_results:
            status = "PASS" if passed else "FAIL"
            logger.info(f"  {test_name}: {status} (error: {error_val:.2e})")
            if not passed:
                all_passed = False
        
        if all_passed:
            logger.info("All coordinate transforms tests passed!")
        else:
            logger.error("Some coordinate transforms tests failed!")
        
        return all_passed
        
    except Exception as e:
        logger.error(f"Coordinate transforms test failed with exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    success = test_coordinate_transforms()
    if success:
        print("Coordinate transforms module tests passed!")
    else:
        print("Coordinate transforms module tests failed!")
        exit(1)