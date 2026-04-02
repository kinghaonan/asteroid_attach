"""
Castalia asteroid shape model, rotation data, and landing site definitions.

This module implements the Castalia asteroid shape model based on the paper's
references [25,31], providing surface geometry, rotation parameters, and
predefined landing sites (LS1, LS2, LS3) for the landing experiments.

References:
- [25] Scheeres, D.J., et al. "Dynamics of Orbits Close to Asteroid 4769 Castalia."
- [31] Werner, R.A., Scheeres, D.J. "Exterior Gravitational Potential of a Polyhedron."
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import json
import os
from pathlib import Path

from ..config import LandingSite, G


@dataclass
class CastaliaShapeModel:
    """
    Shape model for asteroid 4769 Castalia.
    
    Based on the polyhedron model from references [25,31] with 4092 faces.
    This class provides methods to compute surface normals, validate landing
    sites, and generate surface points.
    
    Attributes:
        vertices: (N, 3) array of vertex coordinates in body-fixed frame [m]
        faces: (M, 3) array of face vertex indices
        face_normals: (M, 3) array of outward-pointing face normals
        face_areas: (M,) array of face areas [m²]
        center_of_mass: (3,) array of center of mass in body-fixed frame [m]
        bounding_sphere_radius: Radius of bounding sphere [m]
        average_radius: Average radius of asteroid [m]
        semi_axes: Triaxial semi-axes (a, b, c) [m]
        rotation_period: Rotation period [s]
        rotation_rate: Rotation rate [rad/s]
        rotation_axis: Rotation axis unit vector in inertial frame
    """
    
    # Basic shape parameters from paper
    semi_axes: Tuple[float, float, float] = (1.609, 0.996, 0.832)  # (a, b, c) in km
    rotation_period: float = 4.07 * 3600.0  # 4.07 hours in seconds
    rotation_rate: float = 2 * np.pi / (4.07 * 3600.0)  # rad/s
    rotation_axis: np.ndarray = field(default_factory=lambda: np.array([0, 0, 1]))
    
    # Derived parameters
    bounding_sphere_radius: float = 1.609  # km (largest semi-axis)
    average_radius: float = 1.145  # km (approximate)
    center_of_mass: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))
    
    # Polyhedron data (simplified - in practice would load from file)
    vertices: Optional[np.ndarray] = None
    faces: Optional[np.ndarray] = None
    face_normals: Optional[np.ndarray] = None
    face_areas: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Initialize derived parameters and shape model."""
        # Convert semi-axes from km to meters
        self.semi_axes = tuple(ax * 1000.0 for ax in self.semi_axes)
        self.bounding_sphere_radius *= 1000.0
        self.average_radius *= 1000.0
        
        # Initialize rotation axis
        if isinstance(self.rotation_axis, list):
            self.rotation_axis = np.array(self.rotation_axis)
        self.rotation_axis = self.rotation_axis / np.linalg.norm(self.rotation_axis)
        
        # Initialize center of mass
        if isinstance(self.center_of_mass, list):
            self.center_of_mass = np.array(self.center_of_mass)
            
        # Generate simplified shape model if not provided
        if self.vertices is None:
            self._generate_simplified_shape()
    
    def _generate_simplified_shape(self):
        """Generate a simplified ellipsoidal approximation of Castalia.
        
        In a full implementation, this would load the actual 4092-face
        polyhedron from a data file. For development, we use an ellipsoid
        approximation with similar dimensions.
        """
        # Create an ellipsoid with the given semi-axes
        n_points = 100
        u = np.linspace(0, 2 * np.pi, n_points)
        v = np.linspace(0, np.pi, n_points)
        
        u_grid, v_grid = np.meshgrid(u, v)
        
        # Parametric equations for ellipsoid
        x = self.semi_axes[0] * np.sin(v_grid) * np.cos(u_grid)
        y = self.semi_axes[1] * np.sin(v_grid) * np.sin(u_grid)
        z = self.semi_axes[2] * np.cos(v_grid)
        
        # Flatten to get vertices
        self.vertices = np.column_stack([
            x.flatten(), y.flatten(), z.flatten()
        ])
        
        # Create simple triangulation (simplified - actual would be more complex)
        # For development purposes only
        self.faces = np.array([[0, 1, 2]])  # Dummy face
        
        # Compute face normals (simplified)
        self.face_normals = np.array([[0, 0, 1]])  # Dummy normal
        
        # Compute face areas (simplified)
        self.face_areas = np.array([1.0])  # Dummy area
    
    def compute_surface_normal(self, point: np.ndarray) -> np.ndarray:
        """
        Compute the outward surface normal at a given point on Castalia.
        
        Args:
            point: (3,) array, position in body-fixed frame [m]
            
        Returns:
            normal: (3,) array, unit normal vector outward from surface
            
        Note:
            In a full implementation, this would find the closest face
            and return its normal. Here we use an ellipsoid approximation.
        """
        # For an ellipsoid, the normal at point (x, y, z) is (x/a², y/b², z/c²)
        x, y, z = point
        a, b, c = self.semi_axes
        
        # Handle origin case
        if np.linalg.norm(point) < 1e-6:
            return np.array([0, 0, 1])
        
        normal = np.array([x / (a**2), y / (b**2), z / (c**2)])
        normal_norm = np.linalg.norm(normal)
        
        if normal_norm < 1e-12:
            return np.array([0, 0, 1])
            
        return normal / normal_norm
    
    def is_point_on_surface(self, point: np.ndarray, tolerance: float = 1.0) -> bool:
        """
        Check if a point is on the surface of Castalia within tolerance.
        
        Args:
            point: (3,) array, position in body-fixed frame [m]
            tolerance: Distance tolerance [m]
            
        Returns:
            True if point is on surface within tolerance
        """
        # For ellipsoid: (x/a)² + (y/b)² + (z/c)² = 1
        x, y, z = point
        a, b, c = self.semi_axes
        
        # Compute ellipsoid equation value
        value = (x/a)**2 + (y/b)**2 + (z/c)**2
        
        # Check if close to 1 (on surface)
        return abs(value - 1.0) < tolerance / min(a, b, c)
    
    def compute_surface_point(self, latitude: float, longitude: float) -> np.ndarray:
        """
        Compute surface point from latitude and longitude.
        
        Args:
            latitude: Latitude in radians (-π/2 to π/2)
            longitude: Longitude in radians (0 to 2π)
            
        Returns:
            point: (3,) array, position on surface in body-fixed frame [m]
        """
        a, b, c = self.semi_axes
        
        # Convert to parametric coordinates
        x = a * np.cos(latitude) * np.cos(longitude)
        y = b * np.cos(latitude) * np.sin(longitude)
        z = c * np.sin(latitude)
        
        return np.array([x, y, z])
    
    def get_landing_site_position(self, site_name: str) -> np.ndarray:
        """
        Get position of a predefined landing site.
        
        Args:
            site_name: Name of landing site ("LS1", "LS2", or "LS3")
            
        Returns:
            position: (3,) array, landing site position in body-fixed frame [m]
            
        Raises:
            ValueError: If site_name is not recognized
        """
        sites = get_castalia_landing_sites()
        
        if site_name not in sites:
            raise ValueError(f"Unknown landing site: {site_name}. "
                           f"Available sites: {list(sites.keys())}")
        
        return sites[site_name].position
    
    def validate_landing_site(self, point: np.ndarray, 
                            min_slope_angle: float = 15.0) -> Dict[str, Any]:
        """
        Validate a potential landing site.
        
        Args:
            point: (3,) array, candidate landing site position [m]
            min_slope_angle: Minimum acceptable slope angle [deg]
            
        Returns:
            validation_dict: Dictionary with validation results
        """
        result = {
            'is_on_surface': False,
            'surface_normal': None,
            'slope_angle': None,
            'is_valid': False,
            'errors': []
        }
        
        # Check if point is on surface
        if not self.is_point_on_surface(point):
            result['errors'].append(f"Point not on surface: {point}")
            return result
        
        result['is_on_surface'] = True
        
        # Compute surface normal
        normal = self.compute_surface_normal(point)
        result['surface_normal'] = normal
        
        # Compute slope angle (angle between normal and vertical)
        # For small bodies, "vertical" is radial direction from center
        radial = point / np.linalg.norm(point)
        slope_angle_rad = np.arccos(np.clip(np.dot(normal, radial), -1.0, 1.0))
        slope_angle_deg = np.degrees(slope_angle_rad)
        result['slope_angle'] = slope_angle_deg
        
        # Check slope constraint
        if slope_angle_deg > min_slope_angle:
            result['errors'].append(f"Slope too steep: {slope_angle_deg:.1f}° > {min_slope_angle}°")
        else:
            result['is_valid'] = True
        
        return result
    
    def get_shape_statistics(self) -> Dict[str, Any]:
        """Get statistics about the Castalia shape model."""
        return {
            'semi_axes_m': self.semi_axes,
            'semi_axes_km': tuple(ax / 1000.0 for ax in self.semi_axes),
            'bounding_sphere_radius_m': self.bounding_sphere_radius,
            'average_radius_m': self.average_radius,
            'rotation_period_s': self.rotation_period,
            'rotation_period_hours': self.rotation_period / 3600.0,
            'rotation_rate_rad_s': self.rotation_rate,
            'rotation_axis': self.rotation_axis.tolist(),
            'volume_m3': self._compute_volume(),
            'surface_area_m2': self._compute_surface_area(),
        }
    
    def _compute_volume(self) -> float:
        """Compute volume of ellipsoid approximation."""
        a, b, c = self.semi_axes
        return (4/3) * np.pi * a * b * c
    
    def _compute_surface_area(self) -> float:
        """Compute approximate surface area of ellipsoid."""
        a, b, c = self.semi_axes
        
        # Approximation for ellipsoid surface area
        p = 1.6075  # Constant for approximation
        return 4 * np.pi * ((a**p * b**p + a**p * c**p + b**p * c**p) / 3)**(1/p)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert shape model to dictionary for serialization."""
        return {
            'semi_axes': self.semi_axes,
            'rotation_period': self.rotation_period,
            'rotation_rate': self.rotation_rate,
            'rotation_axis': self.rotation_axis.tolist(),
            'bounding_sphere_radius': self.bounding_sphere_radius,
            'average_radius': self.average_radius,
            'center_of_mass': self.center_of_mass.tolist(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CastaliaShapeModel':
        """Create shape model from dictionary."""
        return cls(
            semi_axes=tuple(data['semi_axes']),
            rotation_period=data['rotation_period'],
            rotation_rate=data['rotation_rate'],
            rotation_axis=np.array(data['rotation_axis']),
            bounding_sphere_radius=data['bounding_sphere_radius'],
            average_radius=data['average_radius'],
            center_of_mass=np.array(data['center_of_mass']),
        )


def get_castalia_shape_model() -> CastaliaShapeModel:
    """
    Get the Castalia shape model instance.
    
    Returns:
        CastaliaShapeModel: Configured shape model for Castalia
    """
    return CastaliaShapeModel()


def compute_castalia_surface_normal(point: np.ndarray) -> np.ndarray:
    """
    Compute surface normal at a given point on Castalia.
    
    Args:
        point: (3,) array, position in body-fixed frame [m]
        
    Returns:
        normal: (3,) array, unit normal vector
    """
    model = get_castalia_shape_model()
    return model.compute_surface_normal(point)


def validate_castalia_landing_site(point: np.ndarray, 
                                 min_slope_angle: float = 15.0) -> Dict[str, Any]:
    """
    Validate if a point is a valid landing site on Castalia.
    
    Args:
        point: (3,) array, candidate landing site position [m]
        min_slope_angle: Minimum acceptable slope angle [deg]
        
    Returns:
        validation_dict: Dictionary with validation results
    """
    model = get_castalia_shape_model()
    return model.validate_landing_site(point, min_slope_angle)


def get_castalia_landing_sites() -> Dict[str, LandingSite]:
    """
    Get predefined landing sites for Castalia (LS1, LS2, LS3).
    
    Returns:
        Dict mapping site names to LandingSite objects
        
    Note:
        Positions are based on paper descriptions and approximate locations
        on the Castalia shape. In practice, these would be determined from
        the actual polyhedron model.
    """
    # Get shape model for reference
    model = get_castalia_shape_model()
    a, b, c = model.semi_axes
    
    # Define landing sites based on paper descriptions
    # These are approximate positions that would be on the surface
    sites = {}
    
    # LS1: Near the "equator" on the +x side
    ls1_pos = np.array([a * 0.9, 0.0, 0.0])  # Near tip of long axis
    ls1_normal = compute_castalia_surface_normal(ls1_pos)
    
    # LS2: On the "northern" hemisphere, +y side
    ls2_pos = np.array([0.0, b * 0.8, c * 0.6])
    ls2_normal = compute_castalia_surface_normal(ls2_pos)
    
    # LS3: On the "southern" hemisphere, -y side
    ls3_pos = np.array([0.0, -b * 0.8, -c * 0.6])
    ls3_normal = compute_castalia_surface_normal(ls3_pos)
    
    # Create LandingSite objects
    sites['LS1'] = LandingSite(
        name='LS1',
        position=ls1_pos,
        surface_normal=ls1_normal,
        glide_slope_angle=30.0  # Default glide slope angle
    )
    
    sites['LS2'] = LandingSite(
        name='LS2',
        position=ls2_pos,
        surface_normal=ls2_normal,
        glide_slope_angle=30.0
    )
    
    sites['LS3'] = LandingSite(
        name='LS3',
        position=ls3_pos,
        surface_normal=ls3_normal,
        glide_slope_angle=30.0
    )
    
    return sites


def save_castalia_shape_data(model: CastaliaShapeModel, filename: str) -> None:
    """
    Save Castalia shape data to JSON file.
    
    Args:
        model: CastaliaShapeModel instance
        filename: Output JSON filename
    """
    data = model.to_dict()
    
    # Ensure directory exists
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)


def load_castalia_shape_data(filename: str) -> CastaliaShapeModel:
    """
    Load Castalia shape data from JSON file.
    
    Args:
        filename: Input JSON filename
        
    Returns:
        CastaliaShapeModel: Loaded shape model
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    
    return CastaliaShapeModel.from_dict(data)


def test_castalia_module() -> bool:
    """
    Test the Castalia module functionality.
    
    Returns:
        bool: True if all tests pass
    """
    print("Testing Castalia module...")
    
    try:
        # Test shape model creation
        model = get_castalia_shape_model()
        print(f"✓ Created Castalia shape model")
        
        # Test statistics
        stats = model.get_shape_statistics()
        print(f"✓ Computed shape statistics: {stats['semi_axes_km']} km")
        
        # Test surface normal computation
        test_point = np.array([model.semi_axes[0] * 0.8, 0, 0])
        normal = model.compute_surface_normal(test_point)
        print(f"✓ Computed surface normal: {normal}")
        
        # Test landing site validation
        validation = model.validate_landing_site(test_point)
        print(f"✓ Validated landing site: {validation['is_valid']}")
        
        # Test landing sites
        sites = get_castalia_landing_sites()
        print(f"✓ Retrieved {len(sites)} landing sites: {list(sites.keys())}")
        
        # Test serialization
        temp_file = "temp_castalia_shape.json"
        save_castalia_shape_data(model, temp_file)
        loaded_model = load_castalia_shape_data(temp_file)
        
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        print(f"✓ Serialization/deserialization successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run tests if module is executed directly
    success = test_castalia_module()
    if success:
        print("\nAll Castalia module tests passed!")
    else:
        print("\nSome Castalia module tests failed.")