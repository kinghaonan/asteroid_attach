# Asteroid Landing Trajectory Optimization using Convex Optimization

This repository implements the convex optimization framework for propellant-optimal powered descent trajectories on asteroids with irregular gravity fields, as described in the paper:

**"Trajectory Design Employing Convex Optimization for Landing on Irregularly Shaped Asteroids"**

## Core Features

- **Irregular Gravity Modeling**: Spherical harmonics (4×4) for exterior gravity and spherical Bessel functions for interior gravity
- **Lossless Convexification**: Transform nonconvex optimal control problem into convex Second-Order Cone Program (SOCP)
- **Successive Solution Method**: Iterative linearization to handle nonlinear gravity fields
- **Flight Time Optimization**: Brent's method to find optimal flight time minimizing propellant usage
- **Multiple Asteroid Models**: Triaxial ellipsoids (A1, A2, A3) and Castalia asteroid

## Installation

```bash
pip install -r requirements.txt
```

## Dependencies

- Python 3.8+
- numpy >= 1.21.0
- scipy >= 1.7.0
- cvxpy >= 1.2.0
- matplotlib >= 3.5.0
- MOSEK (recommended) or SCS for SOCP solving

## Project Structure

```
asteroid_landing_convex_opt/
├── main.py                            # Main entry point: run all experiments
├── config.py                          # Parameters for asteroids, vehicle, experiments
├── gravity_models/                    # Asteroid gravity computation
├── optimization/                      # Core optimization algorithms
├── dynamics/                          # System dynamics and discretization
├── asteroid_data/                     # Asteroid-specific data
├── vehicle/                           # Spacecraft model
├── experiments/                       # Reproduce paper experiments
├── utils/                             # Utilities and helpers
└── tests/                             # Unit tests for components
```

## Quick Start

```python
from main import run_all_experiments

# Run all experiments from the paper
run_all_experiments()
```

## Running Experiments

1. **Parameter Sweeps** (Figures 5-7):
```bash
python -m experiments.parameter_sweeps
```

2. **Castalia Landing** (Tables 5,7):
```bash
python -m experiments.castalia_landing
```

3. **Validation**:
```bash
python -m experiments.validation
```

## Key Algorithms

1. **Gravity Models**: Spherical harmonics (4×4) and spherical Bessel functions
2. **Problem Formulation**: P1 (original nonconvex) → P2 (relaxed) → P3 (convex SOCP)
3. **Successive Solution**: Algorithm 1 - iterative linearization of gravity
4. **Flight Time Optimization**: Algorithm 3 - Brent's method with coarse-to-fine strategy

## Expected Results

- **Triaxial Ellipsoids**: Unimodal propellant vs flight time curves
- **Castalia Landing**: Optimal flight times ~512-513s (full thrust), ~1050-1076s (quarter thrust)
- **Propellant Usage**: ~5.31-5.34kg (full thrust), ~3.4kg (quarter thrust)

## References

1. Park, S. Y., & Scheeres, D. J. (2006). "Trajectory Design Employing Convex Optimization for Landing on Irregularly Shaped Asteroids"
2. Werner, R. A. (1994). "The gravitational potential of a homogeneous polyhedron"
3. Scheeres, D. J. (1994). "Dynamics about uniformly rotating triaxial ellipsoids"

## License

MIT License