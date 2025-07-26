"""
Common density functions for testing algorithms.
"""

import numpy as np
from typing import Dict, Callable


def uniform_density(x: float, y: float) -> float:
    """Uniform density function."""
    return 1.0


def gaussian_density(x: float, y: float) -> float:
    """Gaussian density centered at (0.5, 0.5)."""
    return np.exp(-10 * ((x - 0.5)**2 + (y - 0.5)**2))


def multi_gaussian_density(x: float, y: float) -> float:
    """Multiple Gaussian peaks."""
    peak1 = np.exp(-20 * ((x - 0.3)**2 + (y - 0.3)**2))
    peak2 = np.exp(-20 * ((x - 0.7)**2 + (y - 0.7)**2))
    peak3 = np.exp(-20 * ((x - 0.3)**2 + (y - 0.7)**2))
    return 0.1 + peak1 + peak2 + peak3


def linear_density(x: float, y: float) -> float:
    """Linear gradient density."""
    return 0.1 + x + y


def get_example_density_functions() -> Dict[str, Callable]:
    """Get a dictionary of example density functions."""
    return {
        'uniform': uniform_density,
        'gaussian': gaussian_density,
        'multi_gaussian': multi_gaussian_density,
        'linear': linear_density
    }
