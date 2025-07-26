"""
Visualization utilities for Lloyd algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
from scipy.spatial import Voronoi, voronoi_plot_2d
import warnings

warnings.filterwarnings('ignore')


def visualize_algorithm_results(generators: np.ndarray, 
                               sample_points: np.ndarray,
                               density_func: Optional[callable] = None,
                               title: str = "Algorithm Result",
                               domain_bounds: Tuple[float, float, float, float] = (0, 1, 0, 1)):
    """
    Visualize the results of a Lloyd algorithm.
    
    Args:
        generators: Final generator positions
        sample_points: Sample points used for computation
        density_func: Optional density function for background visualization
        title: Plot title
        domain_bounds: Domain boundaries
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    xmin, xmax, ymin, ymax = domain_bounds
    
    # Plot 1: Voronoi diagram with generators
    if len(generators) > 3:  # Need at least 4 points for Voronoi
        try:
            vor = Voronoi(generators)
            voronoi_plot_2d(vor, ax=ax1, show_vertices=False, line_colors='blue', line_width=1)
        except:
            pass  # Skip Voronoi if it fails
    
    ax1.plot(generators[:, 0], generators[:, 1], 'ro', markersize=8, label='Generators')
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    ax1.set_title(f'{title} - Voronoi Diagram')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Density function background with generators
    if density_func is not None:
        x = np.linspace(xmin, xmax, 100)
        y = np.linspace(ymin, ymax, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.array([[density_func(xi, yi) for xi in x] for yi in y])
        
        im = ax2.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
        plt.colorbar(im, ax=ax2, label='Density')
    
    ax2.plot(generators[:, 0], generators[:, 1], 'ro', markersize=8, label='Generators')
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
    ax2.set_title(f'{title} - Density and Generators')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_convergence_curves(energy_history: List[float], 
                           capacity_variance_history: Optional[List[float]] = None,
                           title: str = "Convergence Analysis"):
    """Plot convergence curves for energy and optionally capacity variance."""
    fig, axes = plt.subplots(1, 2 if capacity_variance_history else 1, figsize=(12, 4))
    
    if capacity_variance_history is None:
        axes = [axes]
    
    # Energy convergence
    axes[0].plot(energy_history, 'b-', linewidth=2)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Energy')
    axes[0].set_title(f'{title} - Energy Convergence')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # Capacity variance convergence (if available)
    if capacity_variance_history is not None:
        axes[1].plot(capacity_variance_history, 'r-', linewidth=2)
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Capacity Variance')
        axes[1].set_title(f'{title} - Capacity Variance')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.show()


def compare_algorithms_visually(results_dict: dict, 
                               density_func: callable,
                               domain_bounds: Tuple[float, float, float, float] = (0, 1, 0, 1)):
    """
    Visually compare multiple algorithm results.
    
    Args:
        results_dict: Dictionary with algorithm names as keys and results as values
        density_func: Density function for background
        domain_bounds: Domain boundaries
    """
    n_algorithms = len(results_dict)
    fig, axes = plt.subplots(2, n_algorithms, figsize=(5*n_algorithms, 10))
    
    if n_algorithms == 1:
        axes = axes.reshape(-1, 1)
    
    xmin, xmax, ymin, ymax = domain_bounds
    
    # Create density background
    x = np.linspace(xmin, xmax, 100)
    y = np.linspace(ymin, ymax, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[density_func(xi, yi) for xi in x] for yi in y])
    
    for i, (name, result) in enumerate(results_dict.items()):
        generators = result['generators']
        energy_history = result['energy_history']
        
        # Plot final configuration
        im = axes[0, i].contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
        axes[0, i].plot(generators[:, 0], generators[:, 1], 'ro', markersize=8)
        axes[0, i].set_title(f'{name}')
        axes[0, i].set_xlabel('x')
        axes[0, i].set_ylabel('y')
        plt.colorbar(im, ax=axes[0, i])
        
        # Plot energy convergence
        axes[1, i].plot(energy_history, 'b-', linewidth=2)
        axes[1, i].set_xlabel('Iteration')
        axes[1, i].set_ylabel('Energy')
        axes[1, i].set_title(f'{name} - Energy')
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].set_yscale('log')
    
    plt.tight_layout()
    plt.show()
