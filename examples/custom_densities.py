"""
Utility functions for creating custom density functions and advanced analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from typing import Callable, Tuple, List

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from algorithms.standard_lloyd import StandardLloydAlgorithm
from algorithms.capacity_constrained import CapacityConstrainedDistributionAlgorithm
from utils.base import VoronoiUtils


class DensityFunctionBuilder:
    """Builder class for creating custom density functions."""
    
    @staticmethod
    def gaussian_peak(center: Tuple[float, float], width: float = 0.1, height: float = 1.0) -> Callable:
        """Create a Gaussian peak at specified location."""
        cx, cy = center
        def gaussian(x, y):
            return height * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * width**2))
        return gaussian
    
    @staticmethod
    def linear_gradient(direction: Tuple[float, float], offset: float = 0.0) -> Callable:
        """Create a linear gradient in specified direction."""
        dx, dy = direction
        norm = np.sqrt(dx**2 + dy**2)
        dx, dy = dx/norm, dy/norm
        
        def gradient(x, y):
            return offset + dx * x + dy * y
        return gradient
    
    @staticmethod
    def radial_function(center: Tuple[float, float], power: float = 1.0) -> Callable:
        """Create a radial function from center."""
        cx, cy = center
        def radial(x, y):
            r = np.sqrt((x - cx)**2 + (y - cy)**2)
            return 1.0 / (1.0 + r**power)
        return radial
    
    @staticmethod
    def combine_functions(functions: List[Callable], weights: List[float] = None) -> Callable:
        """Combine multiple density functions with optional weights."""
        if weights is None:
            weights = [1.0] * len(functions)
        
        def combined(x, y):
            total = 0.0
            for func, weight in zip(functions, weights):
                total += weight * func(x, y)
            return max(0.0, total)  # Ensure non-negative
        return combined
    
    @staticmethod
    def checkerboard_pattern(n_squares: int = 4, amplitude: float = 1.0) -> Callable:
        """Create a checkerboard density pattern."""
        def checkerboard(x, y):
            i = int(x * n_squares)
            j = int(y * n_squares)
            if (i + j) % 2 == 0:
                return amplitude
            else:
                return 0.1 * amplitude
        return checkerboard
    
    @staticmethod
    def ring_pattern(center: Tuple[float, float], radius: float = 0.3, width: float = 0.1) -> Callable:
        """Create a ring-shaped density pattern."""
        cx, cy = center
        def ring(x, y):
            r = np.sqrt((x - cx)**2 + (y - cy)**2)
            return np.exp(-((r - radius)**2) / (2 * width**2))
        return ring


class AlgorithmAnalyzer:
    """Class for analyzing and comparing algorithm performance."""
    
    def __init__(self, domain: Tuple[float, float, float, float] = (0, 1, 0, 1)):
        self.domain = domain
        self.lloyd = StandardLloydAlgorithm(domain)
        self.capacity_lloyd = CapacityConstrainedDistributionAlgorithm(domain)
    
    def convergence_analysis(self, density_func: Callable, n_generators: int = 20,
                           max_iterations: int = 100) -> dict:
        """Analyze convergence properties of both algorithms."""
        
        initial_points = self.lloyd.generate_random_points(n_generators, seed=42)
        
        # Run standard Lloyd
        final_standard, energy_standard = self.lloyd.run(
            initial_points.copy(),
            n_iterations=max_iterations,
            density_func=density_func
        )
        
        # Run capacity-constrained Lloyd
        final_capacity, energy_capacity, variance_capacity, combined_energy = self.capacity_lloyd.run(
            initial_points.copy(),
            density_func,
            n_iterations=max_iterations
        )
        
        # Compute convergence rates
        def compute_convergence_rate(energy_history):
            if len(energy_history) < 10:
                return 0.0
            # Fit exponential decay to last 50% of iterations
            start_idx = len(energy_history) // 2
            iterations = np.arange(start_idx, len(energy_history))
            energies = np.array(energy_history[start_idx:])
            
            if len(energies) > 1 and energies[-1] > 0:
                # Log-linear fit
                log_energies = np.log(energies + 1e-10)
                slope, _ = np.polyfit(iterations, log_energies, 1)
                return -slope
            return 0.0
        
        standard_rate = compute_convergence_rate(energy_standard)
        capacity_rate = compute_convergence_rate(energy_capacity)
        
        return {
            'standard_energy': energy_standard,
            'capacity_energy': energy_capacity,
            'capacity_variance': variance_capacity,
            'standard_final_energy': energy_standard[-1],
            'capacity_final_energy': energy_capacity[-1],
            'standard_convergence_rate': standard_rate,
            'capacity_convergence_rate': capacity_rate,
            'final_points_standard': final_standard,
            'final_points_capacity': final_capacity
        }
    
    def capacity_uniformity_analysis(self, density_func: Callable, 
                                   final_points_standard: np.ndarray,
                                   final_points_capacity: np.ndarray) -> dict:
        """Analyze capacity uniformity for both algorithms."""
        
        # Generate sample points for capacity computation
        sample_points = self.lloyd.generate_random_points(10000, seed=123)
        
        # Compute capacities for standard Lloyd
        capacities_standard, regions_standard = self.capacity_lloyd.compute_capacities(
            final_points_standard, sample_points, density_func)
        
        # Compute capacities for capacity-constrained Lloyd
        capacities_capacity, regions_capacity = self.capacity_lloyd.compute_capacities(
            final_points_capacity, sample_points, density_func)
        
        return {
            'capacities_standard': capacities_standard,
            'capacities_capacity': capacities_capacity,
            'std_dev_standard': np.std(capacities_standard),
            'std_dev_capacity': np.std(capacities_capacity),
            'coefficient_variation_standard': np.std(capacities_standard) / np.mean(capacities_standard),
            'coefficient_variation_capacity': np.std(capacities_capacity) / np.mean(capacities_capacity),
            'improvement_factor': np.std(capacities_standard) / np.std(capacities_capacity)
        }


def demonstrate_custom_densities():
    """Demonstrate creating and using custom density functions."""
    
    print("Custom Density Functions Demo")
    print("=" * 35)
    
    # Create custom density functions
    builder = DensityFunctionBuilder()
    
    # Multiple Gaussian peaks
    peak1 = builder.gaussian_peak((0.2, 0.2), width=0.1, height=2.0)
    peak2 = builder.gaussian_peak((0.8, 0.8), width=0.1, height=1.5)
    peak3 = builder.gaussian_peak((0.2, 0.8), width=0.15, height=1.0)
    multi_peaks = builder.combine_functions([peak1, peak2, peak3])
    
    # Ring pattern
    ring = builder.ring_pattern((0.5, 0.5), radius=0.3, width=0.05)
    
    # Checkerboard pattern
    checkerboard = builder.checkerboard_pattern(n_squares=6, amplitude=2.0)
    
    # Linear gradient
    gradient = builder.linear_gradient((1.0, 0.5), offset=0.2)
    
    custom_densities = {
        'Multi-Peaks': multi_peaks,
        'Ring': ring,
        'Checkerboard': checkerboard,
        'Gradient': gradient
    }
    
    # Test with each density
    domain = (0, 1, 0, 1)
    analyzer = AlgorithmAnalyzer(domain)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    for i, (name, density_func) in enumerate(custom_densities.items()):
        print(f"\nTesting {name} density...")
        
        # Run analysis
        results = analyzer.convergence_analysis(density_func, n_generators=16, max_iterations=30)
        capacity_results = analyzer.capacity_uniformity_analysis(
            density_func, 
            results['final_points_standard'],
            results['final_points_capacity']
        )
        
        # Create density background
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.array([[density_func(xi, yi) for xi in x] for yi in y])
        
        # Plot density and final points
        im = axes[0, i].contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
        axes[0, i].plot(results['final_points_capacity'][:, 0], 
                       results['final_points_capacity'][:, 1], 'ro', markersize=6)
        axes[0, i].set_title(f'{name} Density')
        axes[0, i].set_xlabel('x')
        axes[0, i].set_ylabel('y')
        
        # Plot convergence
        axes[1, i].plot(results['standard_energy'], 'b-', linewidth=2, label='Standard')
        axes[1, i].plot(results['capacity_energy'], 'r-', linewidth=2, label='Capacity-Constrained')
        axes[1, i].set_xlabel('Iteration')
        axes[1, i].set_ylabel('Energy')
        axes[1, i].set_title(f'{name} - Convergence')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].set_yscale('log')
        
        print(f"  Final energies - Standard: {results['standard_final_energy']:.4f}, "
              f"Capacity-Constrained: {results['capacity_final_energy']:.4f}")
        print(f"  Capacity improvement: {capacity_results['improvement_factor']:.2f}x")
    
    plt.tight_layout()
    
    # Save to output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'custom_densities_demo.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nCustom densities demo saved to output/custom_densities_demo.png")


if __name__ == "__main__":
    demonstrate_custom_densities()
