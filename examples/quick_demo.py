"""
Quick demonstration of the Lloyd algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from algorithms.standard_lloyd import StandardLloydAlgorithm
from algorithms.capacity_constrained import CapacityConstrainedDistributionAlgorithm
from utils import get_example_density_functions

def quick_demo():
    """Quick demonstration of both algorithms."""
    
    print("Lloyd Algorithm Quick Demo")
    print("=" * 30)
    
    # Setup
    domain = (0, 1, 0, 1)
    n_generators = 16
    
    # Get algorithms and density
    lloyd = StandardLloydAlgorithm(domain)
    capacity_lloyd = CapacityConstrainedDistributionAlgorithm(domain)
    density_funcs = get_example_density_functions()
    density_func = density_funcs['multi_gaussian']  # Challenging test case
    
    # Initial points
    initial_points = lloyd.generate_random_points(n_generators, seed=42)
    
    print(f"Running algorithms with {n_generators} generators and multi-gaussian density...")
    
    # Run standard Lloyd
    print("\n1. Standard Lloyd Algorithm:")
    final_standard, energy_standard = lloyd.run(
        initial_points.copy(),
        n_iterations=30,
        density_func=density_func
    )
    
    # Run capacity-constrained Lloyd
    print("\n2. Capacity-Constrained Distribution Algorithm:")
    final_capacity, energy_capacity, variance_capacity, combined_energy = capacity_lloyd.run(
        initial_points.copy(),
        density_func,
        n_iterations=30,
        blue_noise_weight=0.3
    )
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Create density background
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[density_func(xi, yi) for xi in x] for yi in y])
    
    # Plot standard Lloyd result
    im1 = axes[0, 0].contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
    axes[0, 0].plot(final_standard[:, 0], final_standard[:, 1], 'ro', markersize=8)
    axes[0, 0].set_title('Standard Lloyd Algorithm')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot capacity-constrained result
    im2 = axes[0, 1].contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
    axes[0, 1].plot(final_capacity[:, 0], final_capacity[:, 1], 'ro', markersize=8)
    axes[0, 1].set_title('Capacity-Constrained Distribution')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot energy convergence
    axes[1, 0].plot(energy_standard, 'b-', linewidth=2, label='Standard')
    axes[1, 0].plot(energy_capacity, 'r-', linewidth=2, label='Capacity-Constrained')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Energy')
    axes[1, 0].set_title('Energy Convergence')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # Plot capacity variance
    axes[1, 1].plot(variance_capacity, 'g-', linewidth=2)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Capacity Variance')
    axes[1, 1].set_title('Capacity Variance (CC Algorithm)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    
    # Save to output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'lloyd_demo_results.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    print(f"Standard Lloyd - Final Energy: {energy_standard[-1]:.6f}")
    print(f"Capacity-Constrained - Final Energy: {energy_capacity[-1]:.6f}")
    print(f"Capacity-Constrained - Final Capacity Variance: {variance_capacity[-1]:.6f}")
    
    # Compute capacity statistics using the analysis method
    standard_analysis = capacity_lloyd.analyze_capacity_distribution(final_standard, density_func)
    capacity_analysis = capacity_lloyd.analyze_capacity_distribution(final_capacity, density_func)
    
    print(f"\nCapacity Uniformity:")
    print(f"Standard Lloyd - Capacity Std Dev: {standard_analysis['capacity_std']:.6f}")
    print(f"Capacity-Constrained - Capacity Std Dev: {capacity_analysis['capacity_std']:.6f}")
    improvement = standard_analysis['capacity_std'] / capacity_analysis['capacity_std']
    print(f"Improvement factor: {improvement:.2f}x more uniform")
    print(f"Blue Noise Quality: {capacity_analysis['blue_noise_quality']:.4f}")
    
    print(f"\nVisualization saved to output/lloyd_demo_results.png")

if __name__ == "__main__":
    quick_demo()
