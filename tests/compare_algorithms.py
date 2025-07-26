"""
Direct comparison between Standard Lloyd and Capacity-Constrained Lloyd algorithms.

This script demonstrates the key differences and trade-offs between the two algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from algorithms.standard_lloyd import StandardLloydAlgorithm
from algorithms.capacity_constrained import CapacityConstrainedDistributionAlgorithm
from utils import get_example_density_functions, visualize_algorithm_results


def compare_algorithms():
    """Compare Standard Lloyd vs Capacity-Constrained Lloyd algorithms."""
    
    print("ALGORITHM COMPARISON: STANDARD vs CAPACITY-CONSTRAINED")
    print("=" * 70)
    
    # Setup
    domain = (0, 1, 0, 1)
    n_generators = 20
    n_iterations = 30
    
    # Initialize both algorithms
    standard_lloyd = StandardLloydAlgorithm(domain)
    capacity_lloyd = CapacityConstrainedDistributionAlgorithm(domain)
    
    # Get density function - use challenging multi-gaussian
    density_funcs = get_example_density_functions()
    density_func = density_funcs['multi_gaussian']
    
    # Generate same initial conditions for fair comparison
    initial_generators = standard_lloyd.generate_random_points(n_generators, seed=42)
    
    print(f"Comparing algorithms with {n_generators} generators")
    print(f"Using multi-gaussian density (challenging case)")
    print(f"Running {n_iterations} iterations each\n")
    
    # Run Standard Lloyd Algorithm
    print("1. RUNNING STANDARD LLOYD ALGORITHM")
    print("-" * 40)
    
    final_standard, energy_standard = standard_lloyd.run(
        initial_generators.copy(),
        n_iterations=n_iterations,
        density_func=density_func,
        tolerance=1e-8,
        verbose=True
    )
    
    print("\n2. RUNNING CAPACITY-CONSTRAINED LLOYD ALGORITHM")
    print("-" * 50)
    
    final_capacity, energy_capacity, variance_capacity = capacity_lloyd.run(
        initial_generators.copy(),
        density_func,
        n_iterations=n_iterations,
        tolerance=1e-8,
        blue_noise_weight=0.3,
        verbose=True
    )[:3]  # Take only first 3 return values
    
    # Analyze capacity distributions for both algorithms
    print("\n3. CAPACITY ANALYSIS")
    print("-" * 20)
    
    # For standard Lloyd (doesn't have built-in capacity analysis)
    sample_points = standard_lloyd.generate_random_points(10000, seed=123)
    
    # Standard Lloyd capacity analysis using capacity algorithm's analysis method
    standard_analysis = capacity_lloyd.analyze_capacity_distribution(final_standard, density_func)
    capacities_standard = standard_analysis['capacities']
    
    # Capacity-constrained analysis
    capacity_analysis = capacity_lloyd.analyze_capacity_distribution(final_capacity, density_func)
    capacities_capacity = capacity_analysis['capacities']
    
    # Print comparison
    print("\nCapacity Distribution Comparison:")
    print(f"{'Metric':<25} {'Standard Lloyd':<20} {'Capacity-Constrained':<20} {'Improvement'}")
    print("-" * 80)
    print(f"{'Mean Capacity':<25} {np.mean(capacities_standard):<20.6f} {np.mean(capacities_capacity):<20.6f} {'Similar'}")
    print(f"{'Capacity Std Dev':<25} {standard_analysis['capacity_std']:<20.6f} {capacity_analysis['capacity_std']:<20.6f} {standard_analysis['capacity_std']/capacity_analysis['capacity_std']:<20.2f}x")
    print(f"{'Capacity Range':<25} {np.ptp(capacities_standard):<20.6f} {np.ptp(capacities_capacity):<20.6f} {np.ptp(capacities_standard)/np.ptp(capacities_capacity):<20.2f}x")
    print(f"{'Blue Noise Quality':<25} {'N/A':<20} {capacity_analysis['blue_noise_quality']:<20.6f} {'N/A'}")
    
    print(f"\nEnergy Comparison:")
    print(f"Standard Lloyd Final Energy: {energy_standard[-1]:.6f}")
    print(f"Capacity-Constrained Final Energy: {energy_capacity[-1]:.6f}")
    print(f"Energy Trade-off: {(energy_capacity[-1]/energy_standard[-1] - 1)*100:.1f}% higher energy for capacity uniformity")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # Create density background
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[density_func(xi, yi) for xi in x] for yi in y])
    
    # Row 1: Initial configuration and final results
    # Initial configuration
    im0 = axes[0, 0].contourf(X, Y, Z, levels=25, cmap='viridis', alpha=0.8)
    axes[0, 0].plot(initial_generators[:, 0], initial_generators[:, 1], 'wo', markersize=6, 
                   markeredgecolor='black', markeredgewidth=1)
    axes[0, 0].set_title('Initial Configuration', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0, 0], shrink=0.8)
    
    # Standard Lloyd result
    im1 = axes[0, 1].contourf(X, Y, Z, levels=25, cmap='viridis', alpha=0.8)
    axes[0, 1].plot(final_standard[:, 0], final_standard[:, 1], 'ro', markersize=6,
                   markeredgecolor='white', markeredgewidth=1)
    axes[0, 1].set_title('Standard Lloyd Result', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)
    
    # Capacity-constrained result
    im2 = axes[0, 2].contourf(X, Y, Z, levels=25, cmap='viridis', alpha=0.8)
    axes[0, 2].plot(final_capacity[:, 0], final_capacity[:, 1], 'bo', markersize=6,
                   markeredgecolor='white', markeredgewidth=1)
    axes[0, 2].set_title('Capacity-Constrained Result', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0, 2], shrink=0.8)
    
    # Row 2: Convergence analysis
    # Energy convergence
    axes[1, 0].plot(energy_standard, 'r-', linewidth=2.5, label='Standard Lloyd')
    axes[1, 0].plot(energy_capacity, 'b-', linewidth=2.5, label='Capacity-Constrained')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Energy')
    axes[1, 0].set_title('Energy Convergence Comparison', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # Capacity variance (only for capacity-constrained)
    axes[1, 1].plot(variance_capacity, 'g-', linewidth=2.5)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Capacity Variance')
    axes[1, 1].set_title('Capacity Variance Evolution', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    # Energy ratio over iterations
    energy_ratio = np.array(energy_capacity) / np.array(energy_standard)
    axes[1, 2].plot(energy_ratio, 'purple', linewidth=2.5)
    axes[1, 2].set_xlabel('Iteration')
    axes[1, 2].set_ylabel('Energy Ratio (CC/Standard)')
    axes[1, 2].set_title('Energy Trade-off Over Time', fontsize=12, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
    
    # Row 3: Capacity distribution analysis
    # Capacity histograms
    axes[2, 0].hist(capacities_standard, bins=12, alpha=0.7, label='Standard Lloyd', 
                   color='red', density=True, edgecolor='black')
    axes[2, 0].hist(capacities_capacity, bins=12, alpha=0.7, label='Capacity-Constrained', 
                   color='blue', density=True, edgecolor='black')
    axes[2, 0].set_xlabel('Capacity Value')
    axes[2, 0].set_ylabel('Density')
    axes[2, 0].set_title('Capacity Distribution Comparison', fontsize=12, fontweight='bold')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Capacity statistics bar chart
    metrics = ['Std Dev', 'Range', 'Coeff. Var.']
    standard_values = [
        np.std(capacities_standard),
        np.ptp(capacities_standard),
        np.std(capacities_standard)/np.mean(capacities_standard)
    ]
    capacity_values = [
        np.std(capacities_capacity),
        np.ptp(capacities_capacity),
        np.std(capacities_capacity)/np.mean(capacities_capacity)  # Calculate coefficient of variation
    ]
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    bars1 = axes[2, 1].bar(x_pos - width/2, standard_values, width, 
                          label='Standard', color='red', alpha=0.7)
    bars2 = axes[2, 1].bar(x_pos + width/2, capacity_values, width, 
                          label='Capacity-Constrained', color='blue', alpha=0.7)
    
    axes[2, 1].set_xlabel('Capacity Metrics')
    axes[2, 1].set_ylabel('Value')
    axes[2, 1].set_title('Capacity Uniformity Metrics', fontsize=12, fontweight='bold')
    axes[2, 1].set_xticks(x_pos)
    axes[2, 1].set_xticklabels(metrics)
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    # Summary statistics text
    axes[2, 2].axis('off')
    summary_text = f"""
ALGORITHM COMPARISON SUMMARY

Energy Performance:
• Standard Lloyd: {energy_standard[-1]:.4f}
• Capacity-Constrained: {energy_capacity[-1]:.4f}
• Trade-off: {(energy_capacity[-1]/energy_standard[-1] - 1)*100:.1f}% higher energy

Capacity Uniformity:
• Standard Std Dev: {np.std(capacities_standard):.4f}
• Capacity-Constrained Std Dev: {np.std(capacities_capacity):.4f}
• Improvement: {np.std(capacities_standard)/np.std(capacities_capacity):.1f}x better

Key Insights:
• Capacity-constrained algorithm sacrifices
  some energy optimality for uniformity
• {np.std(capacities_standard)/np.std(capacities_capacity):.1f}x improvement in capacity uniformity
• Excellent for applications requiring
  equal capacity per cell
• Minimal computational overhead
"""
    
    axes[2, 2].text(0.05, 0.95, summary_text, transform=axes[2, 2].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout(pad=2.0)
    
    # Save to output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'algorithm_comparison.png'), dpi=200, bbox_inches='tight')
    plt.show()
    
    print(f"\n{'='*70}")
    print("COMPARISON COMPLETED")
    print(f"{'='*70}")
    print("Visualization saved to output/algorithm_comparison.png")
    
    print(f"\nCONCLUSIONS:")
    print(f"• Standard Lloyd: Better energy minimization ({energy_standard[-1]:.4f})")
    print(f"• Capacity-Constrained: Better capacity uniformity ({np.std(capacities_standard)/np.std(capacities_capacity):.1f}x improvement)")
    print(f"• Trade-off: {(energy_capacity[-1]/energy_standard[-1] - 1)*100:.1f}% energy increase for uniformity")
    print(f"• Use Standard Lloyd for energy-focused applications")
    print(f"• Use Capacity-Constrained for uniformity-critical applications")


if __name__ == "__main__":
    compare_algorithms()
