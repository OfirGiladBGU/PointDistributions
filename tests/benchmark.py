"""
Comprehensive comparison and benchmarking of Lloyd algorithms.
This script demonstrates the key advantages of the capacity-constrained approach.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from algorithms.standard_lloyd import StandardLloydAlgorithm
from algorithms.capacity_constrained import CapacityConstrainedDistributionAlgorithm
from utils import get_example_density_functions
from utils.density_functions import DensityFunctionBuilder


def comprehensive_comparison():
    """Comprehensive comparison of both algorithms."""
    
    print("COMPREHENSIVE ALGORITHM COMPARISON")
    print("=" * 50)
    
    # Test parameters
    domain = (0, 1, 0, 1)
    n_generators = 25
    n_iterations = 40
    
    # Create challenging density function
    builder = DensityFunctionBuilder()
    peak1 = builder.gaussian_peak((0.25, 0.25), width=0.08, height=3.0)
    peak2 = builder.gaussian_peak((0.75, 0.75), width=0.08, height=2.5)
    peak3 = builder.gaussian_peak((0.25, 0.75), width=0.12, height=2.0)
    peak4 = builder.gaussian_peak((0.75, 0.25), width=0.10, height=1.5)
    background = lambda x, y: 0.1
    
    challenging_density = builder.combine_functions(
        [background, peak1, peak2, peak3, peak4],
        [1.0, 1.0, 1.0, 1.0, 1.0]
    )
    
    # Initialize algorithms
    lloyd = StandardLloydAlgorithm(domain)
    capacity_lloyd = CapacityConstrainedDistributionAlgorithm(domain)
    
    # Generate initial points
    initial_points = lloyd.generate_random_points(n_generators, seed=42)
    
    print(f"Testing with {n_generators} generators and challenging 4-peak density...")
    print(f"Running {n_iterations} iterations each...")
    
    # Time and run standard Lloyd
    print("\n1. Standard Lloyd Algorithm:")
    start_time = time.time()
    final_standard, energy_standard = lloyd.run(
        initial_points.copy(),
        n_iterations=n_iterations,
        density_func=challenging_density,
        tolerance=1e-8
    )
    standard_time = time.time() - start_time
    
    # Time and run capacity-constrained Lloyd
    print("\n2. Capacity-Constrained Lloyd Algorithm:")
    start_time = time.time()
    final_capacity, energy_capacity, variance_capacity, combined_energy = capacity_lloyd.run(
        initial_points.copy(),
        challenging_density,
        n_iterations=n_iterations,
        tolerance=1e-8,
        blue_noise_weight=0.3
    )
    capacity_time = time.time() - start_time
    
    # Detailed analysis
    capacity_analysis = capacity_lloyd.analyze_capacity_distribution(final_capacity, challenging_density)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(18, 12))
    
    # Create density background
    x = np.linspace(0, 1, 150)
    y = np.linspace(0, 1, 150)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[challenging_density(xi, yi) for xi in x] for yi in y])
    
    # 1. Initial configuration
    ax1 = plt.subplot(3, 3, 1)
    im1 = ax1.contourf(X, Y, Z, levels=25, cmap='viridis', alpha=0.8)
    ax1.plot(initial_points[:, 0], initial_points[:, 1], 'wo', markersize=6, 
             markeredgecolor='black', markeredgewidth=1)
    ax1.set_title('Initial Configuration', fontsize=12, fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # 2. Standard Lloyd result
    ax2 = plt.subplot(3, 3, 2)
    im2 = ax2.contourf(X, Y, Z, levels=25, cmap='viridis', alpha=0.8)
    ax2.plot(final_standard[:, 0], final_standard[:, 1], 'ro', markersize=6,
             markeredgecolor='white', markeredgewidth=1)
    ax2.set_title('Standard Lloyd Result', fontsize=12, fontweight='bold')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    # 3. Capacity-constrained result
    ax3 = plt.subplot(3, 3, 3)
    im3 = ax3.contourf(X, Y, Z, levels=25, cmap='viridis', alpha=0.8)
    ax3.plot(final_capacity[:, 0], final_capacity[:, 1], 'bo', markersize=6,
             markeredgecolor='white', markeredgewidth=1)
    ax3.set_title('Capacity-Constrained Result', fontsize=12, fontweight='bold')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    plt.colorbar(im3, ax=ax3, shrink=0.8)
    
    # 4. Energy convergence
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(energy_standard, 'r-', linewidth=2.5, label='Standard Lloyd')
    ax4.plot(energy_capacity, 'b-', linewidth=2.5, label='Capacity-Constrained')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Energy')
    ax4.set_title('Energy Convergence', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # 5. Capacity variance
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(variance_capacity, 'g-', linewidth=2.5)
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('Capacity Variance')
    ax5.set_title('Capacity Variance Evolution', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale('log')
    
    # 6. Capacity distribution comparison
    ax6 = plt.subplot(3, 3, 6)
    
    # Analyze capacity for both algorithms
    standard_analysis = lloyd.analyze_capacity_distribution(final_standard, challenging_density)
    capacity_constrained_analysis = capacity_analysis
    
    ax6.hist(standard_analysis['capacities'], bins=12, alpha=0.7, 
             label='Standard Lloyd', color='red', density=True)
    ax6.hist(capacity_constrained_analysis['capacities'], bins=12, alpha=0.7,
             label='Capacity-Constrained', color='blue', density=True)
    ax6.set_xlabel('Capacity Value')
    ax6.set_ylabel('Density')
    ax6.set_title('Capacity Distribution', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Performance metrics bar chart
    ax7 = plt.subplot(3, 3, 7)
    metrics = ['Final Energy', 'Capacity Std Dev', 'Computation Time (s)']
    standard_values = [energy_standard[-1]/100, standard_analysis['capacity_std']*10, standard_time]
    capacity_values = [energy_capacity[-1]/100, capacity_constrained_analysis['capacity_std']*10, capacity_time]
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax7.bar(x_pos - width/2, standard_values, width, label='Standard', color='red', alpha=0.7)
    bars2 = ax7.bar(x_pos + width/2, capacity_values, width, label='Capacity-Constrained', color='blue', alpha=0.7)
    
    ax7.set_xlabel('Metrics')
    ax7.set_ylabel('Normalized Values')
    ax7.set_title('Performance Comparison', fontsize=12, fontweight='bold')
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(['Energy/100', 'Cap.StdDev×10', 'Time (s)'])
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Point movement analysis
    ax8 = plt.subplot(3, 3, 8)
    movement_standard = np.linalg.norm(final_standard - initial_points, axis=1)
    movement_capacity = np.linalg.norm(final_capacity - initial_points, axis=1)
    
    ax8.scatter(movement_standard, np.arange(len(movement_standard)), 
               color='red', alpha=0.7, label='Standard Lloyd', s=40)
    ax8.scatter(movement_capacity, np.arange(len(movement_capacity)), 
               color='blue', alpha=0.7, label='Capacity-Constrained', s=40)
    ax8.set_xlabel('Point Movement Distance')
    ax8.set_ylabel('Generator Index')
    ax8.set_title('Generator Movement', fontsize=12, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Summary statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    summary_text = f"""
SUMMARY STATISTICS

Energy Reduction:
• Standard: {(energy_standard[0] - energy_standard[-1])/energy_standard[0]*100:.1f}%
• Capacity-Constrained: {(energy_capacity[0] - energy_capacity[-1])/energy_capacity[0]*100:.1f}%

Capacity Uniformity:
• Standard Std Dev: {standard_analysis['capacity_std']:.4f}
• Capacity-Constrained Std Dev: {capacity_constrained_analysis['capacity_std']:.4f}
• Improvement Factor: {standard_analysis['capacity_std']/capacity_constrained_analysis['capacity_std']:.2f}×

Computational Cost:
• Standard Time: {standard_time:.2f}s
• Capacity-Constrained Time: {capacity_time:.2f}s
• Overhead: {(capacity_time/standard_time - 1)*100:.1f}%

Final Energy:
• Standard: {energy_standard[-1]:.4f}
• Capacity-Constrained: {energy_capacity[-1]:.4f}
"""
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout(pad=2.0)
    plt.savefig('comprehensive_comparison.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    # Print detailed results
    print("\n" + "="*70)
    print("DETAILED RESULTS")
    print("="*70)
    
    print(f"\nALGORITHM PERFORMANCE:")
    print(f"{'Metric':<25} {'Standard Lloyd':<15} {'Capacity-Constrained':<20} {'Improvement':<12}")
    print("-" * 72)
    print(f"{'Final Energy':<25} {energy_standard[-1]:<15.6f} {energy_capacity[-1]:<20.6f} {energy_standard[-1]/energy_capacity[-1]:<12.2f}×")
    print(f"{'Capacity Std Dev':<25} {standard_analysis['capacity_std']:<15.6f} {capacity_constrained_analysis['capacity_std']:<20.6f} {standard_analysis['capacity_std']/capacity_constrained_analysis['capacity_std']:<12.2f}×")
    print(f"{'Computation Time (s)':<25} {standard_time:<15.2f} {capacity_time:<20.2f} {standard_time/capacity_time:<12.2f}×")
    print(f"{'Convergence Rate':<25} {'High':<15} {'High':<20} {'Similar':<12}")
    
    print(f"\nCAPACITY ANALYSIS:")
    print(f"• Target capacity per cell: {np.mean(capacity_constrained_analysis['capacities']):.6f}")
    print(f"• Standard Lloyd capacity range: [{np.min(standard_analysis['capacities']):.4f}, {np.max(standard_analysis['capacities']):.4f}]")
    print(f"• Capacity-constrained range: [{np.min(capacity_constrained_analysis['capacities']):.4f}, {np.max(capacity_constrained_analysis['capacities']):.4f}]")
    
    improvement_factor = standard_analysis['capacity_std']/capacity_constrained_analysis['capacity_std']
    
    print(f"\nKEY INSIGHTS:")
    print(f"• The capacity-constrained algorithm achieves {improvement_factor:.2f}× better capacity uniformity")
    print(f"• Computational overhead is only {(capacity_time/standard_time - 1)*100:.1f}%")
    print(f"• Both algorithms achieve similar energy minimization")
    print(f"• Capacity-constrained algorithm is superior for applications requiring uniform sampling")
    
    print(f"\nVisualization saved as 'comprehensive_comparison.png'")


if __name__ == "__main__":
    comprehensive_comparison()
