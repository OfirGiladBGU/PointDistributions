#!/usr/bin/env python3
"""
Generate Capacity-Constrained Distribution Algorithm Results

This script generates comprehensive results for the Capacity-Constrained Distribution Algorithm,
including visualizations and analysis saved to the output directory.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from algorithms.capacity_constrained import CapacityConstrainedDistributionAlgorithm
from utils import get_example_density_functions

def ensure_output_dir():
    """Ensure the output directory exists."""
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def generate_capacity_constrained_results():
    """Generate comprehensive Capacity-Constrained Distribution algorithm results."""
    
    print("=" * 70)
    print("GENERATING CAPACITY-CONSTRAINED DISTRIBUTION ALGORITHM RESULTS")
    print("=" * 70)
    
    output_dir = ensure_output_dir()
    
    # Algorithm setup
    domain = (0, 1, 0, 1)
    algorithm = CapacityConstrainedDistributionAlgorithm(domain)
    
    # Get density functions
    density_funcs = get_example_density_functions()
    
    # Test configurations with different blue noise weights
    configs = [
        {
            'name': 'Uniform Density - Balanced',
            'density': density_funcs['uniform'],
            'n_generators': 16,
            'blue_noise_weight': 0.5,
            'filename': 'capacity_uniform_balanced'
        },
        {
            'name': 'Uniform Density - Blue Noise Focus',
            'density': density_funcs['uniform'],
            'n_generators': 16,
            'blue_noise_weight': 0.8,
            'filename': 'capacity_uniform_blue_noise'
        },
        {
            'name': 'Gaussian Density - Balanced',
            'density': density_funcs['gaussian'],
            'n_generators': 16,
            'blue_noise_weight': 0.5,
            'filename': 'capacity_gaussian_balanced'
        },
        {
            'name': 'Multi-Gaussian - Blue Noise Focus',
            'density': density_funcs['multi_gaussian'],
            'n_generators': 20,
            'blue_noise_weight': 0.7,
            'filename': 'capacity_multi_gaussian'
        },
        {
            'name': 'Linear Density - Balanced',
            'density': density_funcs['linear'],
            'n_generators': 24,
            'blue_noise_weight': 0.5,
            'filename': 'capacity_linear'
        }
    ]
    
    # Generate results for each configuration
    results = []
    
    for i, config in enumerate(configs, 1):
        print(f"\n{i}. {config['name']}")
        print("-" * 50)
        
        # Generate initial points
        np.random.seed(42)
        initial_points = algorithm.generate_random_points(config['n_generators'], seed=42)
        
        # Run algorithm
        print(f"Running Capacity-Constrained Distribution Algorithm...")
        print(f"Generators: {config['n_generators']}, Blue Noise Weight: {config['blue_noise_weight']}")
        
        final_points, energy_history, capacity_variance_history, combined_energy_history = algorithm.run(
            initial_points,
            config['density'],
            n_iterations=50,
            blue_noise_weight=config['blue_noise_weight'],
            verbose=True
        )
        
        # Analyze results
        analysis = algorithm.analyze_capacity_distribution(final_points, config['density'])
        
        # Store results
        result = {
            'config': config,
            'final_points': final_points,
            'energy_history': energy_history,
            'capacity_variance_history': capacity_variance_history,
            'combined_energy_history': combined_energy_history,
            'analysis': analysis
        }
        results.append(result)
        
        print(f"Final CVT Energy: {analysis['cvt_energy']:.6f}")
        print(f"Final Combined Energy: {analysis['combined_energy']:.6f}")
        print(f"Blue Noise Quality: {analysis['blue_noise_quality']:.4f}")
        print(f"Capacity Std Dev: {analysis['capacity_std']:.6f}")
        
        # Create individual visualization
        create_individual_plot(result, output_dir)
    
    # Create summary visualization
    create_summary_plot(results, output_dir)
    
    # Create blue noise comparison plot
    create_blue_noise_comparison(results, output_dir)
    
    # Print summary
    print("\n" + "=" * 70)
    print("CAPACITY-CONSTRAINED DISTRIBUTION ALGORITHM RESULTS SUMMARY")
    print("=" * 70)
    
    for i, result in enumerate(results, 1):
        config = result['config']
        analysis = result['analysis']
        print(f"{i}. {config['name']}")
        print(f"   Generators: {config['n_generators']}, Blue Weight: {config['blue_noise_weight']}")
        print(f"   CVT Energy: {analysis['cvt_energy']:.6f}")
        print(f"   Combined Energy: {analysis['combined_energy']:.6f}")
        print(f"   Blue Noise Quality: {analysis['blue_noise_quality']:.4f}")
        print(f"   Capacity Std Dev: {analysis['capacity_std']:.6f}")
    
    print(f"\nAll results saved to: {output_dir}")
    print("Generated files:")
    print("- capacity_uniform_balanced.png")
    print("- capacity_uniform_blue_noise.png")
    print("- capacity_gaussian_balanced.png")
    print("- capacity_multi_gaussian.png")
    print("- capacity_linear.png")
    print("- capacity_summary.png")
    print("- capacity_blue_noise_comparison.png")

def create_individual_plot(result, output_dir):
    """Create individual plot for a configuration."""
    config = result['config']
    final_points = result['final_points']
    energy_history = result['energy_history']
    combined_energy_history = result['combined_energy_history']
    capacity_variance_history = result['capacity_variance_history']
    analysis = result['analysis']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Final distribution
    x, y = final_points[:, 0], final_points[:, 1]
    ax1.scatter(x, y, c='blue', s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    ax1.set_title(f'{config["name"]}\n{len(final_points)} Generators')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: CVT Energy convergence
    ax2.plot(energy_history, 'r-', linewidth=2, label='CVT Energy')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('CVT Energy')
    ax2.set_title('CVT Energy Convergence')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Combined energy convergence
    ax3.plot(combined_energy_history, 'g-', linewidth=2, label='Combined Energy')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Combined Energy')
    ax3.set_title('Combined Energy Convergence')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Capacity variance evolution
    ax4.plot(capacity_variance_history, 'purple', linewidth=2, label='Capacity Variance')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Capacity Variance')
    ax4.set_title('Capacity Variance Evolution')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Add analysis info
    info_text = f"""Blue Noise Quality: {analysis['blue_noise_quality']:.4f}
Capacity Std Dev: {analysis['capacity_std']:.4f}
Overall Quality: {analysis['overall_quality']:.4f}"""
    
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, 
             verticalalignment='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    filename = os.path.join(output_dir, f'{config["filename"]}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def create_summary_plot(results, output_dir):
    """Create summary plot comparing all configurations."""
    n_results = len(results)
    cols = 3
    rows = (n_results + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i, result in enumerate(results):
        config = result['config']
        final_points = result['final_points']
        analysis = result['analysis']
        
        # Plot final distribution
        x, y = final_points[:, 0], final_points[:, 1]
        axes[i].scatter(x, y, c='blue', s=40, alpha=0.8, edgecolors='black', linewidth=0.5)
        axes[i].set_xlim(0, 1)
        axes[i].set_ylim(0, 1)
        axes[i].set_aspect('equal')
        axes[i].set_title(f'{config["name"]}\nBlue Noise: {analysis["blue_noise_quality"]:.3f}')
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(results), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Capacity-Constrained Distribution Algorithm - All Configurations', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save summary plot
    filename = os.path.join(output_dir, 'capacity_summary.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def create_blue_noise_comparison(results, output_dir):
    """Create comparison plot focusing on blue noise weights."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract data for comparison
    weights = [r['config']['blue_noise_weight'] for r in results]
    blue_noise_qualities = [r['analysis']['blue_noise_quality'] for r in results]
    capacity_stds = [r['analysis']['capacity_std'] for r in results]
    names = [r['config']['name'] for r in results]
    
    # Plot 1: Blue noise quality vs weight
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    for i, (weight, quality, name, color) in enumerate(zip(weights, blue_noise_qualities, names, colors)):
        ax1.scatter(weight, quality, c=[color], s=100, alpha=0.8, 
                   edgecolors='black', linewidth=1, label=name)
    
    ax1.set_xlabel('Blue Noise Weight')
    ax1.set_ylabel('Blue Noise Quality')
    ax1.set_title('Blue Noise Quality vs Weight')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: Capacity uniformity (inverse of std dev)
    capacity_uniformity = [1/std if std > 0 else 0 for std in capacity_stds]
    for i, (weight, uniformity, name, color) in enumerate(zip(weights, capacity_uniformity, names, colors)):
        ax2.scatter(weight, uniformity, c=[color], s=100, alpha=0.8, 
                   edgecolors='black', linewidth=1, label=name)
    
    ax2.set_xlabel('Blue Noise Weight')
    ax2.set_ylabel('Capacity Uniformity (1/std)')
    ax2.set_title('Capacity Uniformity vs Weight')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save comparison plot
    filename = os.path.join(output_dir, 'capacity_blue_noise_comparison.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    generate_capacity_constrained_results()
