#!/usr/bin/env python3
"""
Generate Standard Lloyd Algorithm Results

This script generates comprehensive results for the Standard Lloyd Algorithm,
including visualizations and analysis saved to the output directory.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from algorithms.standard_lloyd import StandardLloydAlgorithm
from utils import get_example_density_functions

def ensure_output_dir():
    """Ensure the output directory exists."""
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def generate_lloyd_results():
    """Generate comprehensive Standard Lloyd algorithm results."""
    
    print("=" * 60)
    print("GENERATING STANDARD LLOYD ALGORITHM RESULTS")
    print("=" * 60)
    
    output_dir = ensure_output_dir()
    
    # Algorithm setup
    domain = (0, 1, 0, 1)
    algorithm = StandardLloydAlgorithm(domain)
    
    # Get density functions
    density_funcs = get_example_density_functions()
    
    # Test configurations
    configs = [
        {
            'name': 'Uniform Density',
            'density': density_funcs['uniform'],
            'n_generators': 16,
            'filename': 'lloyd_uniform'
        },
        {
            'name': 'Gaussian Density',
            'density': density_funcs['gaussian'],
            'n_generators': 16,
            'filename': 'lloyd_gaussian'
        },
        {
            'name': 'Multi-Gaussian Density',
            'density': density_funcs['multi_gaussian'],
            'n_generators': 20,
            'filename': 'lloyd_multi_gaussian'
        },
        {
            'name': 'Linear Density',
            'density': density_funcs['linear'],
            'n_generators': 24,
            'filename': 'lloyd_linear'
        }
    ]
    
    # Generate results for each configuration
    results = []
    
    for i, config in enumerate(configs, 1):
        print(f"\n{i}. {config['name']}")
        print("-" * 40)
        
        # Generate initial points
        np.random.seed(42)
        initial_points = algorithm.generate_random_points(config['n_generators'], seed=42)
        
        # Run algorithm
        print(f"Running Standard Lloyd Algorithm with {config['n_generators']} generators...")
        final_points, energy_history = algorithm.run(
            initial_points,
            n_iterations=50,
            density_func=config['density'],
            verbose=True
        )
        
        # Analyze results
        final_energy = energy_history[-1] if energy_history else 0
        converged = len(energy_history) < 50  # Converged if finished before max iterations
        
        analysis = {
            'final_energy': final_energy,
            'converged': converged,
            'iterations': len(energy_history)
        }
        
        # Store results
        result = {
            'config': config,
            'final_points': final_points,
            'energy_history': energy_history,
            'analysis': analysis
        }
        results.append(result)
        
        print(f"Final Energy: {analysis['final_energy']:.6f}")
        print(f"Convergence: {'Yes' if analysis['converged'] else 'No'}")
        
        # Create individual visualization
        create_individual_plot(result, output_dir)
    
    # Create summary visualization
    create_summary_plot(results, output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("STANDARD LLOYD ALGORITHM RESULTS SUMMARY")
    print("=" * 60)
    
    for i, result in enumerate(results, 1):
        config = result['config']
        analysis = result['analysis']
        print(f"{i}. {config['name']}")
        print(f"   Generators: {config['n_generators']}")
        print(f"   Final Energy: {analysis['final_energy']:.6f}")
        print(f"   Convergence: {'Yes' if analysis['converged'] else 'No'}")
        print(f"   Iterations: {len(result['energy_history'])}")
    
    print(f"\nAll results saved to: {output_dir}")
    print("Generated files:")
    print("- lloyd_uniform.png")
    print("- lloyd_gaussian.png") 
    print("- lloyd_multi_gaussian.png")
    print("- lloyd_linear.png")
    print("- lloyd_summary.png")

def create_individual_plot(result, output_dir):
    """Create individual plot for a configuration."""
    config = result['config']
    final_points = result['final_points']
    energy_history = result['energy_history']
    analysis = result['analysis']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Final distribution
    x, y = final_points[:, 0], final_points[:, 1]
    ax1.scatter(x, y, c='red', s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    ax1.set_title(f'{config["name"]}\n{len(final_points)} Generators')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Energy convergence
    ax2.plot(energy_history, 'b-', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Energy')
    ax2.set_title('Energy Convergence')
    ax2.grid(True, alpha=0.3)
    
    # Add energy info
    ax2.text(0.02, 0.98, f'Final Energy: {analysis["final_energy"]:.4f}', 
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    filename = os.path.join(output_dir, f'{config["filename"]}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def create_summary_plot(results, output_dir):
    """Create summary plot comparing all configurations."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    axes = [ax1, ax2, ax3, ax4]
    
    for i, (result, ax) in enumerate(zip(results, axes)):
        config = result['config']
        final_points = result['final_points']
        
        # Plot final distribution
        x, y = final_points[:, 0], final_points[:, 1]
        ax.scatter(x, y, c='red', s=40, alpha=0.8, edgecolors='black', linewidth=0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(f'{config["name"]}\nEnergy: {result["analysis"]["final_energy"]:.4f}')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Standard Lloyd Algorithm - All Configurations', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save summary plot
    filename = os.path.join(output_dir, 'lloyd_summary.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    generate_lloyd_results()
