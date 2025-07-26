"""
Example script for the Standard Lloyd Algorithm (CVT).

This script demonstrates the classic Lloyd algorithm for Centroidal Voronoi 
Tessellations with various density functions and visualization options.

Run this script from the algorithm folder to test the Standard Lloyd implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from standard_lloyd import StandardLloydAlgorithm
from utils import get_example_density_functions, visualize_algorithm_results


def run_standard_lloyd_examples():
    """
    Run comprehensive examples of the Standard Lloyd Algorithm.
    
    This function demonstrates:
    1. Different density functions
    2. Convergence behavior
    3. Energy minimization
    4. Voronoi tessellation quality
    """
    print("STANDARD LLOYD ALGORITHM EXAMPLES")
    print("=" * 50)
    print("Classic Centroidal Voronoi Tessellation (CVT)")
    print()
    
    # Algorithm setup
    domain = (0, 1, 0, 1)
    algorithm = StandardLloydAlgorithm(domain)
    
    # Print algorithm information
    info = algorithm.get_algorithm_info()
    print(f"Algorithm: {info['name']}")
    print(f"Type: {info['type']}")
    print(f"Description: {info['description']}")
    print()
    
    # Get density functions for testing
    density_funcs = get_example_density_functions()
    test_densities = ['uniform', 'gaussian', 'multi_gaussian']
    
    n_generators = 20
    n_iterations = 40
    results = {}
    
    for density_name in test_densities:
        print(f"\nTesting with {density_name.replace('_', ' ').title()} density")
        print("-" * 45)
        
        density_func = density_funcs[density_name]
        
        # Generate initial points
        np.random.seed(42)
        initial_points = algorithm.generate_random_points(n_generators, seed=42)
        
        # Run algorithm
        final_points, energy_history = algorithm.run(
            initial_points,
            n_iterations=n_iterations,
            density_func=density_func,
            verbose=True
        )
        
        results[density_name] = {
            'initial_points': initial_points,
            'final_points': final_points,
            'energy_history': energy_history,
            'density_func': density_func
        }
        
        print(f"Initial energy: {energy_history[0]:.6f}")
        print(f"Final energy: {energy_history[-1]:.6f}")
        print(f"Energy reduction: {((energy_history[0] - energy_history[-1]) / energy_history[0] * 100):.2f}%")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('Standard Lloyd Algorithm Examples\nCentroidal Voronoi Tessellations', fontsize=16)
    
    for i, density_name in enumerate(test_densities):
        result = results[density_name]
        
        # Plot initial distribution
        ax = axes[i, 0]
        # Simple scatter plot instead of complex visualization
        ax.plot(result['initial_points'][:, 0], result['initial_points'][:, 1], 'ro', markersize=6)
        ax.set_title(f'Initial Points - {density_name.replace("_", " ").title()}')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Plot final distribution
        ax = axes[i, 1]
        # Simple scatter plot instead of complex visualization
        ax.plot(result['final_points'][:, 0], result['final_points'][:, 1], 'bo', markersize=6)
        ax.set_title(f'Final CVT - {density_name.replace("_", " ").title()}')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Plot convergence
        ax = axes[i, 2]
        iterations = range(len(result['energy_history']))
        ax.plot(iterations, result['energy_history'], 'b-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Energy')
        ax.set_title(f'Energy Convergence - {density_name.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)
        
        # Add energy reduction annotation
        energy_reduction = ((result['energy_history'][0] - result['energy_history'][-1]) / 
                           result['energy_history'][0] * 100)
        ax.text(0.02, 0.98, f'Reduction: {energy_reduction:.1f}%', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Performance summary
    print("\n" + "=" * 50)
    print("PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"{'Density':<15} {'Initial Energy':<15} {'Final Energy':<15} {'Reduction %':<12}")
    print("-" * 60)
    
    for density_name in test_densities:
        result = results[density_name]
        initial_energy = result['energy_history'][0]
        final_energy = result['energy_history'][-1]
        reduction = (initial_energy - final_energy) / initial_energy * 100
        
        print(f"{density_name.replace('_', ' ').title():<15} {initial_energy:<15.6f} {final_energy:<15.6f} {reduction:<12.2f}")
    
    print("\nKey Observations:")
    print("• Standard Lloyd minimizes CVT energy through centroidal movement")
    print("• Convergence speed varies with density function complexity")
    print("• Final configurations achieve well-distributed point sets")
    print("• Energy reduction indicates optimization effectiveness")
    
    print(f"\nSTANDARD LLOYD ALGORITHM EXAMPLES COMPLETED")
    print("Classic CVT optimization with energy minimization!")


def quick_test():
    """Quick test of the Standard Lloyd Algorithm."""
    print("QUICK STANDARD LLOYD TEST")
    print("-" * 30)
    
    domain = (0, 1, 0, 1)
    algorithm = StandardLloydAlgorithm(domain)
    
    # Simple test with uniform density
    n_generators = 12
    initial_points = algorithm.generate_random_points(n_generators, seed=42)
    
    final_points, energy_history = algorithm.run(
        initial_points,
        n_iterations=20,
        verbose=False
    )
    
    print(f"Energy: {energy_history[0]:.4f} → {energy_history[-1]:.4f}")
    print(f"Improvement: {((energy_history[0] - energy_history[-1]) / energy_history[0] * 100):.1f}%")
    print("Standard Lloyd test completed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Standard Lloyd Algorithm Examples')
    parser.add_argument('--quick', action='store_true', help='Run quick test only')
    args = parser.parse_args()
    
    if args.quick:
        quick_test()
    else:
        run_standard_lloyd_examples()
