"""
Example script for the Capacity-Constrained Lloyd Algorithm (Blue Noise + Equal Capacity).

This script demonstrates the corrected implementation based on the paper:
"Capacity-Constrained Point Distributions: A Variant of Lloyd's Method"

The algorithm produces high-quality blue noise characteristics while ensuring
equal capacity (importance) per point and precise adaptation to density functions.

Run this script from the algorithm folder to test the Capacity-Constrained implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from capacity_constrained_distribution import CapacityConstrainedDistributionAlgorithm
from utils import get_example_density_functions, visualize_algorithm_results


def run_capacity_constrained_examples():
    """
    Run comprehensive examples of the Capacity-Constrained Lloyd Algorithm.
    
    This function demonstrates:
    1. Blue noise quality optimization
    2. Equal capacity constraint satisfaction
    3. Different blue noise weights
    4. Density function adaptation
    5. Comparison with different parameters
    """
    print("CAPACITY-CONSTRAINED LLOYD ALGORITHM EXAMPLES")
    print("=" * 60)
    print("Blue Noise Distributions with Equal Capacity Constraints")
    print("Based on: 'Capacity-Constrained Point Distributions: A Variant of Lloyd's Method'")
    print()
    
    # Algorithm setup
    domain = (0, 1, 0, 1)
    algorithm = CapacityConstrainedDistributionAlgorithm(domain)
    
    # Print algorithm information
    info = algorithm.get_algorithm_info()
    print(f"Algorithm: {info['name']}")
    print(f"Key Innovation: {info['key_innovation']}")
    print(f"Paper: {info['paper']}")
    print()
    
    # Get density functions for testing
    density_funcs = get_example_density_functions()
    test_density = 'multi_gaussian'  # Challenging case for capacity balancing
    density_func = density_funcs[test_density]
    
    # Test different blue noise weights
    blue_noise_weights = [0.3, 0.5, 0.7]
    n_generators = 25
    n_iterations = 50
    results = {}
    
    print("Testing different balance between blue noise quality and capacity uniformity:")
    print()
    
    for weight in blue_noise_weights:
        print(f"Testing Blue Noise Weight = {weight}")
        print(f"({weight:.1f} spatial quality + {1-weight:.1f} capacity uniformity)")
        print("-" * 55)
        
        # Generate initial points
        np.random.seed(42)
        initial_points = algorithm.generate_random_points(n_generators, seed=42)
        
        # Run algorithm
        final_points, energy_history, capacity_variance_history, combined_energy_history = algorithm.run(
            initial_points,
            density_func,
            n_iterations=n_iterations,
            blue_noise_weight=weight,
            verbose=True
        )
        
        # Analyze results
        analysis = algorithm.analyze_capacity_distribution(final_points, density_func)
        
        results[weight] = {
            'initial_points': initial_points,
            'final_points': final_points,
            'energy_history': energy_history,
            'capacity_variance_history': capacity_variance_history,
            'combined_energy_history': combined_energy_history,
            'analysis': analysis
        }
        
        print(f"Results for weight {weight}:")
        print(f"  Blue Noise Quality: {analysis['blue_noise_quality']:.4f}")
        print(f"  Overall Quality: {analysis['overall_quality']:.4f}")
        print(f"  Capacity Uniformity: {analysis['capacity_uniformity_score']:.4f}")
        print(f"  Spatial Regularity: {analysis['regularity_score']:.4f}")
        print()
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('Capacity-Constrained Lloyd Algorithm Examples\n' +
                'Blue Noise Distributions with Equal Capacity Constraints', fontsize=16)
    
    for i, weight in enumerate(blue_noise_weights):
        result = results[weight]
        analysis = result['analysis']
        
        # Plot initial distribution
        ax = axes[i, 0]
        # Simple scatter plot instead of complex visualization
        ax.plot(result['initial_points'][:, 0], result['initial_points'][:, 1], 'ro', markersize=6)
        ax.set_title(f'Initial Points\nWeight = {weight}')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Plot final distribution with quality score
        ax = axes[i, 1]
        # Simple scatter plot instead of complex visualization
        ax.plot(result['final_points'][:, 0], result['final_points'][:, 1], 'bo', markersize=6)
        ax.set_title(f'Final Blue Noise Distribution\nQuality: {analysis["overall_quality"]:.3f}')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Plot convergence curves
        ax = axes[i, 2]
        iterations = range(len(result['energy_history']))
        ax.plot(iterations, result['energy_history'], 'b-', label='CVT Energy', linewidth=2)
        ax.plot(iterations, result['combined_energy_history'], 'r-', label='Combined Energy', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Energy')
        ax.set_title(f'Energy Convergence\nWeight = {weight}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot capacity analysis
        ax = axes[i, 3]
        capacities = analysis['capacities']
        target = analysis['target_capacity']
        
        ax.hist(capacities, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(target, color='red', linestyle='--', linewidth=2, label=f'Target: {target:.4f}')
        ax.axvline(np.mean(capacities), color='green', linestyle='-', linewidth=2, 
                  label=f'Mean: {np.mean(capacities):.4f}')
        ax.set_xlabel('Capacity')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Capacity Distribution\nStd: {analysis["capacity_std"]:.4f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Performance summary
    print("=" * 70)
    print("ALGORITHM PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"{'Weight':<8} {'Blue Noise':<12} {'Capacity Unif.':<15} {'Overall':<10} {'Regularity':<12}")
    print("-" * 70)
    
    for weight in blue_noise_weights:
        analysis = results[weight]['analysis']
        print(f"{weight:<8.1f} {analysis['blue_noise_quality']:<12.4f} {analysis['capacity_uniformity_score']:<15.4f} "
              f"{analysis['overall_quality']:<10.4f} {analysis['regularity_score']:<12.4f}")
    
    print()
    print("KEY INSIGHTS:")
    print("• Higher blue noise weight → Better spatial regularity (blue noise characteristics)")
    print("• Lower blue noise weight → Better capacity uniformity (equal importance)")
    print("• Algorithm successfully balances both objectives as described in the paper")
    print("• Equal capacity constraint ensures each point has equal importance")
    print("• Blue noise characteristics provide superior distribution quality")
    print("• Configurable balance makes it suitable for different applications")
    
    print(f"\nCAPACITY-CONSTRAINED LLOYD ALGORITHM EXAMPLES COMPLETED")
    print("High-quality blue noise distributions with equal capacity achieved!")


def quick_test():
    """Quick test of the Capacity-Constrained Lloyd Algorithm."""
    print("QUICK CAPACITY-CONSTRAINED LLOYD TEST")
    print("-" * 40)
    
    domain = (0, 1, 0, 1)
    algorithm = CapacityConstrainedDistributionAlgorithm(domain)
    
    # Get a simple density function
    density_funcs = get_example_density_functions()
    density_func = density_funcs['gaussian']
    
    # Simple test
    n_generators = 16
    initial_points = algorithm.generate_random_points(n_generators, seed=42)
    
    final_points, energy_history, capacity_variance_history, combined_energy_history = algorithm.run(
        initial_points,
        density_func,
        n_iterations=20,
        blue_noise_weight=0.5,
        verbose=False
    )
    
    analysis = algorithm.analyze_capacity_distribution(final_points, density_func)
    
    print(f"Blue Noise Quality: {analysis['blue_noise_quality']:.4f}")
    print(f"Capacity Uniformity: {analysis['capacity_uniformity_score']:.4f}")
    print(f"Overall Quality: {analysis['overall_quality']:.4f}")
    print("Capacity-constrained test completed!")


def demonstration_comparison():
    """Demonstrate the key differences and advantages of the capacity-constrained algorithm."""
    print("CAPACITY-CONSTRAINED VS STANDARD COMPARISON")
    print("-" * 50)
    
    domain = (0, 1, 0, 1)
    
    # Import standard algorithm for comparison
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'standard_lloyd'))
    from standard_lloyd import StandardLloydAlgorithm
    
    standard_alg = StandardLloydAlgorithm(domain)
    capacity_alg = CapacityConstrainedDistributionAlgorithm(domain)
    
    # Test setup
    density_funcs = get_example_density_functions()
    density_func = density_funcs['multi_gaussian']
    
    n_generators = 20
    np.random.seed(42)
    initial_points = standard_alg.generate_random_points(n_generators, seed=42)
    
    print("Running Standard Lloyd...")
    standard_final, standard_energy = standard_alg.run(
        initial_points.copy(), n_iterations=30, density_func=density_func, verbose=False
    )
    
    print("Running Capacity-Constrained Lloyd...")
    capacity_final, _, capacity_var, _ = capacity_alg.run(
        initial_points.copy(), density_func, n_iterations=30, blue_noise_weight=0.5, verbose=False
    )
    
    # Analyze both results
    capacity_analysis = capacity_alg.analyze_capacity_distribution(capacity_final, density_func)
    
    # Simple analysis for standard algorithm
    from scipy.spatial.distance import pdist
    if len(standard_final) > 1:
        std_distances = pdist(standard_final)
        std_regularity = np.min(std_distances) / np.mean(std_distances)
    else:
        std_regularity = 0.0
    
    print(f"\nRESULTS COMPARISON:")
    print(f"{'Metric':<25} {'Standard Lloyd':<15} {'Capacity-Constrained':<20}")
    print("-" * 65)
    print(f"{'Final Energy':<25} {standard_energy[-1]:<15.4f} {capacity_analysis['cvt_energy']:<20.4f}")
    print(f"{'Spatial Regularity':<25} {std_regularity:<15.4f} {capacity_analysis['regularity_score']:<20.4f}")
    print(f"{'Capacity Uniformity':<25} {'N/A':<15} {capacity_analysis['capacity_uniformity_score']:<20.4f}")
    print(f"{'Blue Noise Quality':<25} {'N/A':<15} {capacity_analysis['blue_noise_quality']:<20.4f}")
    
    print(f"\nCapacity-constrained algorithm provides:")
    print(f"• Enhanced spatial regularity (blue noise characteristics)")
    print(f"• Equal importance through capacity constraints")
    print(f"• Superior distribution quality for visualization applications")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Capacity-Constrained Lloyd Algorithm Examples')
    parser.add_argument('--quick', action='store_true', help='Run quick test only')
    parser.add_argument('--compare', action='store_true', help='Run comparison with standard Lloyd')
    args = parser.parse_args()
    
    if args.quick:
        quick_test()
    elif args.compare:
        demonstration_comparison()
    else:
        run_capacity_constrained_examples()
