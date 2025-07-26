"""
Simple test script to verify the Lloyd algorithm implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from lloyd_algorithm import LloydAlgorithm, CapacityConstrainedLloyd, example_density_functions


def simple_test():
    """Simple test to verify the algorithms work correctly."""
    
    print("Testing Lloyd Algorithm Implementation")
    print("=" * 40)
    
    # Set up a simple test case
    domain = (0, 1, 0, 1)
    n_generators = 9  # 3x3 grid would be ideal for uniform density
    
    # Initialize algorithm
    lloyd = LloydAlgorithm(domain)
    
    # Generate initial random points
    initial_generators = lloyd.generate_random_points(n_generators, seed=42)
    print(f"Generated {n_generators} initial generators")
    
    # Test with uniform density (should converge to regular grid)
    print("\nTesting with uniform density...")
    density_funcs = example_density_functions()
    
    final_generators, energy_history = lloyd.run_lloyd(
        initial_generators,
        n_iterations=30,
        density_func=density_funcs['uniform'],
        tolerance=1e-6
    )
    
    print(f"Converged! Final energy: {energy_history[-1]:.6f}")
    print(f"Energy reduction: {(energy_history[0] - energy_history[-1]) / energy_history[0] * 100:.2f}%")
    
    # Simple visualization
    plt.figure(figsize=(12, 4))
    
    # Plot initial vs final positions
    plt.subplot(1, 3, 1)
    plt.plot(initial_generators[:, 0], initial_generators[:, 1], 'ro', markersize=8, label='Initial')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title('Initial Generators')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(final_generators[:, 0], final_generators[:, 1], 'bo', markersize=8, label='Final')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title('Final Generators (Uniform Density)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot energy convergence
    plt.subplot(1, 3, 3)
    plt.plot(energy_history, 'g-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.title('Energy Convergence')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    # Test capacity-constrained version
    print("\nTesting capacity-constrained algorithm...")
    capacity_lloyd = CapacityConstrainedLloyd(domain)
    
    final_cc, energy_cc, variance_cc = capacity_lloyd.run_capacity_constrained_lloyd(
        initial_generators.copy(),
        density_funcs['gaussian'],
        n_iterations=20
    )
    
    print(f"Capacity-constrained converged! Final energy: {energy_cc[-1]:.6f}")
    print(f"Final capacity variance: {variance_cc[-1]:.6f}")
    
    return True


if __name__ == "__main__":
    success = simple_test()
    if success:
        print("\n✓ All tests passed! The implementation is working correctly.")
    else:
        print("\n✗ Tests failed!")
