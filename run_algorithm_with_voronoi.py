#!/usr/bin/env python3
"""
Run Algorithms with Voronoi Output

This script provides a simple interface to run either algorithm and get:
1. Final generator positions
2. Voronoi cell tessellation visualization
3. Both outputs saved as images

Usage: python run_algorithm_with_voronoi.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from algorithms.standard_lloyd import StandardLloydAlgorithm
from algorithms.capacity_constrained import CapacityConstrainedDistributionAlgorithm
from utils import get_example_density_functions

def ensure_output_dir():
    """Ensure the output directory exists."""
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def create_voronoi_plot(points, title, density_func=None, domain=(0, 1, 0, 1)):
    """
    Create a Voronoi diagram plot with the given points.
    
    Args:
        points: Generator positions
        title: Plot title
        density_func: Optional density function to show as background
        domain: Domain bounds (xmin, xmax, ymin, ymax)
    
    Returns:
        matplotlib figure and axes
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    xmin, xmax, ymin, ymax = domain
    
    # Left plot: Points only
    ax1.scatter(points[:, 0], points[:, 1], c='red', s=80, 
               alpha=0.8, edgecolors='black', linewidth=1, zorder=5)
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    ax1.set_aspect('equal')
    ax1.set_title(f'{title}\nFinal Generator Positions')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    
    # Add density background if provided
    if density_func is not None:
        x = np.linspace(xmin, xmax, 100)
        y = np.linspace(ymin, ymax, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.array([[density_func(xi, yi) for xi in x] for yi in y])
        ax1.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.3)
    
    # Right plot: Voronoi diagram
    # Add boundary points to help with edge cells
    boundary_points = np.array([
        [xmin-0.1, ymin-0.1], [xmin-0.1, ymax+0.1], 
        [xmax+0.1, ymin-0.1], [xmax+0.1, ymax+0.1],
        [xmin-0.1, (ymin+ymax)/2], [xmax+0.1, (ymin+ymax)/2],
        [(xmin+xmax)/2, ymin-0.1], [(xmin+xmax)/2, ymax+0.1]
    ])
    
    extended_points = np.vstack([points, boundary_points])
    vor = Voronoi(extended_points)
    
    # Plot Voronoi diagram
    voronoi_plot_2d(vor, ax=ax2, show_vertices=False, line_colors='blue', 
                    line_width=1, point_size=0)
    
    # Overlay the actual generator points
    ax2.scatter(points[:, 0], points[:, 1], c='red', s=80, 
               alpha=0.9, edgecolors='black', linewidth=1, zorder=5)
    
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
    ax2.set_aspect('equal')
    ax2.set_title(f'{title}\nVoronoi Tessellation')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    
    # Add density background if provided
    if density_func is not None:
        ax2.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.2)
    
    plt.tight_layout()
    return fig, (ax1, ax2)

def run_standard_lloyd(n_generators=20, density_name='multi_gaussian', 
                      n_iterations=50, verbose=True):
    """
    Run Standard Lloyd Algorithm and return results.
    
    Args:
        n_generators: Number of generator points
        density_name: Name of density function ('uniform', 'gaussian', 'multi_gaussian', 'linear')
        n_iterations: Number of iterations
        verbose: Whether to print progress
    
    Returns:
        tuple: (final_points, energy_history, density_func)
    """
    # Setup
    domain = (0, 1, 0, 1)
    algorithm = StandardLloydAlgorithm(domain)
    
    # Get density function
    density_funcs = get_example_density_functions()
    density_func = density_funcs[density_name]
    
    # Generate initial points
    np.random.seed(42)
    initial_points = algorithm.generate_random_points(n_generators)
    
    if verbose:
        print(f"Running Standard Lloyd Algorithm")
        print(f"Generators: {n_generators}")
        print(f"Density: {density_name}")
        print(f"Iterations: {n_iterations}")
        print("=" * 50)
    
    # Run algorithm
    final_points, energy_history = algorithm.run(
        initial_points,
        n_iterations=n_iterations,
        density_func=density_func,
        verbose=verbose
    )
    
    if verbose:
        print(f"Final energy: {energy_history[-1]:.6f}")
        print(f"Energy reduction: {((energy_history[0] - energy_history[-1])/energy_history[0]*100):.2f}%")
    
    return final_points, energy_history, density_func

def run_capacity_constrained(n_generators=20, density_name='multi_gaussian', 
                           n_iterations=50, blue_noise_weight=0.5, verbose=True):
    """
    Run Capacity-Constrained Algorithm and return results.
    
    Args:
        n_generators: Number of generator points
        density_name: Name of density function ('uniform', 'gaussian', 'multi_gaussian', 'linear')
        n_iterations: Number of iterations
        blue_noise_weight: Weight for blue noise vs capacity uniformity (0.0 to 1.0)
        verbose: Whether to print progress
    
    Returns:
        tuple: (final_points, energy_history, capacity_variance_history, combined_energy_history, density_func)
    """
    # Setup
    domain = (0, 1, 0, 1)
    algorithm = CapacityConstrainedDistributionAlgorithm(domain)
    
    # Get density function
    density_funcs = get_example_density_functions()
    density_func = density_funcs[density_name]
    
    # Generate initial points
    np.random.seed(42)
    initial_points = algorithm.generate_random_points(n_generators)
    
    if verbose:
        print(f"Running Capacity-Constrained Distribution Algorithm")
        print(f"Generators: {n_generators}")
        print(f"Density: {density_name}")
        print(f"Blue noise weight: {blue_noise_weight}")
        print(f"Iterations: {n_iterations}")
        print("=" * 60)
    
    # Run algorithm
    final_points, energy_history, capacity_variance_history, combined_energy_history = algorithm.run(
        initial_points,
        density_func,
        n_iterations=n_iterations,
        blue_noise_weight=blue_noise_weight,
        verbose=verbose
    )
    
    if verbose:
        analysis = algorithm.analyze_capacity_distribution(final_points, density_func)
        print(f"Final CVT energy: {analysis['cvt_energy']:.6f}")
        print(f"Final combined energy: {analysis['combined_energy']:.6f}")
        print(f"Blue noise quality: {analysis['blue_noise_quality']:.4f}")
        print(f"Capacity std deviation: {analysis['capacity_std']:.6f}")
    
    return final_points, energy_history, capacity_variance_history, combined_energy_history, density_func

def save_algorithm_results(algorithm_name, final_points, density_func, 
                          density_name, output_dir, additional_info=""):
    """Save algorithm results with both point positions and Voronoi diagram."""
    
    # Create the plot
    title = f"{algorithm_name} - {density_name.replace('_', ' ').title()}"
    if additional_info:
        title += f" ({additional_info})"
    
    fig, (ax1, ax2) = create_voronoi_plot(final_points, title, density_func)
    
    # Save the combined plot
    safe_name = f"{algorithm_name.lower().replace(' ', '_')}_{density_name}"
    if additional_info:
        safe_name += f"_{additional_info.replace(' ', '_').replace('=', '').replace('.', '')}"
    
    filename = os.path.join(output_dir, f"{safe_name}_results.png")
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Results saved to: {filename}")
    
    # Also save points as text file
    points_filename = os.path.join(output_dir, f"{safe_name}_points.txt")
    np.savetxt(points_filename, final_points, fmt='%.6f', delimiter='\t', 
               header='x\ty\t(Final generator positions)')
    print(f"Points saved to: {points_filename}")
    
    return filename, points_filename

def interactive_menu():
    """Interactive menu for running algorithms."""
    output_dir = ensure_output_dir()
    
    while True:
        print("\n" + "=" * 60)
        print("ALGORITHM RUNNER WITH VORONOI OUTPUT")
        print("=" * 60)
        print("1. Run Standard Lloyd Algorithm")
        print("2. Run Capacity-Constrained Distribution Algorithm")
        print("3. Run Both Algorithms (Same Configuration)")
        print("4. Custom Configuration")
        print("0. Exit")
        print("=" * 60)
        
        choice = input("Enter your choice (0-4): ").strip()
        
        if choice == '0':
            print("Goodbye!")
            break
        
        elif choice == '1':
            print("\nRunning Standard Lloyd Algorithm")
            print("-" * 40)
            final_points, energy_history, density_func = run_standard_lloyd()
            save_algorithm_results("Standard Lloyd", final_points, density_func, 
                                 "multi_gaussian", output_dir)
        
        elif choice == '2':
            print("\nRunning Capacity-Constrained Distribution Algorithm")
            print("-" * 50)
            final_points, *_, density_func = run_capacity_constrained()
            save_algorithm_results("Capacity-Constrained Distribution", final_points, 
                                 density_func, "multi_gaussian", output_dir, "weight_0.5")
        
        elif choice == '3':
            print("\nRunning Both Algorithms")
            print("-" * 30)
            
            # Standard Lloyd
            print("\n1. Standard Lloyd:")
            final_points_std, energy_history_std, density_func = run_standard_lloyd()
            save_algorithm_results("Standard Lloyd", final_points_std, density_func, 
                                 "multi_gaussian", output_dir)
            
            # Capacity-Constrained
            print("\n2. Capacity-Constrained:")
            final_points_cap, *_ = run_capacity_constrained()
            save_algorithm_results("Capacity-Constrained Distribution", final_points_cap, 
                                 density_func, "multi_gaussian", output_dir, "weight_0.5")
            
            # Create comparison plot
            create_side_by_side_comparison(final_points_std, final_points_cap, 
                                         density_func, output_dir)
        
        elif choice == '4':
            run_custom_configuration(output_dir)
        
        else:
            print("Invalid choice. Please try again.")

def create_side_by_side_comparison(points_std, points_cap, density_func, output_dir):
    """Create side-by-side comparison of both algorithms."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    domain = (0, 1, 0, 1)
    xmin, xmax, ymin, ymax = domain
    
    # Background density
    x = np.linspace(xmin, xmax, 100)
    y = np.linspace(ymin, ymax, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[density_func(xi, yi) for xi in x] for yi in y])
    
    # Standard Lloyd - Points
    axes[0, 0].contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.3)
    axes[0, 0].scatter(points_std[:, 0], points_std[:, 1], c='red', s=80, 
                      alpha=0.8, edgecolors='black', linewidth=1)
    axes[0, 0].set_xlim(xmin, xmax)
    axes[0, 0].set_ylim(ymin, ymax)
    axes[0, 0].set_aspect('equal')
    axes[0, 0].set_title('Standard Lloyd - Generator Positions')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Standard Lloyd - Voronoi
    boundary_points = np.array([
        [xmin-0.1, ymin-0.1], [xmin-0.1, ymax+0.1], 
        [xmax+0.1, ymin-0.1], [xmax+0.1, ymax+0.1],
        [xmin-0.1, (ymin+ymax)/2], [xmax+0.1, (ymin+ymax)/2],
        [(xmin+xmax)/2, ymin-0.1], [(xmin+xmax)/2, ymax+0.1]
    ])
    
    extended_points_std = np.vstack([points_std, boundary_points])
    vor_std = Voronoi(extended_points_std)
    voronoi_plot_2d(vor_std, ax=axes[0, 1], show_vertices=False, 
                    line_colors='blue', line_width=1, point_size=0)
    axes[0, 1].scatter(points_std[:, 0], points_std[:, 1], c='red', s=80, 
                      alpha=0.9, edgecolors='black', linewidth=1, zorder=5)
    axes[0, 1].set_xlim(xmin, xmax)
    axes[0, 1].set_ylim(ymin, ymax)
    axes[0, 1].set_aspect('equal')
    axes[0, 1].set_title('Standard Lloyd - Voronoi Tessellation')
    
    # Capacity-Constrained - Points
    axes[1, 0].contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.3)
    axes[1, 0].scatter(points_cap[:, 0], points_cap[:, 1], c='blue', s=80, 
                      alpha=0.8, edgecolors='black', linewidth=1)
    axes[1, 0].set_xlim(xmin, xmax)
    axes[1, 0].set_ylim(ymin, ymax)
    axes[1, 0].set_aspect('equal')
    axes[1, 0].set_title('Capacity-Constrained - Generator Positions')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Capacity-Constrained - Voronoi
    extended_points_cap = np.vstack([points_cap, boundary_points])
    vor_cap = Voronoi(extended_points_cap)
    voronoi_plot_2d(vor_cap, ax=axes[1, 1], show_vertices=False, 
                    line_colors='green', line_width=1, point_size=0)
    axes[1, 1].scatter(points_cap[:, 0], points_cap[:, 1], c='blue', s=80, 
                      alpha=0.9, edgecolors='black', linewidth=1, zorder=5)
    axes[1, 1].set_xlim(xmin, xmax)
    axes[1, 1].set_ylim(ymin, ymax)
    axes[1, 1].set_aspect('equal')
    axes[1, 1].set_title('Capacity-Constrained - Voronoi Tessellation')
    
    plt.suptitle('Algorithm Comparison - Generator Positions and Voronoi Cells', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filename = os.path.join(output_dir, 'algorithm_comparison_voronoi.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison plot saved to: {filename}")

def run_custom_configuration(output_dir):
    """Run algorithm with custom configuration."""
    print("\nCustom Configuration")
    print("-" * 30)
    
    # Get user inputs
    try:
        n_generators = int(input("Number of generators (default 20): ") or "20")
        n_iterations = int(input("Number of iterations (default 50): ") or "50")
        
        print("\nDensity options:")
        print("1. Uniform")
        print("2. Gaussian") 
        print("3. Multi-Gaussian (default)")
        print("4. Linear")
        density_choice = input("Choose density (1-4, default 3): ").strip() or "3"
        
        density_map = {'1': 'uniform', '2': 'gaussian', '3': 'multi_gaussian', '4': 'linear'}
        density_name = density_map.get(density_choice, 'multi_gaussian')
        
        print("\nAlgorithm options:")
        print("1. Standard Lloyd")
        print("2. Capacity-Constrained") 
        print("3. Both")
        algo_choice = input("Choose algorithm (1-3): ").strip()
        
        if algo_choice == '1':
            final_points, _, density_func = run_standard_lloyd(
                n_generators, density_name, n_iterations)
            save_algorithm_results("Standard Lloyd", final_points, density_func, 
                                 density_name, output_dir)
        
        elif algo_choice == '2':
            blue_noise_weight = float(input("Blue noise weight (0.0-1.0, default 0.5): ") or "0.5")
            final_points, *_, density_func = run_capacity_constrained(
                n_generators, density_name, n_iterations, blue_noise_weight)
            save_algorithm_results("Capacity-Constrained Distribution", final_points, 
                                 density_func, density_name, output_dir, 
                                 f"weight_{blue_noise_weight}")
        
        elif algo_choice == '3':
            blue_noise_weight = float(input("Blue noise weight for capacity-constrained (0.0-1.0, default 0.5): ") or "0.5")
            
            print("\n1. Standard Lloyd:")
            final_points_std, _, density_func = run_standard_lloyd(
                n_generators, density_name, n_iterations)
            save_algorithm_results("Standard Lloyd", final_points_std, density_func, 
                                 density_name, output_dir)
            
            print("\n2. Capacity-Constrained:")
            final_points_cap, *_ = run_capacity_constrained(
                n_generators, density_name, n_iterations, blue_noise_weight)
            save_algorithm_results("Capacity-Constrained Distribution", final_points_cap, 
                                 density_func, density_name, output_dir, 
                                 f"weight_{blue_noise_weight}")
            
            create_side_by_side_comparison(final_points_std, final_points_cap, 
                                         density_func, output_dir)
        
        else:
            print("Invalid choice.")
    
    except ValueError:
        print("Invalid input. Please enter valid numbers.")
    except KeyboardInterrupt:
        print("\nOperation cancelled.")

if __name__ == "__main__":
    interactive_menu()
