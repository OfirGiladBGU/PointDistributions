#!/usr/bin/env python3
"""
Point Distribution Algorithms with Voronoi Export

Simple tool to run algorithms and export:
1. Final point positions as TXT files
2. Voronoi tessellation diagrams as PNG files

Usage: python run_algorithms.py
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

def save_points_to_txt(points, filename, algorithm_name="Algorithm"):
    """Save points to a text file."""
    header = f"{algorithm_name} - Final Generator Positions\nFormat: x y\nTotal Points: {len(points)}"
    np.savetxt(filename, points, fmt='%.6f', delimiter='\t', header=header)
    print(f"Points saved to: {filename}")

def create_voronoi_png(points, filename, title="Voronoi Diagram", density_func=None):
    """Create and save Voronoi diagram as PNG."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    domain = (0, 1, 0, 1)
    xmin, xmax, ymin, ymax = domain
    
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
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='blue', 
                    line_width=1.5, point_size=0)
    
    # Add density background if provided
    if density_func is not None:
        x = np.linspace(xmin, xmax, 100)
        y = np.linspace(ymin, ymax, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.array([[density_func(xi, yi) for xi in x] for yi in y])
        ax.contourf(X, Y, Z, levels=15, cmap='viridis', alpha=0.2)
    
    # Overlay the actual generator points
    ax.scatter(points[:, 0], points[:, 1], c='red', s=60, 
               alpha=0.9, edgecolors='black', linewidth=1.5, zorder=5)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', format='png')
    plt.close()
    
    print(f"Voronoi diagram saved to: {filename}")

def run_standard_lloyd(n_generators=20, density_name='multi_gaussian', n_iterations=50):
    """Run Standard Lloyd Algorithm and export results."""
    print(f"\n{'='*50}")
    print("STANDARD LLOYD ALGORITHM")
    print('='*50)
    
    # Setup
    domain = (0, 1, 0, 1)
    algorithm = StandardLloydAlgorithm(domain)
    
    # Get density function
    density_funcs = get_example_density_functions()
    density_func = density_funcs[density_name]
    
    # Generate initial points
    np.random.seed(42)
    initial_points = algorithm.generate_random_points(n_generators)
    
    print(f"Generators: {n_generators}")
    print(f"Density: {density_name}")
    print(f"Iterations: {n_iterations}")
    print("-" * 30)
    
    # Run algorithm
    final_points, energy_history = algorithm.run(
        initial_points,
        n_iterations=n_iterations,
        density_func=density_func,
        verbose=True
    )
    
    print(f"Final energy: {energy_history[-1]:.6f}")
    print(f"Energy reduction: {((energy_history[0] - energy_history[-1])/energy_history[0]*100):.2f}%")
    
    return final_points, density_func

def run_capacity_constrained(n_generators=20, density_name='multi_gaussian', 
                           n_iterations=50, blue_noise_weight=0.5):
    """Run Capacity-Constrained Algorithm and export results."""
    print(f"\n{'='*60}")
    print("CAPACITY-CONSTRAINED DISTRIBUTION ALGORITHM")
    print('='*60)
    
    # Setup
    domain = (0, 1, 0, 1)
    algorithm = CapacityConstrainedDistributionAlgorithm(domain)
    
    # Get density function
    density_funcs = get_example_density_functions()
    density_func = density_funcs[density_name]
    
    # Generate initial points
    np.random.seed(42)
    initial_points = algorithm.generate_random_points(n_generators)
    
    print(f"Generators: {n_generators}")
    print(f"Density: {density_name}")
    print(f"Blue noise weight: {blue_noise_weight}")
    print(f"Iterations: {n_iterations}")
    print("-" * 40)
    
    # Run algorithm
    final_points, energy_history, capacity_variance_history, combined_energy_history = algorithm.run(
        initial_points,
        density_func,
        n_iterations=n_iterations,
        blue_noise_weight=blue_noise_weight,
        verbose=True
    )
    
    # Get analysis
    analysis = algorithm.analyze_capacity_distribution(final_points, density_func)
    print(f"Final CVT energy: {analysis['cvt_energy']:.6f}")
    print(f"Final combined energy: {analysis['combined_energy']:.6f}")
    print(f"Blue noise quality: {analysis['blue_noise_quality']:.4f}")
    print(f"Capacity std deviation: {analysis['capacity_std']:.6f}")
    
    return final_points, density_func

def main():
    """Main interface for running algorithms and exporting results."""
    output_dir = ensure_output_dir()
    
    print("="*60)
    print("POINT DISTRIBUTION ALGORITHMS WITH VORONOI EXPORT")
    print("="*60)
    print("Choose algorithm:")
    print("1. Standard Lloyd Algorithm")
    print("2. Capacity-Constrained Distribution Algorithm")
    print("3. Both Algorithms")
    print("0. Exit")
    print("="*60)
    
    while True:
        try:
            choice = input("\nEnter your choice (0-3): ").strip()
            
            if choice == '0':
                print("Goodbye!")
                break
            
            # Get parameters
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
                
            except ValueError:
                print("Invalid input. Using defaults: 20 generators, 50 iterations, multi-gaussian density")
                n_generators, n_iterations, density_name = 20, 50, 'multi_gaussian'
            
            if choice == '1':
                # Standard Lloyd
                final_points, density_func = run_standard_lloyd(n_generators, density_name, n_iterations)
                
                # Export files
                base_name = f"standard_lloyd_{density_name}_{n_generators}pts"
                txt_file = os.path.join(output_dir, f"{base_name}.txt")
                png_file = os.path.join(output_dir, f"{base_name}_voronoi.png")
                
                save_points_to_txt(final_points, txt_file, "Standard Lloyd Algorithm")
                create_voronoi_png(final_points, png_file, 
                                 f"Standard Lloyd - {density_name.replace('_', ' ').title()}", 
                                 density_func)
            
            elif choice == '2':
                # Capacity-Constrained
                try:
                    blue_noise_weight = float(input("Blue noise weight (0.0-1.0, default 0.5): ") or "0.5")
                except ValueError:
                    blue_noise_weight = 0.5
                
                final_points, density_func = run_capacity_constrained(
                    n_generators, density_name, n_iterations, blue_noise_weight)
                
                # Export files
                base_name = f"capacity_constrained_{density_name}_{n_generators}pts_w{blue_noise_weight:.1f}"
                txt_file = os.path.join(output_dir, f"{base_name}.txt")
                png_file = os.path.join(output_dir, f"{base_name}_voronoi.png")
                
                save_points_to_txt(final_points, txt_file, "Capacity-Constrained Distribution Algorithm")
                create_voronoi_png(final_points, png_file, 
                                 f"Capacity-Constrained - {density_name.replace('_', ' ').title()}", 
                                 density_func)
            
            elif choice == '3':
                # Both algorithms
                try:
                    blue_noise_weight = float(input("Blue noise weight for capacity-constrained (0.0-1.0, default 0.5): ") or "0.5")
                except ValueError:
                    blue_noise_weight = 0.5
                
                # Standard Lloyd
                final_points_std, density_func = run_standard_lloyd(n_generators, density_name, n_iterations)
                
                # Capacity-Constrained
                final_points_cap, _ = run_capacity_constrained(n_generators, density_name, n_iterations, blue_noise_weight)
                
                # Export Standard Lloyd files
                base_name_std = f"standard_lloyd_{density_name}_{n_generators}pts"
                txt_file_std = os.path.join(output_dir, f"{base_name_std}.txt")
                png_file_std = os.path.join(output_dir, f"{base_name_std}_voronoi.png")
                
                save_points_to_txt(final_points_std, txt_file_std, "Standard Lloyd Algorithm")
                create_voronoi_png(final_points_std, png_file_std, 
                                 f"Standard Lloyd - {density_name.replace('_', ' ').title()}", 
                                 density_func)
                
                # Export Capacity-Constrained files
                base_name_cap = f"capacity_constrained_{density_name}_{n_generators}pts_w{blue_noise_weight:.1f}"
                txt_file_cap = os.path.join(output_dir, f"{base_name_cap}.txt")
                png_file_cap = os.path.join(output_dir, f"{base_name_cap}_voronoi.png")
                
                save_points_to_txt(final_points_cap, txt_file_cap, "Capacity-Constrained Distribution Algorithm")
                create_voronoi_png(final_points_cap, png_file_cap, 
                                 f"Capacity-Constrained - {density_name.replace('_', ' ').title()}", 
                                 density_func)
                
                print(f"\n{'='*60}")
                print("BOTH ALGORITHMS COMPLETED")
                print(f"All files saved to: {output_dir}")
                print("="*60)
            
            else:
                print("Invalid choice. Please try again.")
            
            print(f"\nResults saved to: {output_dir}")
            print("="*60)
            
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main()
