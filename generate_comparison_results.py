#!/usr/bin/env python3
"""
Generate Algorithm Comparison Results

This script generates comprehensive comparison between Standard Lloyd and 
Capacity-Constrained Lloyd algorithms with detailed analysis and visualizations
saved to the output directory.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

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

def generate_comparison_results():
    """Generate comprehensive algorithm comparison results."""
    
    print("=" * 70)
    print("GENERATING ALGORITHM COMPARISON RESULTS")
    print("=" * 70)
    
    output_dir = ensure_output_dir()
    
    # Algorithm setup
    domain = (0, 1, 0, 1)
    standard_algorithm = StandardLloydAlgorithm(domain)
    capacity_algorithm = CapacityConstrainedDistributionAlgorithm(domain)
    
    # Get density functions
    density_funcs = get_example_density_functions()
    
    # Test configurations
    configs = [
        {
            'name': 'Uniform Density',
            'density': density_funcs['uniform'],
            'n_generators': 16
        },
        {
            'name': 'Gaussian Density', 
            'density': density_funcs['gaussian'],
            'n_generators': 16
        },
        {
            'name': 'Multi-Gaussian Density',
            'density': density_funcs['multi_gaussian'],
            'n_generators': 20
        },
        {
            'name': 'Linear Density',
            'density': density_funcs['linear'],
            'n_generators': 24
        }
    ]
    
    # Blue noise weights to test for capacity-constrained
    blue_noise_weights = [0.3, 0.5, 0.7]
    
    comparison_results = []
    
    for config_idx, config in enumerate(configs, 1):
        print(f"\n{config_idx}. {config['name']}")
        print("=" * 50)
        
        config_results = {
            'config': config,
            'standard': None,
            'capacity_constrained': []
        }
        
        # Generate initial points (same for both algorithms)
        np.random.seed(42)
        initial_points_base = np.random.rand(config['n_generators'], 2)
        
        # Run Standard Lloyd Algorithm
        print("Running Standard Lloyd Algorithm...")
        start_time = time.time()
        initial_points = initial_points_base.copy()
        
        final_points_std, energy_history_std = standard_algorithm.run(
            initial_points,
            n_iterations=50,
            density_func=config['density'],
            verbose=False
        )
        
        std_time = time.time() - start_time
        # Create analysis results
        analysis_std = {
            'final_energy': energy_history_std[-1],
            'convergence': 'Energy reduced',
            'iterations': len(energy_history_std)
        }
        
        config_results['standard'] = {
            'final_points': final_points_std,
            'energy_history': energy_history_std,
            'analysis': analysis_std,
            'runtime': std_time
        }
        
        print(f"  Final Energy: {analysis_std['final_energy']:.6f}")
        print(f"  Runtime: {std_time:.2f}s")
        
        # Run Capacity-Constrained Lloyd Algorithm with different weights
        for weight in blue_noise_weights:
            print(f"Running Capacity-Constrained Algorithm (weight={weight})...")
            start_time = time.time()
            initial_points = initial_points_base.copy()
            
            final_points_cap, energy_history_cap, cap_var_history, combined_energy_history = capacity_algorithm.run(
                initial_points,
                config['density'],
                n_iterations=50,
                blue_noise_weight=weight,
                verbose=False
            )
            
            cap_time = time.time() - start_time
            analysis_cap = capacity_algorithm.analyze_capacity_distribution(final_points_cap, config['density'])
            
            cap_result = {
                'final_points': final_points_cap,
                'energy_history': energy_history_cap,
                'capacity_variance_history': cap_var_history,
                'combined_energy_history': combined_energy_history,
                'analysis': analysis_cap,
                'runtime': cap_time,
                'blue_noise_weight': weight
            }
            
            config_results['capacity_constrained'].append(cap_result)
            
            print(f"  CVT Energy: {analysis_cap['cvt_energy']:.6f}")
            print(f"  Combined Energy: {analysis_cap['combined_energy']:.6f}")
            print(f"  Blue Noise Quality: {analysis_cap['blue_noise_quality']:.4f}")
            print(f"  Runtime: {cap_time:.2f}s")
        
        comparison_results.append(config_results)
        
        # Create individual comparison plot
        create_individual_comparison_plot(config_results, output_dir)
    
    # Create comprehensive comparison plots
    create_comprehensive_comparison(comparison_results, output_dir)
    create_performance_analysis(comparison_results, output_dir)
    create_quality_metrics_comparison(comparison_results, output_dir)
    
    # Print comprehensive summary
    print_comprehensive_summary(comparison_results, output_dir)

def create_individual_comparison_plot(config_results, output_dir):
    """Create individual comparison plot for one density configuration."""
    config = config_results['config']
    std_result = config_results['standard']
    cap_results = config_results['capacity_constrained']
    
    # Create subplot layout: 2 rows, 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Top row: Point distributions
    # Standard Lloyd
    std_points = std_result['final_points']
    axes[0, 0].scatter(std_points[:, 0], std_points[:, 1], c='red', s=50, 
                      alpha=0.8, edgecolors='black', linewidth=0.5)
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_aspect('equal')
    axes[0, 0].set_title(f'Standard Lloyd\nEnergy: {std_result["analysis"]["final_energy"]:.4f}')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Capacity-Constrained with different weights
    colors = ['blue', 'green', 'purple']
    for i, (cap_result, color) in enumerate(zip(cap_results, colors)):
        cap_points = cap_result['final_points']
        axes[0, i+1].scatter(cap_points[:, 0], cap_points[:, 1], c=color, s=50, 
                            alpha=0.8, edgecolors='black', linewidth=0.5)
        axes[0, i+1].set_xlim(0, 1)
        axes[0, i+1].set_ylim(0, 1)
        axes[0, i+1].set_aspect('equal')
        weight = cap_result['blue_noise_weight']
        quality = cap_result['analysis']['blue_noise_quality']
        axes[0, i+1].set_title(f'Capacity-Constrained (w={weight})\nBlue Noise: {quality:.4f}')
        axes[0, i+1].grid(True, alpha=0.3)
    
    # Bottom row: Energy convergence plots
    # Standard Lloyd energy
    axes[1, 0].plot(std_result['energy_history'], 'r-', linewidth=2, label='CVT Energy')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Energy')
    axes[1, 0].set_title('Standard Lloyd Convergence')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Capacity-Constrained energy convergence
    for i, (cap_result, color) in enumerate(zip(cap_results, colors)):
        axes[1, i+1].plot(cap_result['energy_history'], color=color, linewidth=2, 
                         label='CVT Energy', alpha=0.7)
        axes[1, i+1].plot(cap_result['combined_energy_history'], color=color, 
                         linewidth=2, linestyle='--', label='Combined Energy')
        axes[1, i+1].set_xlabel('Iteration')
        axes[1, i+1].set_ylabel('Energy')
        weight = cap_result['blue_noise_weight']
        axes[1, i+1].set_title(f'Capacity-Constrained (w={weight})')
        axes[1, i+1].grid(True, alpha=0.3)
        axes[1, i+1].legend()
    
    plt.suptitle(f'Algorithm Comparison - {config["name"]}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    safe_name = config['name'].lower().replace(' ', '_').replace('-', '_')
    filename = os.path.join(output_dir, f'comparison_{safe_name}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def create_comprehensive_comparison(comparison_results, output_dir):
    """Create comprehensive comparison overview."""
    n_configs = len(comparison_results)
    fig, axes = plt.subplots(n_configs, 4, figsize=(20, 5*n_configs))
    
    if n_configs == 1:
        axes = axes.reshape(1, -1)
    
    for row, config_result in enumerate(comparison_results):
        config = config_result['config']
        std_result = config_result['standard']
        
        # Show standard algorithm result
        std_points = std_result['final_points']
        axes[row, 0].scatter(std_points[:, 0], std_points[:, 1], c='red', s=40, 
                           alpha=0.8, edgecolors='black', linewidth=0.5)
        axes[row, 0].set_xlim(0, 1)
        axes[row, 0].set_ylim(0, 1)
        axes[row, 0].set_aspect('equal')
        axes[row, 0].set_title(f'{config["name"]}\nStandard Lloyd')
        axes[row, 0].grid(True, alpha=0.3)
        
        # Show best capacity-constrained results (highest blue noise quality)
        best_cap_result = max(config_result['capacity_constrained'], 
                             key=lambda x: x['analysis']['blue_noise_quality'])
        
        for col, cap_result in enumerate(config_result['capacity_constrained'], 1):
            cap_points = cap_result['final_points']
            color = 'blue' if cap_result == best_cap_result else 'lightblue'
            axes[row, col].scatter(cap_points[:, 0], cap_points[:, 1], c=color, s=40, 
                                 alpha=0.8, edgecolors='black', linewidth=0.5)
            axes[row, col].set_xlim(0, 1)
            axes[row, col].set_ylim(0, 1)
            axes[row, col].set_aspect('equal')
            weight = cap_result['blue_noise_weight']
            quality = cap_result['analysis']['blue_noise_quality']
            title = f'Capacity-Constrained\nw={weight}, Quality={quality:.3f}'
            if cap_result == best_cap_result:
                title += ' ⭐'
            axes[row, col].set_title(title)
            axes[row, col].grid(True, alpha=0.3)
    
    plt.suptitle('Comprehensive Algorithm Comparison - All Configurations', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filename = os.path.join(output_dir, 'comprehensive_comparison.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def create_performance_analysis(comparison_results, output_dir):
    """Create performance analysis plots."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract data
    config_names = [r['config']['name'] for r in comparison_results]
    std_energies = [r['standard']['analysis']['final_energy'] for r in comparison_results]
    std_runtimes = [r['standard']['runtime'] for r in comparison_results]
    
    # For capacity-constrained, use best blue noise quality results
    best_cap_results = []
    for config_result in comparison_results:
        best_result = max(config_result['capacity_constrained'], 
                         key=lambda x: x['analysis']['blue_noise_quality'])
        best_cap_results.append(best_result)
    
    cap_cvt_energies = [r['analysis']['cvt_energy'] for r in best_cap_results]
    cap_combined_energies = [r['analysis']['combined_energy'] for r in best_cap_results]
    cap_runtimes = [r['runtime'] for r in best_cap_results]
    blue_noise_qualities = [r['analysis']['blue_noise_quality'] for r in best_cap_results]
    
    # Plot 1: Energy comparison
    x = np.arange(len(config_names))
    width = 0.25
    
    ax1.bar(x - width, std_energies, width, label='Standard Lloyd', color='red', alpha=0.7)
    ax1.bar(x, cap_cvt_energies, width, label='Capacity-Constrained (CVT)', color='blue', alpha=0.7)
    ax1.bar(x + width, cap_combined_energies, width, label='Capacity-Constrained (Combined)', color='green', alpha=0.7)
    
    ax1.set_xlabel('Density Configuration')
    ax1.set_ylabel('Final Energy')
    ax1.set_title('Final Energy Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(config_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Runtime comparison
    ax2.bar(x - width/2, std_runtimes, width, label='Standard Lloyd', color='red', alpha=0.7)
    ax2.bar(x + width/2, cap_runtimes, width, label='Capacity-Constrained', color='blue', alpha=0.7)
    
    ax2.set_xlabel('Density Configuration')
    ax2.set_ylabel('Runtime (seconds)')
    ax2.set_title('Runtime Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(config_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Blue noise quality by configuration
    ax3.bar(range(len(config_names)), blue_noise_qualities, color='purple', alpha=0.7)
    ax3.set_xlabel('Density Configuration')
    ax3.set_ylabel('Blue Noise Quality')
    ax3.set_title('Blue Noise Quality (Capacity-Constrained)')
    ax3.set_xticks(range(len(config_names)))
    ax3.set_xticklabels(config_names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Energy efficiency (energy/runtime)
    std_efficiency = [e/r for e, r in zip(std_energies, std_runtimes)]
    cap_efficiency = [e/r for e, r in zip(cap_combined_energies, cap_runtimes)]
    
    ax4.bar(x - width/2, std_efficiency, width, label='Standard Lloyd', color='red', alpha=0.7)
    ax4.bar(x + width/2, cap_efficiency, width, label='Capacity-Constrained', color='blue', alpha=0.7)
    
    ax4.set_xlabel('Density Configuration')
    ax4.set_ylabel('Energy/Runtime (efficiency)')
    ax4.set_title('Algorithm Efficiency')
    ax4.set_xticks(x)
    ax4.set_xticklabels(config_names, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = os.path.join(output_dir, 'performance_analysis.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def create_quality_metrics_comparison(comparison_results, output_dir):
    """Create quality metrics comparison."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Blue noise weight analysis
    all_weights = []
    all_qualities = []
    all_cap_uniformities = []
    config_labels = []
    
    for config_result in comparison_results:
        config_name = config_result['config']['name']
        for cap_result in config_result['capacity_constrained']:
            all_weights.append(cap_result['blue_noise_weight'])
            all_qualities.append(cap_result['analysis']['blue_noise_quality'])
            all_cap_uniformities.append(1/cap_result['analysis']['capacity_std'] if cap_result['analysis']['capacity_std'] > 0 else 0)
            config_labels.append(config_name)
    
    # Plot 1: Blue noise quality vs weight
    colors = plt.cm.tab10(np.linspace(0, 1, len(comparison_results)))
    for i, config_result in enumerate(comparison_results):
        config_name = config_result['config']['name']
        weights = [r['blue_noise_weight'] for r in config_result['capacity_constrained']]
        qualities = [r['analysis']['blue_noise_quality'] for r in config_result['capacity_constrained']]
        
        ax1.plot(weights, qualities, 'o-', color=colors[i], label=config_name, linewidth=2, markersize=8)
    
    ax1.set_xlabel('Blue Noise Weight')
    ax1.set_ylabel('Blue Noise Quality')
    ax1.set_title('Blue Noise Quality vs Weight')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Capacity uniformity vs weight
    for i, config_result in enumerate(comparison_results):
        config_name = config_result['config']['name']
        weights = [r['blue_noise_weight'] for r in config_result['capacity_constrained']]
        uniformities = [1/r['analysis']['capacity_std'] if r['analysis']['capacity_std'] > 0 else 0 
                       for r in config_result['capacity_constrained']]
        
        ax2.plot(weights, uniformities, 's-', color=colors[i], label=config_name, linewidth=2, markersize=8)
    
    ax2.set_xlabel('Blue Noise Weight')
    ax2.set_ylabel('Capacity Uniformity (1/std)')
    ax2.set_title('Capacity Uniformity vs Weight')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Quality vs uniformity scatter
    ax3.scatter(all_cap_uniformities, all_qualities, c=all_weights, s=100, 
               alpha=0.7, edgecolors='black', linewidth=1, cmap='viridis')
    ax3.set_xlabel('Capacity Uniformity (1/std)')
    ax3.set_ylabel('Blue Noise Quality')
    ax3.set_title('Quality vs Uniformity (Color = Weight)')
    cbar = plt.colorbar(ax3.collections[0], ax=ax3)
    cbar.set_label('Blue Noise Weight')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Overall quality comparison
    config_names = [r['config']['name'] for r in comparison_results]
    
    # Get best overall quality for each configuration
    best_qualities = []
    for config_result in comparison_results:
        best_quality = max(r['analysis']['overall_quality'] for r in config_result['capacity_constrained'])
        best_qualities.append(best_quality)
    
    ax4.bar(range(len(config_names)), best_qualities, color='orange', alpha=0.7)
    ax4.set_xlabel('Density Configuration')
    ax4.set_ylabel('Best Overall Quality Score')
    ax4.set_title('Best Overall Quality by Configuration')
    ax4.set_xticks(range(len(config_names)))
    ax4.set_xticklabels(config_names, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = os.path.join(output_dir, 'quality_metrics_comparison.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def print_comprehensive_summary(comparison_results, output_dir):
    """Print comprehensive summary of all comparisons."""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE ALGORITHM COMPARISON SUMMARY")
    print("=" * 70)
    
    for i, config_result in enumerate(comparison_results, 1):
        config = config_result['config']
        std_result = config_result['standard']
        
        print(f"\n{i}. {config['name']} ({config['n_generators']} generators)")
        print("-" * 50)
        
        # Standard Lloyd results
        print("Standard Lloyd Algorithm:")
        print(f"  Final Energy: {std_result['analysis']['final_energy']:.6f}")
        print(f"  Converged: {'Yes' if std_result['analysis']['converged'] else 'No'}")
        print(f"  Runtime: {std_result['runtime']:.2f}s")
        
        # Capacity-Constrained results
        print("\nCapacity-Constrained Algorithm:")
        best_cap_result = max(config_result['capacity_constrained'], 
                             key=lambda x: x['analysis']['blue_noise_quality'])
        
        for cap_result in config_result['capacity_constrained']:
            weight = cap_result['blue_noise_weight']
            analysis = cap_result['analysis']
            is_best = cap_result == best_cap_result
            best_marker = " ⭐ BEST" if is_best else ""
            
            print(f"  Weight {weight}:{best_marker}")
            print(f"    CVT Energy: {analysis['cvt_energy']:.6f}")
            print(f"    Combined Energy: {analysis['combined_energy']:.6f}")
            print(f"    Blue Noise Quality: {analysis['blue_noise_quality']:.4f}")
            print(f"    Capacity Std Dev: {analysis['capacity_std']:.6f}")
            print(f"    Overall Quality: {analysis['overall_quality']:.4f}")
            print(f"    Runtime: {cap_result['runtime']:.2f}s")
        
        # Comparison insights
        print(f"\nKey Insights:")
        energy_ratio = best_cap_result['analysis']['cvt_energy'] / std_result['analysis']['final_energy']
        runtime_ratio = best_cap_result['runtime'] / std_result['runtime']
        print(f"  Energy Ratio (Cap/Std): {energy_ratio:.2f}x")
        print(f"  Runtime Ratio (Cap/Std): {runtime_ratio:.2f}x")
        print(f"  Blue Noise Achievement: {best_cap_result['analysis']['blue_noise_quality']:.4f}")
    
    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL FINDINGS")
    print("=" * 70)
    
    avg_blue_noise = np.mean([max(r['capacity_constrained'], key=lambda x: x['analysis']['blue_noise_quality'])['analysis']['blue_noise_quality'] 
                             for r in comparison_results])
    
    print(f"Average Best Blue Noise Quality: {avg_blue_noise:.4f}")
    print(f"Total Configurations Tested: {len(comparison_results)}")
    print(f"Blue Noise Weights Tested: {[0.3, 0.5, 0.7]}")
    
    print(f"\nGenerated Comparison Files in {output_dir}:")
    print("Individual Comparisons:")
    for config_result in comparison_results:
        safe_name = config_result['config']['name'].lower().replace(' ', '_').replace('-', '_')
        print(f"  - comparison_{safe_name}.png")
    
    print("\nOverall Analysis:")
    print("  - comprehensive_comparison.png")
    print("  - performance_analysis.png") 
    print("  - quality_metrics_comparison.png")

if __name__ == "__main__":
    generate_comparison_results()
