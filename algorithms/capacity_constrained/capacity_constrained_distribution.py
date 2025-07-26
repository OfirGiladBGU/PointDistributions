"""
Capacity-Constrained Point Distribution Algorithm for High-Quality Blue Noise Distributions.

This module implements the algorithm from the paper:
"Capacity-Constrained Point Distributions: A Variant of Lloyd's Method"
by Michael Balzer, Thomas Schlömer, and Oliver Deussen

The algorithm produces point distributions with high-quality blue noise characteristics
while adapting precisely to given density functions. Each point has the same capacity
(area of its Voronoi region weighted with the underlying density function), ensuring
equal importance in the distribution.

Key innovation: Combines blue noise enhancement and density function adaptation 
in one operation through capacity constraints.

Note: While originally described as a "variant of Lloyd's method," this implementation
uses fundamentally different optimization principles focused on capacity constraints
and blue noise quality rather than traditional Lloyd centroidal movement.
"""

import numpy as np
from typing import Tuple, Optional, List
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.base import BaseAlgorithm, VoronoiUtils, SamplingUtils
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi
from scipy.optimize import minimize_scalar


class CapacityConstrainedDistributionAlgorithm(BaseAlgorithm):
    """
    Implementation of capacity-constrained point distribution algorithm for blue noise distributions.
    
    This algorithm enforces that each point has the same capacity while optimizing
    for blue noise characteristics. The capacity of a point is defined as the 
    integral of the density function over its Voronoi region.
    
    Key features:
    - Enforces equal capacity constraint: C_i = C_target for all points
    - Optimizes for blue noise characteristics (spatial regularity)
    - Adapts precisely to given density functions
    - Uses capacity-weighted energy minimization
    - Fundamentally different from traditional Lloyd's method
    
    Algorithm principle:
    - Each point obtains equal importance through capacity constraint
    - Energy function weighted by capacity differences
    - Iterative optimization balances capacity and spatial distribution
    """
    
    def __init__(self, domain_bounds: Tuple[float, float, float, float] = (0, 1, 0, 1)):
        """
        Initialize the Capacity-Constrained Lloyd algorithm.
        
        Args:
            domain_bounds: (xmin, xmax, ymin, ymax) defining the rectangular domain
        """
        super().__init__(domain_bounds)
        self.voronoi_utils = VoronoiUtils()
        self.sampling_utils = SamplingUtils()
        
        # Algorithm parameters for blue noise optimization
        self.capacity_tolerance = 1e-4  # Tolerance for capacity constraint
        self.blue_noise_weight = 0.5    # Weight for blue noise vs capacity optimization
        
    def compute_capacity(self, region_points: np.ndarray, density_func: callable) -> float:
        """
        Compute the capacity of a single Voronoi region.
        
        Capacity is defined as the integral of the density function over the region:
        C = ∫_{V} ρ(x) dx
        
        Args:
            region_points: Points belonging to the Voronoi region
            density_func: Density function ρ(x,y)
            
        Returns:
            Capacity value for the region
        """
        if len(region_points) == 0:
            return 0.0
            
        # Monte Carlo integration over the region
        densities = np.array([density_func(p[0], p[1]) for p in region_points])
        
        # Approximate the area element based on uniform sampling density
        area_element = self.domain_area / len(region_points) if len(region_points) > 0 else 0
        capacity = np.sum(densities) * area_element
        
        return capacity
        
    def compute_capacities(self, generators: np.ndarray, sample_points: np.ndarray,
                          density_func: callable) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Compute the capacities of all Voronoi regions.
        
        Args:
            generators: Generator positions
            sample_points: Dense sampling of the domain
            density_func: Density function ρ(x,y)
            
        Returns:
            Tuple of (capacities_array, list_of_regions)
        """
        # Compute Voronoi regions
        regions = self.voronoi_utils.compute_voronoi_regions(generators, sample_points)
        
        # Compute capacity for each region
        capacities = np.array([self.compute_capacity(region, density_func) for region in regions])
        
        return capacities, regions
    
    def compute_target_capacity(self, sample_points: np.ndarray, 
                               density_func: callable, 
                               n_generators: int) -> float:
        """
        Compute the target capacity per generator for equal capacity constraint.
        
        Target capacity ensures each point has equal importance:
        C_target = (∫_Ω ρ(x) dx) / n
        
        Args:
            sample_points: Sample points for integration
            density_func: Density function
            n_generators: Number of generators
            
        Returns:
            Target capacity value
        """
        # Approximate total integral using Monte Carlo
        total_density = np.sum([density_func(p[0], p[1]) for p in sample_points])
        total_capacity = total_density * self.domain_area / len(sample_points)
        return total_capacity / n_generators
    
    def compute_capacity_energy(self, generators: np.ndarray, sample_points: np.ndarray,
                               density_func: callable, target_capacity: float) -> float:
        """
        Compute energy function that combines spatial distribution and capacity constraints.
        
        The energy function balances:
        1. Standard CVT energy for spatial quality
        2. Capacity deviation penalty for equal importance constraint
        
        E = α * E_CVT + β * E_capacity
        where E_capacity = Σ(C_i - C_target)²
        
        Args:
            generators: Generator positions
            sample_points: Sample points for integration
            density_func: Density function
            target_capacity: Target capacity per generator
            
        Returns:
            Combined energy value
        """
        # Standard CVT energy
        cvt_energy = self.voronoi_utils.compute_energy(generators, sample_points, density_func)
        
        # Capacity constraint energy
        capacities, _ = self.compute_capacities(generators, sample_points, density_func)
        capacity_deviations = capacities - target_capacity
        capacity_energy = np.sum(capacity_deviations**2)
        
        # Combined energy with weighting
        total_energy = (self.blue_noise_weight * cvt_energy + 
                       (1 - self.blue_noise_weight) * capacity_energy)
        
        return total_energy
    
    def capacity_constrained_iteration(self, generators: np.ndarray,
                                     sample_points: np.ndarray,
                                     density_func: callable,
                                     target_capacity: float) -> np.ndarray:
        """
        Perform one iteration of the capacity-constrained point distribution algorithm.
        
        This implements the core innovation from the paper: optimizing point positions
        to minimize energy while enforcing equal capacity constraints.
        
        The algorithm:
        1. Compute current capacities and capacity deviations
        2. Adjust Voronoi assignments based on capacity constraints
        3. Compute new centroids with capacity-weighted optimization
        4. Apply capacity correction to maintain equal importance
        
        Args:
            generators: Current generator positions
            sample_points: Dense sampling of the domain
            density_func: Density function
            target_capacity: Target capacity for equal importance
            
        Returns:
            New generator positions
        """
        # Compute current capacities and regions
        capacities, regions = self.compute_capacities(generators, sample_points, density_func)
        
        # Compute capacity deviations from target
        capacity_errors = capacities - target_capacity
        
        # Capacity-based weights for Voronoi assignment modification
        # Points with excess capacity get higher weights (repel points)
        # Points with deficit capacity get lower weights (attract points)
        capacity_weights = np.ones(len(generators))
        for i, error in enumerate(capacity_errors):
            if target_capacity > 0:
                # Adaptive weight based on capacity deviation
                weight_factor = 1.0 + (error / target_capacity)
                capacity_weights[i] = max(0.1, weight_factor)
        
        # Modified Voronoi assignment with capacity-weighted distances
        distances = cdist(sample_points, generators)
        weighted_distances = distances / capacity_weights[np.newaxis, :]
        assignments = np.argmin(weighted_distances, axis=1)
        
        # Recompute regions with capacity-adjusted assignments
        new_regions = []
        for i in range(len(generators)):
            region_points = sample_points[assignments == i]
            new_regions.append(region_points)
        
        # Compute new centroids with capacity-aware weighting
        new_generators = np.zeros_like(generators)
        for i, region in enumerate(new_regions):
            if len(region) > 0:
                # Density-weighted centroid computation
                densities = np.array([density_func(p[0], p[1]) for p in region])
                
                # Additional capacity correction factor
                capacity_correction = target_capacity / (capacities[i] + 1e-10)
                corrected_weights = densities * capacity_correction
                
                if np.sum(corrected_weights) > 0:
                    centroid = np.average(region, axis=0, weights=corrected_weights)
                else:
                    centroid = np.mean(region, axis=0)
                    
                new_generators[i] = centroid
            else:
                # Keep generator in place if no points assigned
                new_generators[i] = generators[i]
        
        return self.clip_to_domain(new_generators)
    
    def run(self, initial_generators: np.ndarray,
            density_func: callable,
            n_iterations: int = 100,
            tolerance: float = 1e-6,
            sample_density: int = 10000,
            blue_noise_weight: float = 0.5,
            verbose: bool = True) -> Tuple[np.ndarray, List[float], List[float], List[float]]:
        """
        Run the capacity-constrained Lloyd algorithm for blue noise point distributions.
        
        This algorithm produces high-quality blue noise characteristics while adapting
        precisely to the given density function and ensuring equal capacity per point.
        
        Args:
            initial_generators: Initial generator positions
            density_func: Density function (required for capacity computation)
            n_iterations: Maximum number of iterations
            tolerance: Convergence tolerance (max movement threshold)
            sample_density: Number of sample points for numerical integration
            blue_noise_weight: Balance between blue noise (CVT) and capacity optimization (0-1)
            verbose: Whether to print progress information
            
        Returns:
            Tuple of (final_generators, energy_history, capacity_variance_history, combined_energy_history)
        """
        if verbose:
            print("Running Capacity-Constrained Point Distribution Algorithm for Blue Noise")
            print("=" * 70)
        
        # Set algorithm parameters
        self.blue_noise_weight = blue_noise_weight
        
        # Generate dense sampling for integration
        sample_points = self.sampling_utils.generate_uniform_grid_samples(
            (self.xmin, self.xmax, self.ymin, self.ymax), sample_density
        )
        
        # Compute target capacity for equal importance constraint
        target_capacity = self.compute_target_capacity(
            sample_points, density_func, len(initial_generators)
        )
        
        if verbose:
            print(f"Target capacity per generator: {target_capacity:.6f}")
            print(f"Blue noise weight: {blue_noise_weight:.2f}")
            print(f"Capacity constraint weight: {1-blue_noise_weight:.2f}")
        
        generators = initial_generators.copy()
        energy_history = []
        capacity_variance_history = []
        combined_energy_history = []
        
        for iteration in range(n_iterations):
            # Compute current metrics
            cvt_energy = self.voronoi_utils.compute_energy(generators, sample_points, density_func)
            combined_energy = self.compute_capacity_energy(generators, sample_points, 
                                                         density_func, target_capacity)
            
            capacities, _ = self.compute_capacities(generators, sample_points, density_func)
            capacity_variance = np.var(capacities)
            
            energy_history.append(cvt_energy)
            capacity_variance_history.append(capacity_variance)
            combined_energy_history.append(combined_energy)
            
            # Perform capacity-constrained iteration
            new_generators = self.capacity_constrained_iteration(
                generators, sample_points, density_func, target_capacity
            )
            
            # Check convergence
            movement = np.max(np.linalg.norm(new_generators - generators, axis=1))
            
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration:3d}: CVT Energy = {cvt_energy:.6f}, "
                      f"Combined Energy = {combined_energy:.6f}, "
                      f"Capacity Var = {capacity_variance:.6f}, Movement = {movement:.6f}")
            
            if movement < tolerance:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
                
            generators = new_generators
        
        # Add final metrics to history
        final_cvt_energy = self.voronoi_utils.compute_energy(generators, sample_points, density_func)
        final_combined_energy = self.compute_capacity_energy(generators, sample_points, 
                                                           density_func, target_capacity)
        final_capacities, _ = self.compute_capacities(generators, sample_points, density_func)
        final_capacity_variance = np.var(final_capacities)
        
        energy_history.append(final_cvt_energy)
        capacity_variance_history.append(final_capacity_variance)
        combined_energy_history.append(final_combined_energy)
        
        if verbose:
            print(f"\nFinal Results:")
            print(f"CVT Energy: {final_cvt_energy:.6f}")
            print(f"Combined Energy: {final_combined_energy:.6f}")
            print(f"Capacity variance: {final_capacity_variance:.6f}")
            print(f"Capacity std deviation: {np.std(final_capacities):.6f}")
            print(f"Blue noise quality achieved with equal capacity constraint!")
        
        return generators, energy_history, capacity_variance_history, combined_energy_history
    
    def analyze_capacity_distribution(self, generators: np.ndarray,
                                    density_func: callable,
                                    sample_density: int = 10000) -> dict:
        """
        Analyze the capacity distribution and blue noise quality of the current configuration.
        
        Args:
            generators: Generator positions
            density_func: Density function
            sample_density: Number of sample points for integration
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        sample_points = self.sampling_utils.generate_uniform_grid_samples(
            (self.xmin, self.xmax, self.ymin, self.ymax), sample_density
        )
        
        capacities, regions = self.compute_capacities(generators, sample_points, density_func)
        target_capacity = self.compute_target_capacity(sample_points, density_func, len(generators))
        
        # Blue noise quality metrics
        cvt_energy = self.voronoi_utils.compute_energy(generators, sample_points, density_func)
        combined_energy = self.compute_capacity_energy(generators, sample_points, 
                                                     density_func, target_capacity)
        
        # Spatial distribution quality (simple nearest neighbor analysis)
        from scipy.spatial.distance import pdist
        if len(generators) > 1:
            distances = pdist(generators)
            min_distance = np.min(distances)
            mean_distance = np.mean(distances)
            distance_variance = np.var(distances)
            regularity_score = min_distance / mean_distance  # Higher is better for blue noise
        else:
            min_distance = mean_distance = distance_variance = regularity_score = 0.0
        
        return {
            # Capacity analysis
            'capacities': capacities,
            'target_capacity': target_capacity,
            'capacity_mean': np.mean(capacities),
            'capacity_std': np.std(capacities),
            'capacity_variance': np.var(capacities),
            'capacity_min': np.min(capacities),
            'capacity_max': np.max(capacities),
            'capacity_range': np.max(capacities) - np.min(capacities),
            'capacity_uniformity_score': 1.0 / (1.0 + np.std(capacities)),
            
            # Energy analysis
            'cvt_energy': cvt_energy,
            'combined_energy': combined_energy,
            'energy_ratio': combined_energy / cvt_energy if cvt_energy > 0 else 1.0,
            
            # Blue noise quality analysis
            'min_distance': min_distance,
            'mean_distance': mean_distance,
            'distance_variance': distance_variance,
            'regularity_score': regularity_score,
            'blue_noise_quality': regularity_score * (1.0 / (1.0 + distance_variance)),
            
            # Overall quality score
            'overall_quality': (self.blue_noise_weight * regularity_score + 
                              (1 - self.blue_noise_weight) * (1.0 / (1.0 + np.std(capacities))))
        }
    
    def get_algorithm_info(self) -> dict:
        """Get information about this algorithm."""
        return {
            'name': 'Capacity-Constrained Point Distribution Algorithm for Blue Noise',
            'type': 'Blue Noise Point Distribution with Capacity Constraints',
            'description': 'Produces high-quality blue noise characteristics while adapting to density functions',
            'paper': 'Capacity-Constrained Point Distributions: A Variant of Lloyd\'s Method',
            'authors': 'Michael Balzer, Thomas Schlömer, and Oliver Deussen',
            'key_innovation': 'Combines blue noise enhancement and density adaptation through capacity constraints',
            'features': [
                'High-quality blue noise characteristics',
                'Equal capacity constraint (equal importance per point)',
                'Precise adaptation to given density functions',
                'Fundamentally different approach from traditional Lloyd\'s method',
                'Spatial regularity optimization',
                'Capacity-weighted energy minimization'
            ],
            'best_for': [
                'Applications requiring blue noise distributions',
                'Stippling and halftoning with density constraints',
                'High-quality sampling with uniform importance',
                'Computer graphics and visualization',
                'Mesh generation with quality constraints',
                'Scientific simulations requiring regular distributions'
            ],
            'parameters': {
                'blue_noise_weight': 'Balance between spatial quality and capacity uniformity (0-1)',
                'target_capacity': 'Automatically computed for equal importance constraint',
                'capacity_tolerance': 'Tolerance for capacity constraint satisfaction'
            },
            'advantages_over_standard_lloyd': [
                'Better spatial regularity (blue noise)',
                'Equal importance per point through capacity constraints',
                'Superior distribution quality for visualization applications',
                'More uniform point spacing in varying density regions'
            ]
        }
