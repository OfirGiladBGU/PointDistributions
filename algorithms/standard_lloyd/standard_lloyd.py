"""
Standard Lloyd Algorithm for Centroidal Voronoi Tessellations (CVT).

This module implements the classic Lloyd algorithm that iteratively moves
generators to the centroids of their Voronoi regions to minimize energy.
"""

import numpy as np
from typing import Tuple, Optional, List
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.base import BaseAlgorithm, VoronoiUtils, SamplingUtils


class StandardLloydAlgorithm(BaseAlgorithm):
    """
    Implementation of the standard Lloyd algorithm for Centroidal Voronoi Tessellations (CVT).
    
    The algorithm iteratively:
    1. Computes Voronoi regions for current generators
    2. Calculates centroids of each region (weighted by density if provided)
    3. Moves generators to their region centroids
    4. Repeats until convergence
    
    Energy function: E = Σᵢ ∫_{Vᵢ} ρ(x)||x - pᵢ||² dx
    where Vᵢ is the Voronoi region of generator pᵢ and ρ(x) is the density function.
    """
    
    def __init__(self, domain_bounds: Tuple[float, float, float, float] = (0, 1, 0, 1)):
        """
        Initialize the Standard Lloyd algorithm.
        
        Args:
            domain_bounds: (xmin, xmax, ymin, ymax) defining the rectangular domain
        """
        super().__init__(domain_bounds)
        self.voronoi_utils = VoronoiUtils()
        self.sampling_utils = SamplingUtils()
    
    def lloyd_iteration(self, generators: np.ndarray, 
                       sample_points: np.ndarray,
                       density_func: Optional[callable] = None) -> np.ndarray:
        """
        Perform one iteration of Lloyd's algorithm.
        
        Args:
            generators: Current generator positions
            sample_points: Dense sampling of the domain for integration
            density_func: Optional density function for weighted centroids
            
        Returns:
            New generator positions after one iteration
        """
        # Compute Voronoi regions
        regions = self.voronoi_utils.compute_voronoi_regions(generators, sample_points)
        
        # Compute centroids of regions
        new_generators = self.voronoi_utils.compute_centroids(
            regions, density_func, (self.xmin, self.xmax, self.ymin, self.ymax)
        )
        
        # Ensure generators stay within domain
        return self.clip_to_domain(new_generators)
    
    def run(self, initial_generators: np.ndarray,
            n_iterations: int = 100,
            tolerance: float = 1e-6,
            density_func: Optional[callable] = None,
            sample_density: int = 10000,
            verbose: bool = True) -> Tuple[np.ndarray, List[float]]:
        """
        Run the standard Lloyd algorithm until convergence.
        
        Args:
            initial_generators: Initial generator positions
            n_iterations: Maximum number of iterations
            tolerance: Convergence tolerance (max movement threshold)
            density_func: Optional density function for weighted CVT
            sample_density: Number of sample points for numerical integration
            verbose: Whether to print progress information
            
        Returns:
            Tuple of (final_generators, energy_history)
        """
        if verbose:
            print("Running Standard Lloyd Algorithm")
            print("=" * 40)
        
        # Generate dense sampling of the domain for integration
        sample_points = self.sampling_utils.generate_uniform_grid_samples(
            (self.xmin, self.xmax, self.ymin, self.ymax), sample_density
        )
        
        generators = initial_generators.copy()
        energy_history = []
        
        for iteration in range(n_iterations):
            # Compute current energy
            energy = self.voronoi_utils.compute_energy(generators, sample_points, density_func)
            energy_history.append(energy)
            
            # Perform Lloyd iteration
            new_generators = self.lloyd_iteration(generators, sample_points, density_func)
            
            # Check convergence
            movement = np.max(np.linalg.norm(new_generators - generators, axis=1))
            
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: Energy = {energy:.6f}, Max movement = {movement:.6f}")
            
            if movement < tolerance:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
                
            generators = new_generators
        
        # Add final energy to history
        final_energy = self.voronoi_utils.compute_energy(generators, sample_points, density_func)
        energy_history.append(final_energy)
        
        if verbose:
            print(f"Final energy: {final_energy:.6f}")
        
        return generators, energy_history
    
    def compute_energy(self, generators: np.ndarray, 
                      sample_points: Optional[np.ndarray] = None,
                      density_func: Optional[callable] = None,
                      sample_density: int = 10000) -> float:
        """
        Compute the total energy of the current generator configuration.
        
        Args:
            generators: Current generator positions
            sample_points: Optional pre-computed sample points
            density_func: Optional density function
            sample_density: Number of sample points if not provided
            
        Returns:
            Total energy value
        """
        if sample_points is None:
            sample_points = self.sampling_utils.generate_uniform_grid_samples(
                (self.xmin, self.xmax, self.ymin, self.ymax), sample_density
            )
        
        return self.voronoi_utils.compute_energy(generators, sample_points, density_func)
    
    def get_algorithm_info(self) -> dict:
        """Get information about this algorithm."""
        return {
            'name': 'Standard Lloyd Algorithm',
            'type': 'Centroidal Voronoi Tessellation (CVT)',
            'description': 'Classic Lloyd algorithm for energy minimization',
            'features': [
                'Energy minimization through centroidal Voronoi tessellations',
                'Support for arbitrary density functions',
                'Fast convergence for most cases',
                'Produces well-distributed point sets'
            ],
            'best_for': [
                'Basic point distribution problems',
                'Applications where energy minimization is primary goal',
                'Cases with uniform or smooth density functions'
            ]
        }
