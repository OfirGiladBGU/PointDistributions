"""
Base classes and common utilities for Lloyd algorithms.
"""

import numpy as np
from typing import Tuple, Optional, List
from abc import ABC, abstractmethod


class BaseAlgorithm(ABC):
    """
    Abstract base class for Lloyd-type algorithms.
    """
    
    def __init__(self, domain_bounds: Tuple[float, float, float, float] = (0, 1, 0, 1)):
        """
        Initialize the algorithm.
        
        Args:
            domain_bounds: (xmin, xmax, ymin, ymax) defining the rectangular domain
        """
        self.xmin, self.xmax, self.ymin, self.ymax = domain_bounds
        self.domain_area = (self.xmax - self.xmin) * (self.ymax - self.ymin)
        
    def generate_random_points(self, n_points: int, seed: Optional[int] = None) -> np.ndarray:
        """Generate random points within the domain."""
        if seed is not None:
            np.random.seed(seed)
        
        x = np.random.uniform(self.xmin, self.xmax, n_points)
        y = np.random.uniform(self.ymin, self.ymax, n_points)
        return np.column_stack([x, y])
    
    def clip_to_domain(self, points: np.ndarray) -> np.ndarray:
        """Clip points to stay within the domain bounds."""
        points = points.copy()
        points[:, 0] = np.clip(points[:, 0], self.xmin, self.xmax)
        points[:, 1] = np.clip(points[:, 1], self.ymin, self.ymax)
        return points
    
    @abstractmethod
    def run(self, initial_generators: np.ndarray, **kwargs):
        """Run the algorithm. Must be implemented by subclasses."""
        pass


class VoronoiUtils:
    """
    Utility class for Voronoi-related computations.
    """
    
    @staticmethod
    def compute_voronoi_regions(generators: np.ndarray, 
                               sample_points: np.ndarray) -> List[np.ndarray]:
        """
        Compute Voronoi regions by assigning sample points to nearest generators.
        
        Args:
            generators: Generator points (seeds)
            sample_points: Dense sampling of the domain
            
        Returns:
            List of arrays, each containing sample points in a Voronoi region
        """
        from scipy.spatial.distance import cdist
        
        distances = cdist(sample_points, generators)
        assignments = np.argmin(distances, axis=1)
        
        regions = []
        for i in range(len(generators)):
            region_points = sample_points[assignments == i]
            regions.append(region_points)
        
        return regions
    
    @staticmethod
    def compute_centroids(regions: List[np.ndarray], 
                         density_func: Optional[callable] = None,
                         domain_bounds: Tuple[float, float, float, float] = (0, 1, 0, 1)) -> np.ndarray:
        """
        Compute centroids of Voronoi regions.
        
        Args:
            regions: List of point arrays for each Voronoi region
            density_func: Optional density function for weighted centroids
            domain_bounds: Domain boundaries for handling empty regions
            
        Returns:
            Array of centroid coordinates
        """
        xmin, xmax, ymin, ymax = domain_bounds
        centroids = []
        
        for region in regions:
            if len(region) == 0:
                # Handle empty regions - use domain center
                centroids.append([(xmin + xmax) / 2, (ymin + ymax) / 2])
                continue
                
            if density_func is None:
                # Standard centroid (equal weights)
                centroid = np.mean(region, axis=0)
            else:
                # Weighted centroid
                weights = np.array([density_func(p[0], p[1]) for p in region])
                if np.sum(weights) > 0:
                    centroid = np.average(region, axis=0, weights=weights)
                else:
                    centroid = np.mean(region, axis=0)
            
            centroids.append(centroid)
        
        return np.array(centroids)
    
    @staticmethod
    def compute_energy(generators: np.ndarray, 
                      sample_points: np.ndarray,
                      density_func: Optional[callable] = None) -> float:
        """
        Compute the total energy of the current configuration.
        
        Energy is the sum of squared distances from sample points to their nearest generators,
        weighted by the density function if provided.
        """
        from scipy.spatial.distance import cdist
        
        distances = cdist(sample_points, generators)
        min_distances = np.min(distances, axis=1)
        
        if density_func is None:
            weights = np.ones(len(sample_points))
        else:
            weights = np.array([density_func(p[0], p[1]) for p in sample_points])
        
        energy = np.sum(weights * min_distances**2)
        return energy


class SamplingUtils:
    """
    Utility class for domain sampling.
    """
    
    @staticmethod
    def generate_uniform_grid_samples(domain_bounds: Tuple[float, float, float, float],
                                    sample_density: int = 10000) -> np.ndarray:
        """
        Generate uniform grid sampling of the domain.
        
        Args:
            domain_bounds: (xmin, xmax, ymin, ymax)
            sample_density: Total number of sample points
            
        Returns:
            Array of sample points
        """
        xmin, xmax, ymin, ymax = domain_bounds
        n_samples_per_dim = int(np.sqrt(sample_density))
        x = np.linspace(xmin, xmax, n_samples_per_dim)
        y = np.linspace(ymin, ymax, n_samples_per_dim)
        xx, yy = np.meshgrid(x, y)
        return np.column_stack([xx.ravel(), yy.ravel()])
