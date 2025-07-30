"""
Image-based Density Functions for Point Distribution Algorithms

This module provides functionality to use images as density functions for 
point distribution algorithms, enabling stippling and artistic effects
similar to those shown in the paper.

Supports:
- Grayscale image loading and processing
- Image-to-density function conversion
- Automatic image scaling and normalization
- Multiple sampling strategies
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from scipy.ndimage import gaussian_filter
from typing import Tuple, Callable, Optional


class ImageDensityFunction:
    """
    Convert images to density functions for point distribution algorithms.
    
    This class loads an image and creates a density function that can be used
    with the point distribution algorithms to create stippling effects.
    """
    
    def __init__(self, image_path: str, domain: Tuple[float, float, float, float] = (0, 1, 0, 1),
                 invert: bool = True, blur_sigma: float = 0.0):
        """
        Initialize image-based density function.
        
        Args:
            image_path: Path to the image file
            domain: (xmin, xmax, ymin, ymax) domain bounds
            invert: If True, dark areas become high density (typical for stippling)
            blur_sigma: Gaussian blur sigma for smoothing (0 = no blur)
        """
        self.image_path = image_path
        self.domain = domain
        self.invert = invert
        self.blur_sigma = blur_sigma
        
        # Load and process image
        self.image_data = self._load_and_process_image()
        self.height, self.width = self.image_data.shape
        
    def _load_and_process_image(self) -> np.ndarray:
        """Load image and convert to normalized grayscale density."""
        try:
            # Load image and convert to grayscale
            img = Image.open(self.image_path).convert('L')
            img_array = np.array(img, dtype=np.float64)
            
            # Apply Gaussian blur if requested
            if self.blur_sigma > 0:
                img_array = gaussian_filter(img_array, sigma=self.blur_sigma)
            
            # Normalize to [0, 1]
            img_array = img_array / 255.0
            
            # Invert if requested (dark areas = high density)
            if self.invert:
                img_array = 1.0 - img_array
            
            return img_array
            
        except Exception as e:
            raise ValueError(f"Error loading image {self.image_path}: {e}")
    
    def get_density_function(self) -> Callable[[float, float], float]:
        """
        Get density function that can be used with point distribution algorithms.
        
        Returns:
            Function that takes (x, y) coordinates and returns density value
        """
        def density_func(x: float, y: float) -> float:
            # Map domain coordinates to image coordinates
            xmin, xmax, ymin, ymax = self.domain
            
            # Clamp coordinates to domain
            x = np.clip(x, xmin, xmax)
            y = np.clip(y, ymin, ymax)
            
            # Map to image coordinates
            img_x = int((x - xmin) / (xmax - xmin) * (self.width - 1))
            img_y = int((y - ymin) / (ymax - ymin) * (self.height - 1))
            
            # Clamp to image bounds
            img_x = np.clip(img_x, 0, self.width - 1)
            img_y = np.clip(img_y, 0, self.height - 1)
            
            return float(self.image_data[img_y, img_x])
        
        return density_func
    
    def visualize_density(self, figsize: Tuple[int, int] = (10, 5)) -> None:
        """Visualize the original image and resulting density function."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Original image
        original_img = np.array(Image.open(self.image_path).convert('L'))
        ax1.imshow(original_img, cmap='gray', origin='upper')
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Density function
        ax2.imshow(self.image_data, cmap='viridis', origin='lower', 
                  extent=[self.domain[0], self.domain[1], self.domain[2], self.domain[3]])
        ax2.set_title('Density Function')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        
        plt.tight_layout()
        plt.show()
    
    def save_density_preview(self, output_path: str) -> None:
        """Save a preview of the density function."""
        plt.figure(figsize=(8, 8))
        plt.imshow(self.image_data, cmap='viridis', origin='lower',
                  extent=[self.domain[0], self.domain[1], self.domain[2], self.domain[3]])
        plt.title(f'Density Function from {os.path.basename(self.image_path)}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar(label='Density')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


def create_image_based_algorithms_runner():
    """
    Create a specialized runner for image-based point distribution.
    
    This creates a simple interface for using images with the algorithms.
    """
    
    def run_image_stippling(image_path: str, algorithm_choice: str = 'optimized',
                           n_points: int = 5000, n_iterations: int = 50,
                           invert: bool = True, blur_sigma: float = 1.0,
                           output_dir: str = 'output') -> Tuple[np.ndarray, str]:
        """
        Run point distribution algorithm on an image for stippling effect.
        
        Args:
            image_path: Path to input image
            algorithm_choice: 'lloyd', 'capacity', or 'optimized'
            n_points: Number of points to generate
            n_iterations: Algorithm iterations
            invert: Invert image (dark areas = high density)
            blur_sigma: Gaussian blur for smoothing
            output_dir: Output directory
            
        Returns:
            Tuple of (final_points, output_filename)
        """
        from algorithms.standard_lloyd import StandardLloydAlgorithm
        from algorithms.capacity_constrained import (
            CapacityConstrainedDistributionAlgorithm,
            OptimizedCapacityConstrainedVoronoiAlgorithm
        )
        
        # Create image density function
        img_density = ImageDensityFunction(image_path, invert=invert, blur_sigma=blur_sigma)
        density_func = img_density.get_density_function()
        
        # Set up algorithm
        domain = (0, 1, 0, 1)
        if algorithm_choice == 'lloyd':
            algorithm = StandardLloydAlgorithm(domain)
        elif algorithm_choice == 'capacity':
            algorithm = CapacityConstrainedDistributionAlgorithm(domain)
        elif algorithm_choice == 'optimized':
            algorithm = OptimizedCapacityConstrainedVoronoiAlgorithm(domain)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_choice}")
        
        # Generate initial points
        np.random.seed(42)
        initial_points = algorithm.generate_random_points(n_points)
        
        print(f"Running {algorithm_choice} algorithm on image: {os.path.basename(image_path)}")
        print(f"Points: {n_points}, Iterations: {n_iterations}")
        print(f"Image processing: invert={invert}, blur_sigma={blur_sigma}")
        
        # Run algorithm
        if algorithm_choice == 'lloyd':
            final_points, _ = algorithm.run(initial_points, n_iterations=n_iterations, 
                                          density_func=density_func, verbose=True)
        elif algorithm_choice == 'capacity':
            final_points, _, _, _ = algorithm.run(initial_points, density_func,
                                                n_iterations=n_iterations, verbose=True)
        else:  # optimized
            final_points, _, _ = algorithm.run(initial_points, density_func,
                                             n_iterations=n_iterations, verbose=True)
        
        # Create output filename
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        output_name = f"{algorithm_choice}_stippling_{image_name}_{n_points}pts"
        
        return final_points, output_name
    
    return run_image_stippling


def create_stippling_visualization(points: np.ndarray, image_path: str, 
                                 output_path: str, title: str = "Stippling Result",
                                 point_size: float = 1.0, domain: Tuple[float, float, float, float] = (0, 1, 0, 1)):
    """
    Create a stippling visualization overlaying points on the original image.
    
    Args:
        points: Generated point positions
        image_path: Path to original image
        output_path: Output file path
        title: Plot title
        point_size: Size of the points
        domain: Domain bounds
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Load original image
    original_img = np.array(Image.open(image_path).convert('L'))
    
    # Show original image
    ax1.imshow(original_img, cmap='gray', origin='upper')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Show stippling result
    ax2.set_facecolor('white')
    ax2.scatter(points[:, 0], points[:, 1], s=point_size, c='black', alpha=0.8)
    ax2.set_xlim(domain[0], domain[1])
    ax2.set_ylim(domain[2], domain[3])
    ax2.set_aspect('equal')
    ax2.set_title(f'{title} ({len(points)} points)')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Stippling result saved to: {output_path}")


# Example usage and convenience functions
def get_sample_images_info():
    """Information about using sample images."""
    return {
        'description': 'To use your own images for stippling:',
        'requirements': [
            'Place image files in the project directory',
            'Supported formats: PNG, JPG, JPEG, BMP, TIFF',
            'Images will be converted to grayscale automatically',
            'Higher resolution images work better for detailed stippling'
        ],
        'tips': [
            'Use high contrast images for best results',
            'Dark areas will have more points (high density)',
            'Adjust blur_sigma (0-3) to smooth details',
            'Use more points (5000-20000) for detailed images',
            'Try different algorithms for different artistic effects'
        ]
    }


def create_sample_image_if_needed():
    """Create a simple sample image if no images are available."""
    # Ensure sample_output directory exists
    sample_dir = 'sample_output'
    os.makedirs(sample_dir, exist_ok=True)
    
    # Create a simple geometric sample image
    img_size = 256
    x, y = np.meshgrid(np.linspace(-2, 2, img_size), np.linspace(-2, 2, img_size))
    
    # Create interesting pattern
    pattern = np.sin(x*3) * np.cos(y*3) + np.exp(-(x**2 + y**2)/2)
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
    pattern = (pattern * 255).astype(np.uint8)
    
    # Save sample image in sample_output directory
    sample_path = os.path.join(sample_dir, 'sample_pattern.png')
    Image.fromarray(pattern).save(sample_path)
    
    return sample_path
