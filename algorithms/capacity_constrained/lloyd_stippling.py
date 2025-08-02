#!/usr/bin/env python3
"""
Standard Lloyd Algorithm Wrapper
Provides the same output format as paper_accurate_stippling
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
import os
import sys
import time

# Import the existing Lloyd algorithm
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'standard_lloyd'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))

from standard_lloyd import StandardLloydAlgorithm
from image_density import ImageDensityFunction

def load_or_create_density_lloyd(image_path, output_dir="output"):
    """Create Lloyd-style density based on image content"""
    from pathlib import Path
    
    # Create unique density file name based on input image
    image_name = Path(image_path).stem
    density_file = os.path.join(output_dir, f'{image_name}_lloyd_density.npy')
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(density_file):
        print(f"Creating Lloyd density for {image_name}...")
        
        # Load and process image using standard approach
        img = Image.open(image_path).convert('L')
        img_array = np.array(img, dtype=np.float64)
        
        # Apply gentle blur and normalization (less aggressive than paper method)
        img_array = gaussian_filter(img_array, sigma=0.8)
        img_array = img_array / 255.0
        img_array = 1.0 - img_array  # Invert for stippling
        
        # Use moderate thresholding (Lloyd doesn't need aggressive thresholding)
        threshold = 0.1  # Gentle threshold
        img_array = np.maximum(img_array - threshold, 0.0)
        
        # Renormalize after thresholding
        if img_array.max() > 0:
            img_array = img_array / img_array.max()
        
        # Apply moderate contrast enhancement
        img_array = np.power(img_array, 0.7)  # Less aggressive than paper
        
        # Add small minimum density to allow background points
        img_array = img_array * 0.95 + 0.05
        
        print(f"After Lloyd processing: nonzero={np.count_nonzero(img_array)}/{img_array.size} ({100*np.count_nonzero(img_array)/img_array.size:.1f}%)")
        
        np.save(density_file, img_array)
        print(f"Saved Lloyd density to {density_file}")
    else:
        print(f"Loading existing Lloyd density from {density_file}...")
        img_array = np.load(density_file)
    
    print(f"Loaded density: {img_array.shape}, range: {img_array.min():.3f} to {img_array.max():.3f}")
    return img_array

def run_lloyd_stippling(image_path, num_points, output_dir="output"):
    """
    Run standard Lloyd algorithm for stippling
    
    Args:
        image_path: Path to input image
        num_points: Number of stippling points to generate
        output_dir: Output directory for results
        
    Returns:
        points: Generated stippling points
    """
    
    print(f"\\n{'='*60}")
    print(f"STANDARD LLOYD STIPPLING")
    print(f"{'='*60}")
    print(f"Image: {image_path}")
    print(f"Points: {num_points}")
    print(f"Output: {output_dir}")
    
    start_time = time.time()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Create density function
    print(f"\\n🎯 Creating density function for Lloyd algorithm...")
    density_array = load_or_create_density_lloyd(image_path, output_dir)
    
    # Create density function using existing utility
    img_density = ImageDensityFunction(image_path, invert=True, blur_sigma=0.8)
    density_func = img_density.get_density_function()
    
    # Step 2: Initialize Lloyd algorithm
    domain = (0, 1, 0, 1)
    algorithm = StandardLloydAlgorithm(domain)
    
    # Step 3: Generate initial points randomly
    np.random.seed(42)  # For reproducible results
    initial_points = algorithm.generate_random_points(num_points)
    print(f"Generated {len(initial_points)} initial points")
    
    # Step 4: Run Lloyd iterations
    print(f"\\n🔄 Running Lloyd iterations...")
    final_points, energy_history = algorithm.run(
        initial_points,
        n_iterations=50,  # More iterations for Lloyd
        density_func=density_func,
        verbose=True
    )
    
    if len(final_points) == 0:
        print("ERROR: No points generated!")
        return None
    
    print(f"✅ Generated {len(final_points)} points")
    print(f"📊 Final energy: {energy_history[-1]:.6f}")
    
    # Step 5: Create output files (same format as paper_accurate_stippling)
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    algo_name = "lloyd"  # Algorithm identifier
    
    print("\\n📁 Creating output files...")
    
    # 1. Clean stippling (black dots on white background)
    clean_path = os.path.join(output_dir, f"{image_name}_{algo_name}_clean_stippling.png")
    create_clean_stippling_lloyd(final_points, image_path, clean_path)
    
    # 2. Points visualization (yellow points over original image)
    points_viz_path = os.path.join(output_dir, f"{image_name}_{algo_name}_points_visualization.png")
    create_points_visualization_lloyd(final_points, image_path, points_viz_path)
    
    # 3. Voronoi centers visualization (if we have enough points for meaningful Voronoi)
    if len(final_points) >= 10:
        centers_path = os.path.join(output_dir, f"{image_name}_{algo_name}_centers_visualization.png")
        create_centers_visualization_lloyd(final_points, image_path, centers_path)
    
    # 4. Save point coordinates (centroids)
    points_path = os.path.join(output_dir, f"{image_name}_{algo_name}_points.txt")
    centroids_path = os.path.join(output_dir, f"{image_name}_{algo_name}_centroids.txt")
    np.savetxt(points_path, final_points, fmt='%.6f', header='x y (normalized coordinates)')
    np.savetxt(centroids_path, final_points, fmt='%.6f', header='x y (centroids in normalized coordinates)')
    
    # 5. Save additional metadata
    metadata_path = os.path.join(output_dir, f"{image_name}_{algo_name}_metadata.txt")
    with open(metadata_path, 'w') as f:
        f.write(f"Image: {os.path.basename(image_path)}\\n")
        f.write(f"Algorithm: lloyd\\n")
        f.write(f"Points generated: {len(final_points)}\\n")
        f.write(f"Lloyd iterations: 50\\n")
        f.write(f"Final energy: {energy_history[-1]:.6f}\\n")
        f.write(f"Generation time: {time.time() - start_time:.2f} seconds\\n")
        f.write(f"Coordinate system: normalized [0,1] x [0,1]\\n")
        f.write(f"Density file: {image_name}_lloyd_density.npy\\n")
    
    total_time = time.time() - start_time
    print(f"\\n✅ Complete! Generated in {total_time:.2f} seconds")
    print(f"📁 Files created:")
    print(f"   • {clean_path} (Clean stippling)")
    print(f"   • {points_viz_path} (Points visualization)")
    if len(final_points) >= 10:
        print(f"   • {centers_path} (Centers visualization)")
    print(f"   • {points_path} (Points coordinates)")
    print(f"   • {centroids_path} (Centroids coordinates)")
    print(f"   • {metadata_path} (Metadata)")
    
    return final_points

def create_clean_stippling_lloyd(points, image_path, output_path):
    """Create a clean stippling without background image overlay"""
    
    # Load original image to get dimensions
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    width, height = img.size
    
    # Create white background image
    stipple_img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(stipple_img)
    
    # Draw points as black dots
    for point in points:
        x = int(point[0] * width)
        y = int(point[1] * height)
        draw.ellipse([x-2, y-2, x+2, y+2], fill='black')
    
    stipple_img.save(output_path)
    print(f"Clean stippling saved: {output_path}")

def create_points_visualization_lloyd(points, image_path, output_path):
    """Create points visualization with yellow dots over image"""
    
    # Load original image
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_array = np.array(img)
    
    # Create matplotlib figure  
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Show original image with transparency for density reference
    ax.imshow(img_array, alpha=0.4)
    
    # Overlay points as yellow dots
    if len(points) > 0:
        ax.scatter(points[:, 0] * img_array.shape[1], points[:, 1] * img_array.shape[0], 
                  s=6, c='yellow', alpha=0.9, edgecolors='orange', linewidth=0.3)
    
    # Set limits and remove axes
    ax.set_xlim(0, img_array.shape[1])
    ax.set_ylim(img_array.shape[0], 0)  # Flip Y axis to match image
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Points visualization saved: {output_path}")

def create_centers_visualization_lloyd(points, image_path, output_path):
    """Create Voronoi centers visualization for Lloyd algorithm"""
    
    # Load original image
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_array = np.array(img)
    
    # Create matplotlib figure  
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Show original image with light transparency
    ax.imshow(img_array, alpha=0.3)
    
    # Generate Voronoi diagram
    from scipy.spatial import Voronoi, voronoi_plot_2d
    
    # Scale points to image dimensions for Voronoi calculation
    scaled_points = points * [img_array.shape[1], img_array.shape[0]]
    
    # Add boundary points to avoid issues at edges
    boundary_points = [
        [0, 0], [img_array.shape[1], 0], 
        [0, img_array.shape[0]], [img_array.shape[1], img_array.shape[0]],
        [img_array.shape[1]/2, 0], [img_array.shape[1]/2, img_array.shape[0]],
        [0, img_array.shape[0]/2], [img_array.shape[1], img_array.shape[0]/2]
    ]
    all_points = np.vstack([scaled_points, boundary_points])
    
    # Create Voronoi diagram
    vor = Voronoi(all_points)
    
    # Plot Voronoi cells (only the edges)
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='green', 
                    line_width=0.5, line_alpha=0.6, point_size=0)
    
    # Plot the original points as blue dots (to distinguish from CCVT)
    ax.scatter(points[:, 0] * img_array.shape[1], points[:, 1] * img_array.shape[0], 
              s=8, c='blue', alpha=0.8, edgecolors='darkblue', linewidth=0.5, zorder=5)
    
    # Set limits and remove axes
    ax.set_xlim(0, img_array.shape[1])
    ax.set_ylim(img_array.shape[0], 0)  # Flip Y axis to match image
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Centers visualization saved: {output_path}")

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python lloyd_stippling.py <image_path> <num_points>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    num_points = int(sys.argv[2])
    
    if not os.path.exists(image_path):
        print(f"ERROR: Image file '{image_path}' not found")
        sys.exit(1)
    
    points = run_lloyd_stippling(image_path, num_points)
    
    if points is not None:
        print("\\n🎉 SUCCESS! Lloyd stippling completed.")
    else:
        print("\\n❌ FAILED: Could not generate Lloyd stippling.")
