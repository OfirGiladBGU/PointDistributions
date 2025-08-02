#!/usr/bin/env python3
"""
Paper-Accurate Stippling Algorithm
Based on the successful generate_dual_visualizations.py approach

This is the working algorithm that produces paper-quality results:
- Plant.png: 20,000 points (Figure 7c)
- Buildings.png: 3,000 points (Figure 8 detail)
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
import os
import time

def load_or_create_density(image_path, output_dir="output", threshold=0.35):
    """Create paper-accurate density based on image content"""
    from pathlib import Path
    
    # Create unique density file name based on input image and threshold
    image_name = Path(image_path).stem
    # Include threshold in filename to avoid conflicts between different thresholds
    threshold_str = str(threshold).replace('.', '_')
    density_file = os.path.join(output_dir, f'{image_name}_density_t{threshold_str}.npy')
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(density_file):
        print(f"Creating paper-accurate density for {image_name} (threshold={threshold})...")
        
        # Load and process image using paper-accurate approach
        img = Image.open(image_path).convert('L')
        img_array = np.array(img, dtype=np.float64)
        
        # Apply blur and normalization (following paper_accurate_ccvt.py)
        img_array = gaussian_filter(img_array, sigma=1.0)
        img_array = img_array / 255.0
        img_array = 1.0 - img_array  # Invert for stippling
        
        # Use configurable aggressive thresholding  
        # threshold parameter passed from config (default: 0.35 - aggressive)
        img_array = np.maximum(img_array - threshold, 0.0)
        
        # Renormalize after thresholding
        if img_array.max() > 0:
            img_array = img_array / img_array.max()
        
        # Apply contrast enhancement (paper-accurate)
        img_array = np.power(img_array, 0.2)  # Very aggressive contrast
        
        # CRITICAL: Don't add minimum density - let background be zero!
        print(f"After paper-accurate processing: nonzero={np.count_nonzero(img_array)}/{img_array.size} ({100*np.count_nonzero(img_array)/img_array.size:.1f}%)")
        
        np.save(density_file, img_array)
        print(f"Saved paper-accurate density to {density_file}")
    else:
        print(f"Loading existing density from {density_file} (threshold={threshold})...")
        img_array = np.load(density_file)
    
    print(f"Loaded density: {img_array.shape}, range: {img_array.min():.3f} to {img_array.max():.3f}")
    return img_array

def generate_discrete_points_from_density(density, num_points=10000):
    """Generate discrete points using the density function"""
    height, width = density.shape
    points = []
    attempts = 0
    max_attempts = num_points * 100  # More attempts since we're more selective
    
    print(f"Generating {num_points} points using paper-accurate density...")
    
    while len(points) < num_points and attempts < max_attempts:
        x = np.random.rand()
        y = np.random.rand()
        
        img_x = int(x * (width - 1))
        img_y = int(y * (height - 1))
        
        density_val = density[img_y, img_x]
        
        # Paper-accurate approach: Only accept if density > 0 (eliminates background)
        if density_val > 0 and np.random.rand() < density_val:
            points.append([x, y])
        
        attempts += 1
        
        if len(points) % 1000 == 0 and len(points) > 0:
            print(f"   Generated {len(points)}/{num_points} points (attempts: {attempts})")
    
    points = np.array(points)
    acceptance_rate = len(points) / attempts if attempts > 0 else 0
    
    print(f"✅ Generated {len(points)} points using paper-accurate approach, rate: {acceptance_rate:.4f}")
    return points

def create_clean_stippling(points, image_path, output_path):
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

def run_paper_accurate_stippling(image_path, num_points, output_dir="output", threshold=0.35):
    """
    Main function to run paper-accurate stippling algorithm
    
    Args:
        image_path: Path to input image
        num_points: Number of stippling points to generate
        output_dir: Output directory for results
        threshold: Density threshold for aggressive filtering (default: 0.35)
        
    Returns:
        points: Generated stippling points
    """
    
    print(f"\\n{'='*60}")
    print(f"PAPER-ACCURATE STIPPLING")
    print(f"{'='*60}")
    print(f"Image: {image_path}")
    print(f"Points: {num_points}")
    print(f"Output: {output_dir}")
    
    start_time = time.time()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Generate paper-accurate discrete points
    print(f"\\n🎯 Generating {num_points} discrete points from image density...")
    density = load_or_create_density(image_path, output_dir, threshold)
    points = generate_discrete_points_from_density(density, num_points)
    
    if len(points) == 0:
        print("ERROR: No points generated!")
        return None
    
    print(f"✅ Generated {len(points)} points")
    
    # Step 2: Create output files
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    algo_name = "ccvt"  # Algorithm identifier
    
    # Create all required outputs
    print("\\n📁 Creating output files...")
    
    # 1. Clean stippling (black dots on white background)
    clean_path = os.path.join(output_dir, f"{image_name}_{algo_name}_clean_stippling.png")
    create_clean_stippling(points, image_path, clean_path)
    
    # 2. Points visualization (yellow points over original image)
    points_viz_path = os.path.join(output_dir, f"{image_name}_{algo_name}_points_visualization.png")
    create_points_visualization(points, image_path, points_viz_path)
    
    # 3. Voronoi centers visualization (if we have enough points for meaningful Voronoi)
    if len(points) >= 10:
        centers_path = os.path.join(output_dir, f"{image_name}_{algo_name}_centers_visualization.png")
        create_centers_visualization(points, image_path, centers_path)
    
    # 4. Save point coordinates (centroids)
    points_path = os.path.join(output_dir, f"{image_name}_{algo_name}_points.txt")
    centroids_path = os.path.join(output_dir, f"{image_name}_{algo_name}_centroids.txt")
    np.savetxt(points_path, points, fmt='%.6f', header='x y (normalized coordinates)')
    np.savetxt(centroids_path, points, fmt='%.6f', header='x y (centroids in normalized coordinates)')
    
    # 5. Save additional metadata
    metadata_path = os.path.join(output_dir, f"{image_name}_{algo_name}_metadata.txt")
    with open(metadata_path, 'w') as f:
        f.write(f"Image: {os.path.basename(image_path)}\\n")
        f.write(f"Algorithm: ccvt\\n")
        f.write(f"Points generated: {len(points)}\\n")
        f.write(f"Generation time: {time.time() - start_time:.2f} seconds\\n")
        f.write(f"Coordinate system: normalized [0,1] x [0,1]\\n")
        f.write(f"Density file: {image_name}_density.npy\\n")
    
    total_time = time.time() - start_time
    print(f"\\n✅ Complete! Generated in {total_time:.2f} seconds")
    print(f"📁 Files created:")
    print(f"   • {clean_path} (Clean stippling)")
    print(f"   • {points_viz_path} (Points visualization)")
    if len(points) >= 10:
        print(f"   • {centers_path} (Centers visualization)")
    print(f"   • {points_path} (Points coordinates)")
    print(f"   • {centroids_path} (Centroids coordinates)")
    print(f"   • {metadata_path} (Metadata)")
    
    return points

def create_points_visualization(points, image_path, output_path):
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

def create_centers_visualization(points, image_path, output_path):
    """Create Voronoi centers visualization"""
    
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
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='cyan', 
                    line_width=0.5, line_alpha=0.6, point_size=0)
    
    # Plot the original points as red dots
    ax.scatter(points[:, 0] * img_array.shape[1], points[:, 1] * img_array.shape[0], 
              s=8, c='red', alpha=0.8, edgecolors='darkred', linewidth=0.5, zorder=5)
    
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
        print("Usage: python paper_accurate_stippling.py <image_path> <num_points>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    num_points = int(sys.argv[2])
    
    if not os.path.exists(image_path):
        print(f"ERROR: Image file '{image_path}' not found")
        sys.exit(1)
    
    points = run_paper_accurate_stippling(image_path, num_points)
    
    if points is not None:
        print("\\n🎉 SUCCESS! Stippling completed.")
    else:
        print("\\n❌ FAILED: Could not generate stippling.")
