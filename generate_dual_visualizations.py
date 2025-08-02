#!/usr/bin/env python3
"""
Generate separate point visualization and stippling outputs
Based on the successful test_new_approach.py script
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
import os
import time
import argparse

def load_or_create_density(image_path):
    """Create paper-accurate density based on image content"""
    import os
    from pathlib import Path
    from PIL import Image
    from scipy.ndimage import gaussian_filter
    
    # Create unique density file name based on input image
    image_name = Path(image_path).stem
    density_file = f'{image_name}_density.npy'
    
    if not os.path.exists(density_file):
        print(f"Creating paper-accurate density for {image_name}...")
        
        # Load and process image using paper-accurate approach
        img = Image.open(image_path).convert('L')
        img_array = np.array(img, dtype=np.float64)
        
        # Apply blur and normalization (following paper_accurate_ccvt.py)
        img_array = gaussian_filter(img_array, sigma=1.0)
        img_array = img_array / 255.0
        img_array = 1.0 - img_array  # Invert for stippling
        
        # Use paper-accurate aggressive thresholding
        threshold = 0.35  # Aggressive - only accept very dense areas
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
        print(f"Loading existing density from {density_file}...")
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

def run_ccvt_optimization(points, num_sites=800, max_iterations=20):
    """Run CCVT optimization on the generated points"""
    
    if len(points) == 0:
        print("ERROR: No points to optimize!")
        return None
    
    # Initialize sites randomly
    sites = np.random.rand(num_sites, 2)
    
    print(f"Running CCVT optimization with {num_sites} sites...")
    
    for iteration in range(max_iterations):
        # Assign points to nearest sites
        distances = np.sqrt(((points[:, np.newaxis, :] - sites[np.newaxis, :, :]) ** 2).sum(axis=2))
        assignments = np.argmin(distances, axis=1)
        
        # Update sites to centroids
        new_sites = np.zeros_like(sites)
        for i in range(num_sites):
            mask = assignments == i
            if np.sum(mask) > 0:
                new_sites[i] = np.mean(points[mask], axis=0)
            else:
                new_sites[i] = sites[i]
        
        # Check convergence
        movement = np.sqrt(np.sum((sites - new_sites) ** 2, axis=1))
        max_movement = np.max(movement)
        
        sites = new_sites
        
        if iteration % 5 == 0:
            print(f"  Iteration {iteration+1}: max movement = {max_movement:.6f}")
        
        if max_movement < 1e-6:
            print(f"  Converged at iteration {iteration+1}")
            break
    
    return sites

def create_points_visualization(points, image_path, output_name):
    """Create the middle image: Generated Points visualization"""
    
    # Load original image
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_array = np.array(img)
    
    # Create matplotlib figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Show original image with transparency
    ax.imshow(img_array, alpha=0.7)
    
    # Overlay points as blue dots
    if len(points) > 0:
        ax.scatter(points[:, 0] * img_array.shape[1], points[:, 1] * img_array.shape[0], 
                  s=3, alpha=0.8, c='blue', edgecolors='darkblue', linewidth=0.2)
    
    ax.set_title(f'Generated Points Distribution\\n({len(points)} points)', fontsize=16)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'output/{output_name}_points_visualization.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Points visualization saved: output/{output_name}_points_visualization.png")

def create_stippling_visualization(sites, image_path, output_name):
    """Create the right image: Final Stippling visualization"""
    
    # Load original image
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_array = np.array(img)
    
    # Create matplotlib figure  
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Show original image with more transparency
    ax.imshow(img_array, alpha=0.3)
    
    # Overlay stippling sites as black dots
    if len(sites) > 0:
        ax.scatter(sites[:, 0] * img_array.shape[1], sites[:, 1] * img_array.shape[0], 
                  s=4, c='black', alpha=0.9)
    
    ax.set_title(f'CCVT Stippling Result\\n({len(sites)} stippling dots)', fontsize=16)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'output/{output_name}_stippling_visualization.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Stippling visualization saved: output/{output_name}_stippling_visualization.png")

def create_clean_stippling(sites, image_path, output_name):
    """Create a clean stippling without background image overlay"""
    
    # Load original image to get dimensions
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    width, height = img.size
    
    # Create white background image
    stipple_img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(stipple_img)
    
    # Draw sites as black dots
    for site in sites:
        x = int(site[0] * width)
        y = int(site[1] * height)
        draw.ellipse([x-2, y-2, x+2, y+2], fill='black')
    
    stipple_img.save(f'output/{output_name}_clean_stippling.png')
    print(f"Clean stippling saved: output/{output_name}_clean_stippling.png")

def generate_dual_visualizations(image_path, output_name=None, num_sites=800, num_points=12000, max_iterations=20):
    """
    Main function to generate both point visualization and stippling visualization
    
    Args:
        image_path: Path to input image
        output_name: Base name for output files
        num_sites: Number of stippling dots
        num_points: Number of discrete points for generation  
        max_iterations: CCVT optimization iterations
    
    Returns:
        tuple: (sites, points) arrays
    """
    
    if output_name is None:
        output_name = os.path.splitext(os.path.basename(image_path))[0]
    
    print(f"\\n{'='*60}")
    print(f"GENERATING DUAL VISUALIZATIONS FOR {output_name.upper()}")
    print(f"{'='*60}")
    print(f"Parameters: {num_sites} sites, {num_points} points, {max_iterations} iterations")
    
    start_time = time.time()
    
    # Ensure output directory exists
    os.makedirs('output', exist_ok=True)
    
    # Step 1: Create or load density function
    density = load_or_create_density(image_path)
    
    # Step 2: Generate discrete points
    points = generate_discrete_points_from_density(density, num_points)
    
    if len(points) == 0:
        print("ERROR: No points generated!")
        return None, None
    
    # Step 3: Run CCVT optimization
    sites = run_ccvt_optimization(points, num_sites, max_iterations)
    
    if sites is None:
        print("ERROR: CCVT optimization failed!")
        return None, None
    
    # Step 4: Create visualizations
    print("\\nCreating visualizations...")
    
    # Middle image: Points visualization
    create_points_visualization(points, image_path, output_name)
    
    # Right image: Stippling visualization  
    create_stippling_visualization(sites, image_path, output_name)
    
    # Bonus: Clean stippling without background
    create_clean_stippling(sites, image_path, output_name)
    
    # Save numerical data
    np.savetxt(f'output/{output_name}_sites.txt', sites, fmt='%.6f')
    np.savetxt(f'output/{output_name}_points.txt', points, fmt='%.6f')
    
    total_time = time.time() - start_time
    print(f"\\n✅ Complete! Generated in {total_time:.2f} seconds")
    print(f"📁 Files created:")
    print(f"   • output/{output_name}_points_visualization.png (Middle image)")
    print(f"   • output/{output_name}_stippling_visualization.png (Right image)")  
    print(f"   • output/{output_name}_clean_stippling.png (Clean stippling)")
    print(f"   • output/{output_name}_sites.txt (Stippling coordinates)")
    print(f"   • output/{output_name}_points.txt (Point coordinates)")
    
    return sites, points

def main():
    """Command line interface"""
    
    parser = argparse.ArgumentParser(description='Generate separate point and stippling visualizations')
    parser.add_argument('image', help='Input image path')
    parser.add_argument('-o', '--output', help='Output base name (default: derived from input)')
    parser.add_argument('-s', '--sites', type=int, default=800, help='Number of stippling dots (default: 800)')
    parser.add_argument('-p', '--points', type=int, default=12000, help='Number of discrete points (default: 12000)')
    parser.add_argument('-i', '--iterations', type=int, default=20, help='CCVT iterations (default: 20)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"ERROR: Image file '{args.image}' not found")
        return
    
    print("DUAL VISUALIZATION GENERATOR")
    print("=" * 50)
    print(f"Input: {args.image}")
    print(f"Output base: {args.output or 'auto-generated'}")
    
    sites, points = generate_dual_visualizations(
        args.image,
        args.output,
        args.sites,
        args.points,
        args.iterations
    )
    
    if sites is not None:
        print("\\n🎉 SUCCESS! Both visualizations created.")
    else:
        print("\\n❌ FAILED: Could not generate visualizations.")

if __name__ == "__main__":
    main()
