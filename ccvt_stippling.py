#!/usr/bin/env python3
"""
Main entry point for CCVT stippling algorithms
This is the primary script for running the point distribution algorithms.
"""

import numpy as np
from PIL import Image, ImageDraw
import os
import time
import argparse

def create_ccvt_stippling(image_path, output_name=None, num_sites=600, num_points=8000, max_iterations=15):
    """
    Create CCVT stippling for any image using the ultra-targeted approach
    
    Args:
        image_path: Path to input image
        output_name: Name for output files (default: derived from input)
        num_sites: Number of stippling dots
        num_points: Number of discrete points for generation
        max_iterations: CCVT optimization iterations
    
    Returns:
        tuple: (sites, points) arrays
    """
    
    # Import the ultra-targeted flower detection
    from ultra_targeted_flower import create_ultra_targeted_flower_mask
    
    if output_name is None:
        output_name = os.path.splitext(os.path.basename(image_path))[0]
    
    print(f"\\n{'='*60}")
    print(f"CREATING CCVT STIPPLING FOR {output_name.upper()}")
    print(f"{'='*60}")
    print(f"Parameters: {num_sites} sites, {num_points} points, {max_iterations} iterations")
    
    start_time = time.time()
    
    # Create density function
    print("Creating adaptive density function...")
    density = create_ultra_targeted_flower_mask(image_path, debug=False)
    
    # Generate points using rejection sampling
    height, width = density.shape
    points = []
    attempts = 0
    max_attempts = num_points * 25
    
    print(f"Generating {num_points} points...")
    
    while len(points) < num_points and attempts < max_attempts:
        x = np.random.rand()
        y = np.random.rand()
        
        img_x = int(x * (width - 1))
        img_y = int(y * (height - 1))
        
        density_val = density[img_y, img_x]
        
        if np.random.rand() < density_val:
            points.append([x, y])
        
        attempts += 1
        
        if attempts % 10000 == 0 and len(points) > 0:
            rate = len(points) / attempts
            print(f"  Progress: {len(points)}/{num_points} points, rate: {rate:.4f}")
    
    points = np.array(points)
    acceptance_rate = len(points) / attempts if attempts > 0 else 0
    
    print(f"Point generation: {len(points)} points, rate: {acceptance_rate:.4f}")
    
    if len(points) == 0:
        print("ERROR: No points generated!")
        return None, None
    
    # Initialize sites with density bias
    sites = []
    site_attempts = 0
    max_site_attempts = num_sites * 50
    
    while len(sites) < num_sites and site_attempts < max_site_attempts:
        x = np.random.rand()
        y = np.random.rand()
        
        img_x = int(x * (width - 1))
        img_y = int(y * (height - 1))
        
        density_val = density[img_y, img_x]
        
        if np.random.rand() < density_val * 1.5:
            sites.append([x, y])
        
        site_attempts += 1
    
    # Fill remaining sites randomly
    while len(sites) < num_sites:
        sites.append([np.random.rand(), np.random.rand()])
    
    sites = np.array(sites)
    print(f"Initialized {len(sites)} sites")
    
    # Run CCVT optimization
    print("Running CCVT optimization...")
    
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
    
    total_time = time.time() - start_time
    print(f"Completed in {total_time:.2f} seconds")
    
    # Save results
    os.makedirs('output', exist_ok=True)
    
    # Create visualization
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    width, height = img.size
    
    # Create stippling visualization
    stipple_img = img.copy()
    draw = ImageDraw.Draw(stipple_img)
    
    # Draw sites as black dots
    for site in sites:
        x = int(site[0] * width)
        y = int(site[1] * height)
        draw.ellipse([x-2, y-2, x+2, y+2], fill='black')
    
    stipple_img.save(f'output/{output_name}_stippling.png')
    print(f"Stippling saved: output/{output_name}_stippling.png")
    
    # Save numerical results
    np.savetxt(f'output/{output_name}_sites.txt', sites, fmt='%.6f')
    np.savetxt(f'output/{output_name}_points.txt', points, fmt='%.6f')
    print(f"Coordinates saved: output/{output_name}_sites.txt, output/{output_name}_points.txt")
    
    return sites, points

def main():
    """Main command line interface"""
    
    parser = argparse.ArgumentParser(description='Create CCVT stippling for images')
    parser.add_argument('image', help='Input image path')
    parser.add_argument('-o', '--output', help='Output name (default: derived from input)')
    parser.add_argument('-s', '--sites', type=int, default=600, help='Number of stippling dots (default: 600)')
    parser.add_argument('-p', '--points', type=int, default=8000, help='Number of discrete points (default: 8000)')
    parser.add_argument('-i', '--iterations', type=int, default=15, help='CCVT iterations (default: 15)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"ERROR: Image file '{args.image}' not found")
        return
    
    print("CCVT STIPPLING ALGORITHM")
    print("=" * 50)
    print(f"Input: {args.image}")
    print(f"Output: {args.output or 'auto-generated'}")
    print(f"Sites: {args.sites}")
    print(f"Points: {args.points}")
    print(f"Iterations: {args.iterations}")
    
    sites, points = create_ccvt_stippling(
        args.image, 
        args.output, 
        args.sites, 
        args.points, 
        args.iterations
    )
    
    if sites is not None:
        print("\\n✅ SUCCESS! Stippling completed.")
        print("Check the 'output/' directory for results.")
    else:
        print("\\n❌ FAILED: Could not generate stippling.")

if __name__ == "__main__":
    main()
