#!/usr/bin/env python3
"""
Quick demonstration of image stippling to show where points appear.
This script runs a fast stippling example and shows the results.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from utils.image_density import ImageDensityFunction, create_stippling_visualization
from algorithms.standard_lloyd import StandardLloydAlgorithm

def demo_stippling():
    """Run a quick stippling demo to show where points appear."""
    
    print("🎨 QUICK STIPPLING DEMONSTRATION")
    print("="*50)
    
    # Check if we have a test image
    test_images = [
        'sample_output/flower_test.png',
        'sample_output/sample_pattern.png',
        'flower_test.png', 
        'sample_pattern.png'
    ]
    image_path = None
    
    for img in test_images:
        if os.path.exists(img):
            image_path = img
            break
    
    if not image_path:
        print("❌ No test image found. Let me create one...")
        # Create a simple test pattern
        from utils.image_density import create_sample_image_if_needed
        image_path = create_sample_image_if_needed()
        print(f"✅ Created test image: {image_path}")
    
    print(f"📸 Using image: {image_path}")
    
    # Setup parameters for quick demo
    n_points = 1000  # Fewer points for faster demo
    n_iterations = 20  # Fewer iterations for speed
    
    print(f"🔧 Parameters: {n_points} points, {n_iterations} iterations")
    print("⏳ Running stippling algorithm...")
    
    # Create image density function
    img_density = ImageDensityFunction(image_path, invert=True, blur_sigma=0.8)
    density_func = img_density.get_density_function()
    
    # Run Standard Lloyd algorithm (fastest)
    domain = (0, 1, 0, 1)
    algorithm = StandardLloydAlgorithm(domain)
    
    # Generate initial points
    np.random.seed(42)
    initial_points = algorithm.generate_random_points(n_points)
    
    # Run algorithm
    final_points, energy_history = algorithm.run(
        initial_points, 
        n_iterations=n_iterations, 
        density_func=density_func, 
        verbose=False
    )
    
    print(f"✅ Algorithm completed! Generated {len(final_points)} points")
    print(f"📊 Final energy: {energy_history[-1]:.6f}")
    
    # Create visualization showing where points appear
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Original image
    from PIL import Image
    original_img = np.array(Image.open(image_path).convert('L'))
    ax1.imshow(original_img, cmap='gray', origin='upper')
    ax1.set_title('1. Original Image', fontsize=14)
    ax1.axis('off')
    
    # 2. Density function visualization
    ax2.imshow(img_density.image_data, cmap='viridis', origin='lower',
              extent=[domain[0], domain[1], domain[2], domain[3]])
    ax2.set_title('2. Density Function\n(Yellow = High Density)', fontsize=14)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    
    # 3. Points overlaid on density
    ax3.imshow(img_density.image_data, cmap='viridis', origin='lower', alpha=0.3,
              extent=[domain[0], domain[1], domain[2], domain[3]])
    ax3.scatter(final_points[:, 0], final_points[:, 1], 
               s=8, c='red', alpha=0.8, edgecolors='black', linewidth=0.5)
    ax3.set_title('3. Generated Points\n(Red dots = Point locations)', fontsize=14)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_xlim(domain[0], domain[1])
    ax3.set_ylim(domain[2], domain[3])
    
    plt.tight_layout()
    
    # Save the demonstration in sample_output
    os.makedirs('sample_output', exist_ok=True)
    demo_file = 'sample_output/demo_stippling_explanation.png'
    plt.savefig(demo_file, dpi=200, bbox_inches='tight')
    plt.show()
    
    print(f"📁 Demonstration saved as: {demo_file}")
    
    # Also create a pure stippling result
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_facecolor('white')
    ax.scatter(final_points[:, 0], final_points[:, 1], 
               s=12, c='black', alpha=0.9)
    ax.set_xlim(domain[0], domain[1])
    ax.set_ylim(domain[2], domain[3])
    ax.set_aspect('equal')
    ax.set_title(f'Pure Stippling Result\n({len(final_points)} points)', fontsize=16)
    ax.axis('off')
    
    stippling_file = 'sample_output/demo_pure_stippling.png'
    plt.savefig(stippling_file, dpi=200, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"📁 Pure stippling saved as: {stippling_file}")
    
    # Show coordinate examples
    print("\n📍 POINT COORDINATES (first 10 points):")
    print("Format: (x, y) in normalized coordinates [0,1]")
    print("-" * 40)
    for i in range(min(10, len(final_points))):
        x, y = final_points[i]
        print(f"Point {i+1:2d}: ({x:.4f}, {y:.4f})")
    
    print(f"\n💾 All {len(final_points)} coordinates saved in output files")
    
    # Explain the coordinate system
    print("\n🗺️  COORDINATE SYSTEM EXPLANATION:")
    print("- Points are in normalized coordinates: (0,0) = bottom-left, (1,1) = top-right")
    print("- Dark areas in image → High density → More points")
    print("- Light areas in image → Low density → Fewer points")
    print("- Points follow the image structure while maintaining good spatial distribution")
    
    return final_points, image_path


if __name__ == "__main__":
    demo_stippling()
