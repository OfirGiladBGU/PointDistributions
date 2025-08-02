#!/usr/bin/env python3
"""
Quick demonstration of image stippling to show where points appear.
This script runs a fast stippling example and shows the results.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image, ImageDraw

# Add algorithm path
sys.path.append(os.path.join(os.path.dirname(__file__), 'algorithms', 'capacity_constrained'))
from paper_accurate_stippling import run_paper_accurate_stippling

def create_demo_image():
    """Create a simple demo image if none exists"""
    # Create a simple geometric pattern
    width, height = 400, 400
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw some geometric shapes
    draw.ellipse([100, 100, 300, 300], fill='black')  # Circle
    draw.rectangle([50, 350, 150, 380], fill='black')  # Rectangle
    draw.polygon([(200, 50), (350, 150), (250, 200)], fill='black')  # Triangle
    
    os.makedirs('input', exist_ok=True)
    demo_path = 'input/demo_pattern.png'
    img.save(demo_path)
    return demo_path

def demo_stippling():
    """Run a quick stippling demo to show where points appear."""
    
    print("🎨 QUICK STIPPLING DEMONSTRATION")
    print("="*50)
    
    # Check if we have test images in input folder
    test_images = [
        'input/Plant.png',
        'input/Buildings.png',
        'input/demo_pattern.png'
    ]
    image_path = None
    
    for img in test_images:
        if os.path.exists(img):
            image_path = img
            break
    
    if not image_path:
        print("❌ No test image found. Creating demo pattern...")
        image_path = create_demo_image()
        print(f"✅ Created test image: {image_path}")
    
    print(f"📸 Using image: {image_path}")
    
    # Setup parameters for quick demo
    n_points = 1000  # Fewer points for faster demo
    
    print(f"🔧 Parameters: {n_points} points")
    print("⏳ Running stippling algorithm...")
    
    # Run our paper-accurate stippling
    points = run_paper_accurate_stippling(
        image_path=image_path,
        num_points=n_points,
        output_dir='output'
    )
    
    if points is None:
        print("❌ Failed to generate points")
        return None, None
    
    print(f"✅ Algorithm completed! Generated {len(points)} points")
    
    # Create demonstration visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Load original image for display
    original_img = np.array(Image.open(image_path).convert('RGB'))
    
    # 1. Original image
    ax1.imshow(original_img)
    ax1.set_title('1. Original Image', fontsize=14)
    ax1.axis('off')
    
    # 2. Points on transparent background
    ax2.imshow(original_img, alpha=0.3)
    ax2.scatter(points[:, 0] * original_img.shape[1], 
               points[:, 1] * original_img.shape[0], 
               s=8, c='red', alpha=0.8, edgecolors='black', linewidth=0.5)
    ax2.set_title('2. Generated Points\n(Red dots = Point locations)', fontsize=14)
    ax2.axis('off')
    
    # 3. Pure stippling (black dots on white)
    ax3.set_facecolor('white')
    ax3.scatter(points[:, 0] * original_img.shape[1], 
               points[:, 1] * original_img.shape[0], 
               s=12, c='black', alpha=0.9)
    ax3.set_xlim(0, original_img.shape[1])
    ax3.set_ylim(original_img.shape[0], 0)  # Flip Y axis to match image
    ax3.set_title('3. Pure Stippling Result', fontsize=14)
    ax3.axis('off')
    
    plt.tight_layout()
    
    # Save the demonstration
    os.makedirs('output', exist_ok=True)
    demo_file = 'output/demo_stippling_explanation.png'
    plt.savefig(demo_file, dpi=200, bbox_inches='tight')
    plt.show()
    
    print(f"📁 Demonstration saved as: {demo_file}")
    
    # Show coordinate examples
    print("\n📍 POINT COORDINATES (first 10 points):")
    print("Format: (x, y) in normalized coordinates [0,1]")
    print("-" * 40)
    for i in range(min(10, len(points))):
        x, y = points[i]
        print(f"Point {i+1:2d}: ({x:.4f}, {y:.4f})")
    
    print(f"\n💾 All {len(points)} coordinates saved in output files")
    
    # Explain the coordinate system
    print("\n🗺️  COORDINATE SYSTEM EXPLANATION:")
    print("- Points are in normalized coordinates: (0,0) = top-left, (1,1) = bottom-right")
    print("- Dark areas in image → High density → More points")
    print("- Light areas in image → Low density → Fewer points")
    print("- Points follow the image structure while maintaining good spatial distribution")
    
    return points, image_path


if __name__ == "__main__":
    demo_stippling()
