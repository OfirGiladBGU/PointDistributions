#!/usr/bin/env python3
"""
Create sample images for testing image stippling functionality.

This script creates various test images that demonstrate different
aspects of the stippling algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os


def create_simple_shapes():
    """Create a simple image with geometric shapes."""
    img_size = 512
    img = Image.new('L', (img_size, img_size), 255)  # White background
    draw = ImageDraw.Draw(img)
    
    # Draw some shapes
    # Circle
    draw.ellipse([100, 100, 200, 200], fill=0, outline=0)  # Black circle
    
    # Rectangle
    draw.rectangle([300, 100, 450, 250], fill=80, outline=0)  # Gray rectangle
    
    # Triangle (approximate with polygon)
    triangle_points = [(250, 300), (200, 400), (300, 400)]
    draw.polygon(triangle_points, fill=40, outline=0)  # Dark gray triangle
    
    # Some text
    try:
        # Try to use a font, fall back to default if not available
        font = ImageFont.load_default()
        draw.text((50, 450), "STIPPLING", font=font, fill=0)
    except:
        draw.text((50, 450), "STIPPLING", fill=0)
    
    return img


def create_gradient_image():
    """Create an image with gradients."""
    img_size = 512
    img_array = np.zeros((img_size, img_size), dtype=np.uint8)
    
    # Create radial gradient
    center_x, center_y = img_size // 2, img_size // 2
    for y in range(img_size):
        for x in range(img_size):
            # Distance from center
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            # Normalize and invert (center = white, edges = black)
            normalized_dist = min(dist / (img_size * 0.7), 1.0)
            img_array[y, x] = int(255 * (1 - normalized_dist))
    
    return Image.fromarray(img_array, mode='L')


def create_pattern_image():
    """Create an image with interesting mathematical patterns."""
    img_size = 512
    x = np.linspace(-4, 4, img_size)
    y = np.linspace(-4, 4, img_size)
    X, Y = np.meshgrid(x, y)
    
    # Create an interesting pattern combining multiple functions
    pattern1 = np.sin(X * 2) * np.cos(Y * 2)
    pattern2 = np.exp(-(X**2 + Y**2) / 4)
    pattern3 = np.sin(np.sqrt(X**2 + Y**2) * 3) / (np.sqrt(X**2 + Y**2) + 0.1)
    
    # Combine patterns
    combined = pattern1 * 0.3 + pattern2 * 0.4 + pattern3 * 0.3
    
    # Normalize to [0, 255]
    combined = (combined - combined.min()) / (combined.max() - combined.min())
    combined = (combined * 255).astype(np.uint8)
    
    return Image.fromarray(combined, mode='L')


def create_flower_like_pattern():
    """Create a flower-like pattern similar to the paper example."""
    img_size = 512
    center_x, center_y = img_size // 2, img_size // 2
    img_array = np.zeros((img_size, img_size), dtype=np.uint8)
    
    for y in range(img_size):
        for x in range(img_size):
            # Convert to polar coordinates
            dx, dy = x - center_x, y - center_y
            r = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)
            
            # Create flower-like pattern
            petals = 8
            petal_shape = np.cos(petals * theta) * 0.5 + 0.5
            radial_falloff = np.exp(-r / 120)  # Falloff from center
            
            # Combine patterns
            intensity = petal_shape * radial_falloff
            
            # Add some texture
            noise = np.sin(r * 0.1) * 0.1 + 0.9
            intensity *= noise
            
            # Convert to grayscale value
            img_array[y, x] = int(255 * (1 - intensity))  # Invert so petals are dark
    
    return Image.fromarray(img_array, mode='L')


def main():
    """Create various sample images for testing."""
    print("Creating sample images for stippling...")
    
    # Create sample_output directory if it doesn't exist
    sample_dir = 'sample_output'
    os.makedirs(sample_dir, exist_ok=True)
    
    # Create different types of sample images
    samples = {
        os.path.join(sample_dir, 'simple_shapes.png'): create_simple_shapes(),
        os.path.join(sample_dir, 'gradient.png'): create_gradient_image(),
        os.path.join(sample_dir, 'pattern.png'): create_pattern_image(),
        os.path.join(sample_dir, 'flower.png'): create_flower_like_pattern()
    }
    
    for filename, img in samples.items():
        img.save(filename)
        print(f"✅ Created: {filename}")
    
    print(f"\n🎨 Created {len(samples)} sample images in '{sample_dir}/' directory")
    print("\nTo use these images:")
    print("1. Run: python image_stippling.py (images will be automatically detected)")
    print("2. Select any image from the list")
    print("3. Choose your stippling parameters")
    
    print("\nRecommended settings for each image:")
    print("• simple_shapes.png: 2000-5000 points, blur_sigma=0.5")
    print("• gradient.png: 3000-8000 points, blur_sigma=1.0") 
    print("• pattern.png: 5000-15000 points, blur_sigma=0.5")
    print("• flower.png: 5000-15000 points, blur_sigma=0.8")


if __name__ == "__main__":
    main()
