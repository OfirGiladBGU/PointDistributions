#!/usr/bin/env python3
"""
Ultra-targeted flower detection that only accepts points on actual flower petals
"""

import numpy as np
from PIL import Image
from scipy import ndimage
import os

def create_ultra_targeted_flower_mask(image_path, debug=True):
    """
    Create an extremely targeted flower mask that only includes flower petals
    """
    # Load image
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img_array = np.array(img)
    height, width = img_array.shape[:2]
    
    # Convert to different color spaces for analysis
    r, g, b = img_array[:,:,0]/255.0, img_array[:,:,1]/255.0, img_array[:,:,2]/255.0
    gray = np.mean(img_array, axis=2) / 255.0
    
    print(f"Image size: {width}x{height}")
    print(f"RGB ranges - R: {r.min():.3f}-{r.max():.3f}, G: {g.min():.3f}-{g.max():.3f}, B: {b.min():.3f}-{b.max():.3f}")
    
    # Method 1: Color analysis - flower petals have specific color characteristics
    # Pink/purple flower detection
    pinkness = (r + b) / 2 - g  # Pink flowers have more red+blue than green
    
    # Method 2: Texture detection - flower petals have fine texture
    kernel_size = 7  # Smaller kernel for fine texture
    local_std = ndimage.generic_filter(gray, np.std, size=kernel_size)
    
    # Method 3: Edge density - flower petals have many small edges
    grad_x = ndimage.sobel(gray, axis=1)
    grad_y = ndimage.sobel(gray, axis=0)
    edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Method 4: Brightness variation within local regions
    local_mean = ndimage.uniform_filter(gray, size=kernel_size)
    brightness_variation = np.abs(gray - local_mean)
    
    print(f"Feature ranges:")
    print(f"  Pinkness: {pinkness.min():.3f} to {pinkness.max():.3f}")
    print(f"  Local std: {local_std.min():.3f} to {local_std.max():.3f}")
    print(f"  Edge magnitude: {edge_magnitude.min():.3f} to {edge_magnitude.max():.3f}")
    print(f"  Brightness variation: {brightness_variation.min():.3f} to {brightness_variation.max():.3f}")
    
    # Normalize features
    pinkness_norm = np.clip((pinkness - pinkness.min()) / (pinkness.max() - pinkness.min()), 0, 1)
    texture_norm = np.clip((local_std - local_std.min()) / (local_std.max() - local_std.min()), 0, 1)
    edge_norm = np.clip((edge_magnitude - edge_magnitude.min()) / (edge_magnitude.max() - edge_magnitude.min()), 0, 1)
    variation_norm = np.clip((brightness_variation - brightness_variation.min()) / (brightness_variation.max() - brightness_variation.min()), 0, 1)
    
    # Create combined flower score with more aggressive weighting
    flower_score = (0.4 * pinkness_norm +     # Color is important
                   0.3 * texture_norm +       # Fine texture is key
                   0.2 * edge_norm +          # Edges indicate detail
                   0.1 * variation_norm)      # Local variation
    
    print(f"Combined flower score range: {flower_score.min():.3f} to {flower_score.max():.3f}")
    
    # Use MUCH more aggressive thresholding - only top 10% of pixels
    threshold = np.percentile(flower_score, 90)  # Only top 10%!
    flower_mask = flower_score > threshold
    
    initial_flower_pixels = np.sum(flower_mask)
    print(f"Initial threshold {threshold:.3f}: {initial_flower_pixels} pixels ({initial_flower_pixels/flower_mask.size*100:.1f}%)")
    
    # Additional constraint: flower regions should not be pure white (background)
    # and not be pure black (shadows)
    too_bright = gray > 0.9
    too_dark = gray < 0.1
    flower_mask = flower_mask & ~too_bright & ~too_dark
    
    filtered_flower_pixels = np.sum(flower_mask)
    print(f"After brightness filtering: {filtered_flower_pixels} pixels ({filtered_flower_pixels/flower_mask.size*100:.1f}%)")
    
    # Clean up with morphological operations
    from scipy.ndimage import binary_opening, binary_closing, binary_erosion
    
    # Remove small isolated regions
    flower_mask = binary_opening(flower_mask, structure=np.ones((3,3)))
    
    # Fill small gaps
    flower_mask = binary_closing(flower_mask, structure=np.ones((5,5)))
    
    # Erode slightly to be more conservative
    flower_mask = binary_erosion(flower_mask, structure=np.ones((2,2)))
    
    final_flower_pixels = np.sum(flower_mask)
    print(f"After morphological operations: {final_flower_pixels} pixels ({final_flower_pixels/flower_mask.size*100:.1f}%)")
    
    # Create density function with extreme contrast
    density = flower_mask.astype(float) * 0.95  # Strong signal on flower
    
    # Very slight smoothing to avoid pixelation but maintain sharp boundaries
    density = ndimage.gaussian_filter(density, sigma=1)
    
    # Tiny background probability to allow some points elsewhere
    density = density + 0.01  # Much smaller background probability
    
    print(f"Final density range: {density.min():.3f} to {density.max():.3f}")
    print(f"Expected acceptance rate: {density.mean():.4f} (should be < 0.1 for good concentration)")
    
    # Save debug info
    if debug:
        np.save('debug_ultra_pinkness.npy', pinkness_norm)
        np.save('debug_ultra_texture.npy', texture_norm)
        np.save('debug_ultra_edges.npy', edge_norm)
        np.save('debug_ultra_variation.npy', variation_norm)
        np.save('debug_ultra_flower_score.npy', flower_score)
        np.save('debug_ultra_flower_mask.npy', flower_mask)
        print("Saved ultra debug arrays as .npy files")
    
    return density

def test_ultra_targeted():
    """Test the ultra-targeted approach"""
    sample_dir = r"c:\Users\ofirg\PycharmProjects\PointDistributions\sample_output"
    f_in_path = os.path.join(sample_dir, "F_in.png")
    
    if os.path.exists(f_in_path):
        print("Testing ultra-targeted flower detection...")
        density = create_ultra_targeted_flower_mask(f_in_path, debug=True)
        
        # Save the density
        np.save('flower_density_ultra.npy', density)
        print("Saved ultra-targeted density to 'flower_density_ultra.npy'")
        
        # Quick test of point generation
        print("\\nTesting point generation...")
        height, width = density.shape
        points = []
        attempts = 0
        max_attempts = 50000
        
        while len(points) < 1000 and attempts < max_attempts:
            x = np.random.rand()
            y = np.random.rand()
            
            img_x = int(x * (width - 1))
            img_y = int(y * (height - 1))
            
            density_val = density[img_y, img_x]
            
            if np.random.rand() < density_val:
                points.append([x, y])
            
            attempts += 1
        
        points = np.array(points)
        acceptance_rate = len(points) / attempts if attempts > 0 else 0
        
        print(f"Test results:")
        print(f"  Generated {len(points)} points in {attempts} attempts")
        print(f"  Acceptance rate: {acceptance_rate:.4f}")
        
        if acceptance_rate < 0.1:
            print(f"  ✅ Good! Low acceptance rate means high concentration")
        else:
            print(f"  ⚠️  Acceptance rate still too high")
        
        return density
    else:
        print(f"Could not find {f_in_path}")
        return None

if __name__ == "__main__":
    test_ultra_targeted()
