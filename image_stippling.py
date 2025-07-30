#!/usr/bin/env python3
"""
Image Stippling with Point Distribution Algorithms

This script allows you to create stippling effects from images using the point
distribution algorithms, similar to the results shown in the paper.

Usage: python image_stippling.py

Features:
- Load any image and convert to stippling
- Multiple algorithm choices (Lloyd, Capacity-Constrained, Optimized Paper)
- Adjustable point density and processing parameters
- Export both point coordinates and stippling visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from utils.image_density import (ImageDensityFunction, create_image_based_algorithms_runner,
                                create_stippling_visualization, get_sample_images_info,
                                create_sample_image_if_needed)
from algorithms.standard_lloyd import StandardLloydAlgorithm
from algorithms.capacity_constrained import (CapacityConstrainedDistributionAlgorithm,
                                            OptimizedCapacityConstrainedVoronoiAlgorithm)


def ensure_output_dir():
    """Ensure the output directory exists."""
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_points_to_txt(points, filename, algorithm_name="Image Stippling"):
    """Save points to a text file."""
    header = f"{algorithm_name} - Final Point Positions\nFormat: x y\nTotal Points: {len(points)}"
    np.savetxt(filename, points, fmt='%.6f', delimiter='\t', header=header)
    print(f"Points saved to: {filename}")


def find_available_images():
    """Find image files in the current directory and sample_output directory."""
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
    images = []
    
    # Check main directory
    for file in os.listdir('.'):
        if os.path.splitext(file.lower())[1] in image_extensions:
            images.append(file)
    
    # Check sample_output directory if it exists
    sample_dir = 'sample_output'
    if os.path.exists(sample_dir):
        for file in os.listdir(sample_dir):
            if os.path.splitext(file.lower())[1] in image_extensions:
                images.append(os.path.join(sample_dir, file))
    
    return sorted(images)


def run_image_stippling_algorithm(image_path, algorithm_choice, n_points, n_iterations, 
                                 invert, blur_sigma, output_dir):
    """Run a specific algorithm on an image."""
    
    print(f"\n{'='*70}")
    print(f"RUNNING {algorithm_choice.upper()} ALGORITHM ON IMAGE")
    print('='*70)
    
    # Create image density function
    img_density = ImageDensityFunction(image_path, invert=invert, blur_sigma=blur_sigma)
    density_func = img_density.get_density_function()
    
    # Set up algorithm
    domain = (0, 1, 0, 1)
    if algorithm_choice == 'lloyd':
        algorithm = StandardLloydAlgorithm(domain)
        algorithm_name = "Standard Lloyd"
    elif algorithm_choice == 'capacity':
        algorithm = CapacityConstrainedDistributionAlgorithm(domain)
        algorithm_name = "Capacity-Constrained"
    elif algorithm_choice == 'optimized':
        algorithm = OptimizedCapacityConstrainedVoronoiAlgorithm(domain)
        algorithm_name = "Optimized Paper Algorithm"
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_choice}")
    
    # Generate initial points
    np.random.seed(42)
    initial_points = algorithm.generate_random_points(n_points)
    
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Algorithm: {algorithm_name}")
    print(f"Points: {n_points}")
    print(f"Iterations: {n_iterations}")
    print(f"Invert colors: {invert}")
    print(f"Blur sigma: {blur_sigma}")
    print("-" * 50)
    
    # Run algorithm
    if algorithm_choice == 'lloyd':
        final_points, energy_history = algorithm.run(
            initial_points, n_iterations=n_iterations, 
            density_func=density_func, verbose=True)
        print(f"Final energy: {energy_history[-1]:.6f}")
        
    elif algorithm_choice == 'capacity':
        final_points, energy_history, capacity_variance_history, combined_energy_history = algorithm.run(
            initial_points, density_func, n_iterations=n_iterations, verbose=True)
        analysis = algorithm.analyze_capacity_distribution(final_points, density_func)
        print(f"Final CVT energy: {analysis['cvt_energy']:.6f}")
        print(f"Blue noise quality: {analysis['blue_noise_quality']:.4f}")
        
    else:  # optimized
        final_points, energy_history, capacity_variance_history = algorithm.run(
            initial_points, density_func, n_iterations=n_iterations, verbose=True)
        print(f"Final energy: {energy_history[-1]:.6f}")
        print(f"Final capacity variance: {capacity_variance_history[-1]:.6f}")
    
    # Create output filenames
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    base_name = f"stippling_{algorithm_choice}_{image_name}_{n_points}pts"
    
    # Save points
    txt_file = os.path.join(output_dir, f"{base_name}.txt")
    save_points_to_txt(final_points, txt_file, f"{algorithm_name} Image Stippling")
    
    # Create stippling visualization
    stippling_file = os.path.join(output_dir, f"{base_name}_stippling.png")
    create_stippling_visualization(
        final_points, image_path, stippling_file, 
        title=f"{algorithm_name} Stippling", 
        point_size=300/n_points*10  # Adjust point size based on density
    )
    
    print(f"\n✅ {algorithm_name} stippling completed!")
    print(f"📁 Files: {base_name}.txt/.png")
    
    return final_points, base_name


def main():
    """Main interface for image stippling."""
    output_dir = ensure_output_dir()
    
    print("\n" + "="*80)
    print("IMAGE STIPPLING WITH POINT DISTRIBUTION ALGORITHMS")
    print("="*80)
    print("Create stippling effects from images using various algorithms")
    print("Similar to the results shown in the research paper")
    
    while True:
        print("\n" + "="*80)
        print("IMAGE STIPPLING MENU")
        print("="*80)
        
        # Find available images
        available_images = find_available_images()
        
        if not available_images:
            print("\n⚠️  No image files found in current directory!")
            print("\nTo use this tool:")
            print("1. Place image files (PNG, JPG, etc.) in the project directory")
            print("2. Or create a sample image using option 'S'")
            print("\nOptions:")
            print("S. Create sample image")
            print("I. Show usage information")
            print("0. Exit")
        else:
            print("\nAvailable images:")
            for i, img in enumerate(available_images, 1):
                print(f"{i:2d}. {img}")
            
            print("\nOptions:")
            print("R. Refresh image list")
            print("S. Create sample image") 
            print("I. Show usage information")
            print("0. Exit")
        
        print("\n" + "="*80)
        
        try:
            choice = input("\nEnter your choice: ").strip().upper()
            
            if choice == '0':
                print("Goodbye!")
                break
                
            elif choice == 'S':
                # Create sample image
                print("\nCreating sample pattern image...")
                sample_path = create_sample_image_if_needed()
                print(f"✅ Sample image created: {sample_path}")
                print("You can now use this image for stippling!")
                continue
                
            elif choice == 'I':
                # Show information
                info = get_sample_images_info()
                print(f"\n{info['description']}")
                print("\nRequirements:")
                for req in info['requirements']:
                    print(f"  • {req}")
                print("\nTips for best results:")
                for tip in info['tips']:
                    print(f"  • {tip}")
                continue
                
            elif choice == 'R':
                # Refresh list
                print("Refreshing image list...")
                continue
                
            elif choice.isdigit():
                # Select image by number
                img_index = int(choice) - 1
                if 0 <= img_index < len(available_images):
                    selected_image = available_images[img_index]
                else:
                    print(f"❌ Invalid image number. Choose 1-{len(available_images)}")
                    continue
            else:
                print("❌ Invalid choice. Please try again.")
                continue
            
            # Get stippling parameters
            print(f"\n📸 Selected image: {selected_image}")
            print("\nStippling Parameters:")
            print("-" * 40)
            
            try:
                n_points = int(input("Number of points (default 5000): ") or "5000")
                n_iterations = int(input("Number of iterations (default 50): ") or "50")
                
                print("\nAlgorithm options:")
                print("1. Standard Lloyd - Classic CVT algorithm")
                print("2. Capacity-Constrained - Blue noise distribution")
                print("3. Optimized Paper - Fast heap-based algorithm (recommended)")
                algo_choice = input("Choose algorithm (1-3, default 3): ").strip() or "3"
                
                algo_map = {'1': 'lloyd', '2': 'capacity', '3': 'optimized'}
                algorithm = algo_map.get(algo_choice, 'optimized')
                
                # Image processing options
                print("\nImage processing options:")
                invert_choice = input("Invert colors? Dark areas = more points (Y/n): ").strip().lower()
                invert = invert_choice != 'n'
                
                try:
                    blur_sigma = float(input("Blur sigma for smoothing (0-3, default 1.0): ") or "1.0")
                except ValueError:
                    blur_sigma = 1.0
                
            except ValueError:
                print("Invalid input. Using defaults: 5000 points, 50 iterations, optimized algorithm")
                n_points, n_iterations, algorithm = 5000, 50, 'optimized'
                invert, blur_sigma = True, 1.0
            
            # Run stippling algorithm
            try:
                final_points, base_name = run_image_stippling_algorithm(
                    selected_image, algorithm, n_points, n_iterations,
                    invert, blur_sigma, output_dir
                )
                
                print(f"\n✅ Stippling completed successfully!")
                print(f"📁 Results saved to: {output_dir}")
                print(f"📊 Generated {len(final_points)} points")
                
                # Ask if user wants to run another algorithm on same image
                another = input("\nRun another algorithm on this image? (y/N): ").strip().lower()
                if another == 'y':
                    print("Returning to parameter selection...")
                    # Continue inner loop for same image
                    while True:
                        print(f"\n📸 Image: {selected_image}")
                        print("Choose different algorithm or parameters:")
                        
                        try:
                            new_points = int(input(f"Number of points (current: {n_points}): ") or str(n_points))
                            new_iterations = int(input(f"Number of iterations (current: {n_iterations}): ") or str(n_iterations))
                            
                            print("\nAlgorithm options:")
                            print("1. Standard Lloyd")
                            print("2. Capacity-Constrained") 
                            print("3. Optimized Paper")
                            new_algo_choice = input("Choose algorithm (1-3): ").strip()
                            new_algorithm = algo_map.get(new_algo_choice, algorithm)
                            
                            new_invert_choice = input(f"Invert colors? (current: {invert}) (Y/n): ").strip().lower()
                            new_invert = new_invert_choice != 'n' if new_invert_choice else invert
                            
                            try:
                                new_blur = float(input(f"Blur sigma (current: {blur_sigma}): ") or str(blur_sigma))
                            except ValueError:
                                new_blur = blur_sigma
                                
                        except ValueError:
                            print("Using previous parameters...")
                            new_points, new_iterations, new_algorithm = n_points, n_iterations, algorithm
                            new_invert, new_blur = invert, blur_sigma
                        
                        # Run with new parameters
                        final_points2, base_name2 = run_image_stippling_algorithm(
                            selected_image, new_algorithm, new_points, new_iterations,
                            new_invert, new_blur, output_dir
                        )
                        
                        # Ask if they want to continue with this image
                        continue_same = input("\nRun another variation? (y/N): ").strip().lower()
                        if continue_same != 'y':
                            break
                
            except Exception as e:
                print(f"❌ Error processing image: {e}")
                print("Please try again with different parameters.")
            
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")


if __name__ == "__main__":
    main()
