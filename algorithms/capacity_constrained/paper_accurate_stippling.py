#!/usr/bin/env python3
"""
Paper-Accurate Capacity-Constrained Voronoi Tessellation (CCVT) Algorithm
Based on Balzer2009: "Capacity-constrained Point Distributions: A Variant of Lloyd's Method"

Implementation of the full CCVT algorithm including:
1. Initial point distribution using density-based rejection sampling
2. Capacity-constrained Voronoi tessellation optimization
3. Iterative point swapping and energy minimization
4. Centroidal optimization (optional)

This produces paper-quality results:
- Plant.png: 20,000 points (Figure 7c)
- Buildings.png: 3,000 points (Figure 8 detail)
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from scipy.spatial import Voronoi, distance_matrix
from scipy.spatial.distance import cdist
import os
import time
from typing import List, Tuple, Optional

class CCVTSite:
    """Represents a Voronoi site with capacity constraint"""
    
    def __init__(self, site_id: int, location: np.ndarray, capacity: int):
        self.id = site_id
        self.location = location.copy()
        self.capacity = capacity
        self.assigned_points = []
        self.energy = 0.0
        self.stable = False
    
    def assign_point(self, point_idx: int, point: np.ndarray, distance_sq: float):
        """Assign a point to this site"""
        self.assigned_points.append(point_idx)
        self.energy += distance_sq
    
    def clear_assignments(self):
        """Clear all point assignments"""
        self.assigned_points.clear()
        self.energy = 0.0
        self.stable = False
    
    def compute_centroid(self, points: np.ndarray) -> np.ndarray:
        """Compute centroid of assigned points"""
        if not self.assigned_points:
            return self.location.copy()
        assigned = points[self.assigned_points]
        return np.mean(assigned, axis=0)


class CapacityConstrainedVoronoiTessellation:
    """
    Implementation of Capacity-Constrained Voronoi Tessellation (CCVT)
    Based on Balzer2009: "Capacity-constrained Point Distributions"
    
    Algorithm workflow:
    1. Initialize sites with random positions and equal capacities
    2. Assign points to closest sites (standard Voronoi)
    3. For each pair of sites, attempt beneficial point swaps
    4. Optionally move sites to centroids of assigned points
    5. Repeat until stable (no beneficial swaps found)
    """
    
    def __init__(self, domain_bounds: Tuple[float, float, float, float] = (0, 1, 0, 1)):
        self.xmin, self.xmax, self.ymin, self.ymax = domain_bounds
        self.sites = []
        self.points = None
        self.iteration_count = 0
        
    def initialize_sites(self, points: np.ndarray, num_sites: int) -> None:
        """
        Initialize Voronoi sites with equal capacity
        
        Args:
            points: Point distribution to tessellate  
            num_sites: Number of Voronoi sites to create
        """
        self.points = points.copy()
        self.sites = []
        
        n_points = len(points)
        
        # Calculate equal capacity for each site
        base_capacity = n_points // num_sites
        extra_points = n_points % num_sites
        
        # Initialize sites with random locations
        np.random.seed(42)  # For reproducible results
        
        for i in range(num_sites):
            # Random location within domain
            x = np.random.uniform(self.xmin, self.xmax)
            y = np.random.uniform(self.ymin, self.ymax)
            location = np.array([x, y])
            
            # Distribute remaining points evenly
            capacity = base_capacity + (1 if i < extra_points else 0)
            
            site = CCVTSite(i, location, capacity)
            self.sites.append(site)
            
        print(f"Initialized {num_sites} sites with capacities: {[s.capacity for s in self.sites]}")
    
    def assign_points_to_sites(self) -> None:
        """
        Capacity-constrained assignment of points to sites
        Uses priority queue to ensure capacity constraints are respected
        """
        
        # Clear previous assignments
        for site in self.sites:
            site.clear_assignments()
        
        # Compute distance matrix between points and sites
        site_locations = np.array([site.location for site in self.sites])
        distances = cdist(self.points, site_locations, 'euclidean')
        
        # Create priority queue: (distance, point_idx, site_idx)
        import heapq
        candidates = []
        
        for point_idx in range(len(self.points)):
            for site_idx in range(len(self.sites)):
                distance = distances[point_idx, site_idx]
                heapq.heappush(candidates, (distance, point_idx, site_idx))
        
        # Assign points respecting capacity constraints
        assigned_points = set()
        
        while candidates and len(assigned_points) < len(self.points):
            distance, point_idx, site_idx = heapq.heappop(candidates)
            
            # Skip if point already assigned or site at capacity
            if point_idx in assigned_points:
                continue
            if len(self.sites[site_idx].assigned_points) >= self.sites[site_idx].capacity:
                continue
            
            # Assign point to site
            site = self.sites[site_idx]
            distance_sq = distance ** 2
            site.assign_point(point_idx, self.points[point_idx], distance_sq)
            assigned_points.add(point_idx)
        
        # Verify all points are assigned
        total_assigned = sum(len(site.assigned_points) for site in self.sites)
        print(f"Capacity-constrained assignment: {total_assigned}/{len(self.points)} points assigned")
    
    def optimize_iteration(self, centroidalize: bool = True) -> bool:
        """
        Perform one CCVT optimization iteration
        
        Implementation of Algorithm 1 from the paper:
        - For each pair of sites, find beneficial point swaps
        - Perform swaps greedily based on energy improvement
        - Optionally move sites to centroids
        
        Args:
            centroidalize: Whether to move sites to centroids after swapping
            
        Returns:
            True if stable (no beneficial swaps found), False otherwise
        """
        
        self.iteration_count += 1
        
        # Track stability for each site
        site_stability = [True] * len(self.sites)
        total_swaps = 0
        total_energy_improvement = 0.0
        
        # Check all pairs of sites for beneficial swaps
        for i in range(len(self.sites)):
            for j in range(i + 1, len(self.sites)):
                site1 = self.sites[i]
                site2 = self.sites[j]
                
                # Skip if both sites are stable and far apart
                if site1.stable and site2.stable:
                    dist = np.linalg.norm(site1.location - site2.location)
                    # Skip if sites are far apart (optimization)
                    if dist > 0.5:  # Heuristic threshold
                        continue
                
                # Consider all point pairs for swapping
                swap_candidates = []
                
                for p1_idx in site1.assigned_points:
                    for p2_idx in site2.assigned_points:
                        
                        # Current energies
                        e1_current = np.sum((self.points[p1_idx] - site1.location) ** 2)
                        e2_current = np.sum((self.points[p2_idx] - site2.location) ** 2)
                        current_energy = e1_current + e2_current
                        
                        # Energy after swap
                        e1_swap = np.sum((self.points[p1_idx] - site2.location) ** 2) 
                        e2_swap = np.sum((self.points[p2_idx] - site1.location) ** 2)
                        swap_energy = e1_swap + e2_swap
                        
                        energy_improvement = current_energy - swap_energy
                        
                        if energy_improvement > 1e-8:  # Beneficial swap threshold
                            swap_candidates.append((p1_idx, p2_idx, energy_improvement))
                
                # Sort swaps by energy improvement (best first)
                swap_candidates.sort(key=lambda x: x[2], reverse=True)
                
                # Perform beneficial swaps greedily
                performed_swaps = []
                for p1_idx, p2_idx, improvement in swap_candidates:
                    # Check if points haven't been swapped already in this iteration
                    if any((p1_idx in swap or p2_idx in swap) for swap in performed_swaps):
                        continue
                    
                    # Perform the swap
                    site1.assigned_points.remove(p1_idx)
                    site2.assigned_points.remove(p2_idx)
                    
                    # Update energies
                    site1.energy -= np.sum((self.points[p1_idx] - site1.location) ** 2)
                    site2.energy -= np.sum((self.points[p2_idx] - site2.location) ** 2)
                    
                    # Add to new sites
                    site2.assigned_points.append(p1_idx)
                    site1.assigned_points.append(p2_idx)
                    
                    site2.energy += np.sum((self.points[p1_idx] - site2.location) ** 2)
                    site1.energy += np.sum((self.points[p2_idx] - site1.location) ** 2)
                    
                    performed_swaps.append((p1_idx, p2_idx))
                    total_swaps += 1
                    total_energy_improvement += improvement
                    
                    # Mark sites as unstable
                    site_stability[i] = False
                    site_stability[j] = False
                
                # If swaps were made, update site locations (centroidalize)
                if performed_swaps and centroidalize:
                    site1.location = site1.compute_centroid(self.points)
                    site2.location = site2.compute_centroid(self.points)
                    
                    # Recompute energies after centroidal update
                    site1.energy = sum(np.sum((self.points[p] - site1.location) ** 2) 
                                     for p in site1.assigned_points)
                    site2.energy = sum(np.sum((self.points[p] - site2.location) ** 2) 
                                     for p in site2.assigned_points)
        
        # Update site stability
        for i, stable in enumerate(site_stability):
            self.sites[i].stable = stable
        
        # Compute total energy
        total_energy = sum(site.energy for site in self.sites)
        
        print(f"Iteration {self.iteration_count}: {total_swaps} swaps, "
              f"Energy = {total_energy:.6f}, Improvement = {total_energy_improvement:.6f}")
        
        # Return True if stable (no swaps made)
        return total_swaps == 0
    
    def run_ccvt_optimization(self, max_iterations: int = 50, centroidalize: bool = True) -> np.ndarray:
        """
        Run the complete CCVT optimization
        
        Args:
            max_iterations: Maximum number of iterations
            centroidalize: Whether to move sites to centroids
            
        Returns:
            Final site locations
        """
        
        print(f"\n🔄 Running CCVT optimization...")
        print(f"Sites: {len(self.sites)}, Points: {len(self.points)}")
        
        # Initial assignment
        self.assign_points_to_sites()
        initial_energy = sum(site.energy for site in self.sites)
        print(f"Initial energy: {initial_energy:.6f}")
        
        # Optimization loop
        start_time = time.time()
        
        for iteration in range(max_iterations):
            stable = self.optimize_iteration(centroidalize)
            
            if stable:
                print(f"✅ Converged after {iteration + 1} iterations")
                break
        else:
            print(f"⚠️  Reached maximum iterations ({max_iterations})")
        
        # Final energy
        final_energy = sum(site.energy for site in self.sites)
        optimization_time = time.time() - start_time
        
        print(f"Final energy: {final_energy:.6f} (improvement: {initial_energy - final_energy:.6f})")
        print(f"Optimization completed in {optimization_time:.2f} seconds")
        
        # Return final site locations
        return np.array([site.location for site in self.sites])


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


def generate_ccvt_points_from_density(density: np.ndarray, num_sites: int, 
                                      points_per_site_ratio: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate CCVT-optimized points from density function
    
    Args:
        density: 2D density array
        num_sites: Number of Voronoi sites 
        points_per_site_ratio: Ratio of discrete points to sites (for optimization)
        
    Returns:
        (sites, discrete_points): Final site locations and discrete point distribution
    """
    
    height, width = density.shape
    num_discrete_points = num_sites * points_per_site_ratio
    
    print(f"Generating {num_discrete_points} discrete points for {num_sites} sites...")
    
    # Step 1: Generate discrete points using rejection sampling
    discrete_points = []
    attempts = 0
    max_attempts = num_discrete_points * 100
    
    while len(discrete_points) < num_discrete_points and attempts < max_attempts:
        x = np.random.rand()
        y = np.random.rand()
        
        img_x = int(x * (width - 1))
        img_y = int(y * (height - 1))
        
        density_val = density[img_y, img_x]
        
        # Paper-accurate approach: Only accept if density > 0
        if density_val > 0 and np.random.rand() < density_val:
            discrete_points.append([x, y])
        
        attempts += 1
        
        if len(discrete_points) % 1000 == 0 and len(discrete_points) > 0:
            print(f"   Generated {len(discrete_points)}/{num_discrete_points} discrete points")
    
    discrete_points = np.array(discrete_points)
    print(f"✅ Generated {len(discrete_points)} discrete points")
    
    # Step 2: Run CCVT optimization
    ccvt = CapacityConstrainedVoronoiTessellation()
    ccvt.initialize_sites(discrete_points, num_sites)
    
    # Run optimization  
    final_sites = ccvt.run_ccvt_optimization(max_iterations=50, centroidalize=True)
    
    print(f"✅ CCVT optimization complete: {len(final_sites)} sites")
    
    return final_sites, discrete_points
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


def run_paper_accurate_ccvt(image_path, num_points, output_dir="output", threshold=0.35):
    """
    Main function to run paper-accurate CCVT algorithm
    
    Args:
        image_path: Path to input image
        num_points: Number of final stippling points (sites)
        output_dir: Output directory for results
        threshold: Density threshold for aggressive filtering (default: 0.35)
        
    Returns:
        points: Generated CCVT-optimized points
    """
    
    print(f"\\n{'='*60}")
    print(f"PAPER-ACCURATE CCVT ALGORITHM")
    print(f"{'='*60}")
    print(f"Image: {image_path}")
    print(f"Final Points (Sites): {num_points}")
    print(f"Output: {output_dir}")
    
    start_time = time.time()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Generate paper-accurate density
    print(f"\\n🎯 Loading/creating paper-accurate density...")
    density = load_or_create_density(image_path, output_dir, threshold)
    
    # Step 2: Generate CCVT-optimized points
    print(f"\\n🔄 Running CCVT optimization...")
    sites, discrete_points = generate_ccvt_points_from_density(
        density, num_points, points_per_site_ratio=4
    )
    
    if len(sites) == 0:
        print("ERROR: No sites generated!")
        return None
    
    print(f"✅ Generated {len(sites)} CCVT sites from {len(discrete_points)} discrete points")
    
    # Step 3: Create output files
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    algo_name = "ccvt"
    
    print("\\n📁 Creating output files...")
    
    # 1. Clean stippling (black dots on white background)
    clean_path = os.path.join(output_dir, f"{image_name}_{algo_name}_clean_stippling.png")
    create_clean_stippling(sites, image_path, clean_path)
    
    # 2. Points visualization (red points over original image)
    points_viz_path = os.path.join(output_dir, f"{image_name}_{algo_name}_points_visualization.png")
    create_points_visualization(sites, image_path, points_viz_path)
    
    # 3. Voronoi centers visualization
    centers_path = os.path.join(output_dir, f"{image_name}_{algo_name}_centers_visualization.png")
    create_centers_visualization(sites, image_path, centers_path)
    
    # 4. Save point coordinates
    points_path = os.path.join(output_dir, f"{image_name}_{algo_name}_points.txt")
    centroids_path = os.path.join(output_dir, f"{image_name}_{algo_name}_centroids.txt")
    np.savetxt(points_path, sites, fmt='%.6f', header='x y (CCVT sites in normalized coordinates)')
    np.savetxt(centroids_path, sites, fmt='%.6f', header='x y (CCVT centroids in normalized coordinates)')
    
    # 5. Save discrete points used for optimization
    discrete_path = os.path.join(output_dir, f"{image_name}_{algo_name}_discrete_points.txt") 
    np.savetxt(discrete_path, discrete_points, fmt='%.6f', header='x y (discrete points used for CCVT optimization)')
    
    # 6. Save metadata
    metadata_path = os.path.join(output_dir, f"{image_name}_{algo_name}_metadata.txt")
    with open(metadata_path, 'w') as f:
        f.write(f"Image: {os.path.basename(image_path)}\\n")
        f.write(f"Algorithm: CCVT (Capacity-Constrained Voronoi Tessellation)\\n")
        f.write(f"Final sites: {len(sites)}\\n")
        f.write(f"Discrete points: {len(discrete_points)}\\n")
        f.write(f"Points per site ratio: 4\\n")
        f.write(f"Density threshold: {threshold}\\n")
        f.write(f"Generation time: {time.time() - start_time:.2f} seconds\\n")
        f.write(f"Coordinate system: normalized [0,1] x [0,1]\\n")
        f.write(f"Density file: {image_name}_density_t{str(threshold).replace('.', '_')}.npy\\n")
    
    total_time = time.time() - start_time
    print(f"\\n✅ CCVT Complete! Generated in {total_time:.2f} seconds")
    print(f"📁 Files created:")
    print(f"   • {clean_path} (Clean stippling)")
    print(f"   • {points_viz_path} (Points visualization)")  
    print(f"   • {centers_path} (Centers visualization)")
    print(f"   • {points_path} (Final CCVT sites)")
    print(f"   • {centroids_path} (CCVT centroids)")
    print(f"   • {discrete_path} (Discrete points)")
    print(f"   • {metadata_path} (Metadata)")
    
    return sites

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
    
    points = run_paper_accurate_ccvt(image_path, num_points)
    
    if points is not None:
        print("\\n🎉 SUCCESS! CCVT stippling completed.")
    else:
        print("\\n❌ FAILED: Could not generate CCVT stippling.")
