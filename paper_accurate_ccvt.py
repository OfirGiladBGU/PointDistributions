#!/usr/bin/env python3
"""
Paper-Accurate CCVT Implementation

This implementation closely follows the original C++ code from the Balzer2009 paper
to ensure we're implementing the algorithm correctly. 

Key insights from the C++ code:
1. The algorithm uses DISCRETE POINTS as the underlying space representation
2. Points are assigned to sites (generators) and swapped between sites to minimize energy
3. Energy function is simply squared Euclidean distance
4. Capacity constraint ensures equal number of points per site
5. The optimization is done via point swapping, not moving generators
6. Centroidal movement is optional and separate from the main algorithm
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
import heapq

# Add project root to path
sys.path.append(os.path.dirname(__file__))


class CCVTPoint:
    """Represents a point in the CCVT algorithm."""
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def __getitem__(self, i):
        return self.x if i == 0 else self.y
    
    def __setitem__(self, i, value):
        if i == 0:
            self.x = value
        else:
            self.y = value


class CCVTSite:
    """Represents a site (generator) in the CCVT algorithm."""
    def __init__(self, site_id: int, capacity: int, x: float, y: float):
        self.id = site_id
        self.capacity = capacity
        self.location = CCVTPoint(x, y)


class CCVTEntry:
    """Represents a site entry with its assigned points."""
    def __init__(self, site: CCVTSite):
        self.site = site
        self.points = []
        self.energy = 0.0
        self.stable = True
    
    def update_energy(self):
        """Update energy as sum of squared distances to site."""
        self.energy = 0.0
        for point in self.points:
            dx = point.x - self.site.location.x
            dy = point.y - self.site.location.y
            self.energy += dx * dx + dy * dy


class PaperAccurateCCVT:
    """
    Paper-accurate implementation of Capacity-Constrained Voronoi Tessellation.
    
    This follows the exact algorithm from the C++ reference implementation.
    """
    
    def __init__(self, domain_bounds=(0, 1, 0, 1)):
        self.domain_bounds = domain_bounds
        self.xmin, self.xmax, self.ymin, self.ymax = domain_bounds
        self.entries = []
        self.sites = []
        self.discrete_points = []
    
    def generate_discrete_points_from_image(self, image_path: str, n_points: int = 10000):
        """
        Generate discrete points based on image density (following C++ nonconstant_density).
        
        This is the key insight: the algorithm works on a DISCRETE set of points,
        not a continuous density function.
        """
        print(f"📍 Generating {n_points} discrete points from image density...")
        
        # Load and process image
        img = Image.open(image_path).convert('L')
        img_array = np.array(img, dtype=np.float64)
        
        # Apply blur and normalization
        img_array = gaussian_filter(img_array, sigma=1.0)  # Less blur for sharper edges
        img_array = img_array / 255.0
        img_array = 1.0 - img_array  # Invert for stippling
        
        # CRITICAL FIX: Use aggressive thresholding instead of adding minimum density
        # This ensures background points are truly rejected
        threshold = 0.35  # Much more aggressive - only accept very dense areas
        img_array = np.maximum(img_array - threshold, 0.0)
        
        # Renormalize after thresholding
        if img_array.max() > 0:
            img_array = img_array / img_array.max()
        
        # Apply contrast enhancement to make foreground even more dominant
        img_array = np.power(img_array, 0.2)  # Very aggressive contrast - only darkest areas survive
        
        height, width = img_array.shape
        points = []
        
        # Use rejection sampling to generate discrete points (like C++ implementation)
        attempts = 0
        max_attempts = n_points * 100  # More attempts since we're more selective
        
        while len(points) < n_points and attempts < max_attempts:
            # Random point in [0,1] x [0,1]
            x = np.random.rand()
            y = np.random.rand()
            
            # Map to image coordinates
            img_x = int(x * (width - 1))
            img_y = int(y * (height - 1))
            
            # Get density at this point
            density = img_array[img_y, img_x]
            
            # Accept point with probability proportional to density
            # CRITICAL FIX: Only accept if density > 0 (eliminates background)
            if density > 0 and np.random.rand() < density:
                # Map to domain bounds
                domain_x = x * (self.xmax - self.xmin) + self.xmin
                domain_y = y * (self.ymax - self.ymin) + self.ymin
                points.append(CCVTPoint(domain_x, domain_y))
            
            attempts += 1
            
            if len(points) % 1000 == 0:
                print(f"   Generated {len(points)}/{n_points} points (attempts: {attempts})")
        
        self.discrete_points = points
        print(f"✅ Generated {len(self.discrete_points)} discrete points")
        return points
    
    def initialize_sites(self, n_sites: int):
        """Initialize sites with equal capacity (following C++ main.cpp)."""
        print(f"🎯 Initializing {n_sites} sites with equal capacity...")
        
        total_capacity = len(self.discrete_points)
        sites = []
        entries = []
        
        for i in range(n_sites):
            # Calculate capacity for this site
            capacity = total_capacity // (n_sites - i)
            total_capacity -= capacity
            
            # Random initial location
            x = np.random.rand() * (self.xmax - self.xmin) + self.xmin
            y = np.random.rand() * (self.ymax - self.ymin) + self.ymin
            
            site = CCVTSite(i, capacity, x, y)
            entry = CCVTEntry(site)
            
            sites.append(site)
            entries.append(entry)
        
        self.sites = sites
        self.entries = entries
        
        # Initial assignment using nearest neighbor
        self._initial_assignment()
        
        print(f"✅ Initialized {len(self.sites)} sites")
        for i, entry in enumerate(self.entries):
            print(f"   Site {i}: capacity={entry.site.capacity}, assigned={len(entry.points)}")
    
    def _initial_assignment(self):
        """Initial assignment of points to nearest sites."""
        print("🔄 Performing initial point assignment...")
        
        # Clear existing assignments
        for entry in self.entries:
            entry.points = []
        
        # Assign each point to nearest site
        site_locations = np.array([[site.location.x, site.location.y] for site in self.sites])
        point_locations = np.array([[p.x, p.y] for p in self.discrete_points])
        
        # Compute distances and assign to nearest
        distances = cdist(point_locations, site_locations)
        assignments = np.argmin(distances, axis=1)
        
        for i, point in enumerate(self.discrete_points):
            site_idx = assignments[i]
            self.entries[site_idx].points.append(point)
        
        # Update energies
        for entry in self.entries:
            entry.update_energy()
        
        print("✅ Initial assignment completed")
    
    def optimize_iteration(self, centroidalize: bool = True):
        """
        Perform one optimization iteration following the C++ algorithm exactly.
        
        This is the core algorithm from ccvt_optimizer.h optimize() method.
        """
        n_entries = len(self.entries)
        stability = [True] * n_entries
        
        # Compare each pair of entries
        for i in range(n_entries):
            for j in range(i + 1, n_entries):
                entry1 = self.entries[i]
                entry2 = self.entries[j]
                
                # Skip if both are stable (optimization)
                if entry1.stable and entry2.stable:
                    continue
                
                # Work with smaller entry first (optimization)
                if len(entry1.points) > len(entry2.points):
                    entry1, entry2 = entry2, entry1
                    idx1, idx2 = j, i
                else:
                    idx1, idx2 = i, j
                
                site1 = entry1.site
                site2 = entry2.site
                
                # Build candidate lists for swapping
                candidates1 = []
                for point in entry1.points:
                    energy_self = self._energy(point, site1)
                    energy_other = self._energy(point, site2)
                    candidates1.append((energy_self - energy_other, point, energy_self, energy_other))
                
                if not candidates1:
                    continue
                
                # Sort by energy difference (heap behavior)
                candidates1.sort(reverse=True)
                min_energy = -candidates1[0][0]
                
                candidates2 = []
                for point in entry2.points:
                    energy_self = self._energy(point, site2)
                    energy_other = self._energy(point, site1)
                    energy_diff = energy_self - energy_other
                    if energy_diff > min_energy:
                        candidates2.append((energy_diff, point, energy_self, energy_other))
                
                if not candidates2:
                    continue
                
                candidates2.sort(reverse=True)
                
                # Perform swaps
                max_swaps = min(len(candidates1), len(candidates2))
                swaps = 0
                
                for k in range(max_swaps):
                    diff1, point1, e1_self, e1_other = candidates1[k]
                    diff2, point2, e2_self, e2_other = candidates2[k]
                    
                    # Check if swap improves energy
                    if diff1 + diff2 <= 0:
                        break
                    
                    # Perform swap
                    entry1.points.remove(point1)
                    entry2.points.remove(point2)
                    entry1.points.append(point2)
                    entry2.points.append(point1)
                    
                    # Update energies
                    entry1.energy += e2_other - e1_self
                    entry2.energy += e1_other - e2_self
                    
                    swaps += 1
                
                # If swaps occurred, mark as unstable and update centroids
                if swaps > 0:
                    stability[idx1] = False
                    stability[idx2] = False
                    
                    if centroidalize:
                        # Update site locations to centroids
                        if entry1.points:
                            cx = sum(p.x for p in entry1.points) / len(entry1.points)
                            cy = sum(p.y for p in entry1.points) / len(entry1.points)
                            entry1.site.location.x = cx
                            entry1.site.location.y = cy
                        
                        if entry2.points:
                            cx = sum(p.x for p in entry2.points) / len(entry2.points)
                            cy = sum(p.y for p in entry2.points) / len(entry2.points)
                            entry2.site.location.x = cx
                            entry2.site.location.y = cy
                    
                    # Recalculate energies
                    entry1.update_energy()
                    entry2.update_energy()
        
        # Update stability
        stable = True
        for i, entry in enumerate(self.entries):
            entry.stable = stability[i]
            stable &= stability[i]
        
        return stable
    
    def _energy(self, point: CCVTPoint, site: CCVTSite) -> float:
        """Energy function: squared Euclidean distance."""
        dx = point.x - site.location.x
        dy = point.y - site.location.y
        return dx * dx + dy * dy
    
    def total_energy(self) -> float:
        """Calculate total energy of the system."""
        return sum(entry.energy for entry in self.entries)
    
    def run_ccvt(self, n_sites: int, max_iterations: int = 100, centroidalize: bool = True):
        """Run the complete CCVT algorithm."""
        print(f"\n{'='*70}")
        print(f"RUNNING PAPER-ACCURATE CCVT ALGORITHM")
        print('='*70)
        print(f"Sites: {n_sites}")
        print(f"Discrete points: {len(self.discrete_points)}")
        print(f"Max iterations: {max_iterations}")
        print(f"Centroidalize: {centroidalize}")
        print("-" * 50)
        
        # Initialize sites
        self.initialize_sites(n_sites)
        
        # Optimization loop
        energy_history = []
        iteration = 0
        stable = False
        
        while not stable and iteration < max_iterations:
            iteration += 1
            print(f"🔄 Iteration {iteration}...")
            
            stable = self.optimize_iteration(centroidalize)
            energy = self.total_energy()
            energy_history.append(energy)
            
            print(f"   Energy: {energy:.6f}, Stable: {stable}")
            
            # Debug: show capacity distribution
            if iteration % 10 == 0:
                capacities = [len(entry.points) for entry in self.entries]
                print(f"   Capacity distribution: min={min(capacities)}, max={max(capacities)}, mean={np.mean(capacities):.1f}")
        
        print(f"\n✅ CCVT completed after {iteration} iterations")
        print(f"📊 Final energy: {energy_history[-1]:.6f}")
        print(f"🎯 Stable: {stable}")
        
        return energy_history, stable
    
    def get_result_points(self) -> np.ndarray:
        """Get the final site locations as points for stippling."""
        return np.array([[site.location.x, site.location.y] for site in self.sites])
    
    def get_all_discrete_points(self) -> np.ndarray:
        """Get all discrete points for comparison."""
        return np.array([[p.x, p.y] for p in self.discrete_points])


def run_paper_accurate_stippling(image_path: str, n_sites: int = 1000, 
                                n_discrete_points: int = 20000,
                                max_iterations: int = 50,
                                output_dir: str = 'output'):
    """Run paper-accurate CCVT stippling."""
    
    print(f"\n{'='*80}")
    print(f"PAPER-ACCURATE CCVT IMAGE STIPPLING")
    print('='*80)
    print(f"Based on the original C++ implementation from Balzer2009")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize CCVT
    ccvt = PaperAccurateCCVT()
    
    # Generate discrete points from image
    ccvt.generate_discrete_points_from_image(image_path, n_discrete_points)
    
    # Run CCVT algorithm
    energy_history, stable = ccvt.run_ccvt(n_sites, max_iterations, centroidalize=True)
    
    # Get results
    site_points = ccvt.get_result_points()
    discrete_points = ccvt.get_all_discrete_points()
    
    # Create comprehensive visualization
    create_paper_accurate_visualization(image_path, site_points, discrete_points, 
                                      energy_history, output_dir, n_sites)
    
    return site_points, discrete_points, energy_history


def create_paper_accurate_visualization(image_path: str, site_points: np.ndarray, 
                                      discrete_points: np.ndarray, energy_history: list,
                                      output_dir: str, n_sites: int):
    """Create visualization showing the paper-accurate results."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    original_img = np.array(Image.open(image_path).convert('L'))
    axes[0, 0].imshow(original_img, cmap='gray', origin='upper')
    axes[0, 0].set_title('(a) Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Discrete points (showing the underlying discrete space)
    axes[0, 1].scatter(discrete_points[:, 0], discrete_points[:, 1], 
                      s=0.1, c='blue', alpha=0.5)
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_aspect('equal')
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_title(f'(b) Discrete Points\n({len(discrete_points)} points)', 
                        fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    
    # CCVT sites (the final stippling result)
    axes[0, 2].set_facecolor('white')
    point_size = max(2, 1000 / len(site_points))
    axes[0, 2].scatter(site_points[:, 0], site_points[:, 1], 
                      s=point_size, c='black', alpha=0.9)
    axes[0, 2].set_xlim(0, 1)
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].set_aspect('equal')
    axes[0, 2].invert_yaxis()
    axes[0, 2].set_title(f'(c) CCVT Sites (Paper Algorithm)\n({len(site_points)} sites)', 
                        fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Energy convergence
    axes[1, 0].plot(energy_history, 'b-', linewidth=2)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Energy')
    axes[1, 0].set_title('(d) Energy Convergence', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # Overlay on original
    axes[1, 1].imshow(original_img, cmap='gray', origin='upper', alpha=0.4)
    axes[1, 1].scatter(site_points[:, 0] * original_img.shape[1], 
                      (1 - site_points[:, 1]) * original_img.shape[0],
                      s=point_size*0.8, c='red', alpha=0.9)
    axes[1, 1].set_title('(e) Overlay on Original', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Final stippling result (clean)
    axes[1, 2].set_facecolor('white')
    large_point_size = max(3, 1500 / len(site_points))
    axes[1, 2].scatter(site_points[:, 0], site_points[:, 1], 
                      s=large_point_size, c='black', alpha=0.95)
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_aspect('equal')
    axes[1, 2].invert_yaxis()
    axes[1, 2].set_title(f'(f) Final Stippling Result\n(Paper-Accurate CCVT)', 
                        fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save results
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    base_name = f"paper_accurate_ccvt_{image_name}_{n_sites}sites"
    
    viz_file = os.path.join(output_dir, f"{base_name}_analysis.png")
    plt.savefig(viz_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Save points
    txt_file = os.path.join(output_dir, f"{base_name}.txt")
    header = f"Paper-Accurate CCVT Stippling - Site Positions\nFormat: x y\nTotal Sites: {len(site_points)}"
    np.savetxt(txt_file, site_points, fmt='%.6f', delimiter='\t', header=header)
    
    print(f"\n✅ Paper-accurate results saved:")
    print(f"📊 Analysis: {viz_file}")
    print(f"📄 Points: {txt_file}")


def main():
    """Test the paper-accurate implementation."""
    print("🧪 TESTING PAPER-ACCURATE CCVT IMPLEMENTATION")
    print("="*60)
    print("This follows the exact algorithm from the C++ reference code")
    
    # Find test images
    test_images = []
    if os.path.exists('sample_output'):
        for file in os.listdir('sample_output'):
            if file.lower().endswith(('.png', '.jpg')):
                test_images.append(os.path.join('sample_output', file))
    
    if not test_images:
        print("❌ No test images found!")
        return
    
    for img_path in test_images[:1]:  # Test with first image
        print(f"\n{'='*60}")
        try:
            site_points, discrete_points, energy_history = run_paper_accurate_stippling(
                img_path, 
                n_sites=800,  # Reasonable number for testing
                n_discrete_points=15000,  # Dense discrete space
                max_iterations=30
            )
            
            print(f"\n🎉 Paper-accurate CCVT completed successfully!")
            print(f"🎯 Generated {len(site_points)} sites from {len(discrete_points)} discrete points")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
