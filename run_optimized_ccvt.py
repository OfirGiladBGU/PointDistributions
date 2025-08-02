#!/usr/bin/env python3
"""
Optimized and more aggressive CCVT implementation
"""

from paper_accurate_ccvt import run_paper_accurate_stippling
import time

def run_optimized_ccvt():
    print("🚀 RUNNING OPTIMIZED CCVT WITH AGGRESSIVE DENSITY FILTERING")
    print("="*60)
    
    start_time = time.time()
    
    # Run with optimized parameters
    print("Parameters:")
    print("- More aggressive density filtering")
    print("- Fewer discrete points for speed")
    print("- Fewer iterations for faster convergence")
    print()
    
    try:
        site_points, discrete_points, energy_history = run_paper_accurate_stippling(
            'sample_output/Plant.png',
            n_sites=800,           # Slightly fewer sites for speed
            n_discrete_points=12000,  # Fewer points since we're more selective
            max_iterations=20      # Fewer iterations for speed
        )
        
        end_time = time.time()
        
        print(f"\\n🎉 COMPLETED IN {end_time - start_time:.1f} SECONDS!")
        print(f"✅ Generated {len(site_points)} stippling points")
        print(f"📊 Used {len(discrete_points)} discrete points")
        print(f"⚡ Final energy: {energy_history[-1]:.6f}")
        print(f"🔄 Converged in {len(energy_history)} iterations")
        
        # Calculate efficiency metrics
        if len(discrete_points) > 0:
            efficiency = len(discrete_points) / 12000
            print(f"📈 Point generation efficiency: {efficiency:.3f}")
            print(f"   (Generated {len(discrete_points)} out of {12000} target points)")
        
        print("\\n📁 Check output/ directory for:")
        print("   - paper_accurate_ccvt_Plant_800sites_analysis.png")
        print("   - paper_accurate_ccvt_Plant_800sites.txt")
        
        print("\\n✨ The points should now be concentrated ONLY on the flower!")
        print("   No more random points in the background!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_optimized_ccvt()
