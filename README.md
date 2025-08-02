# Point Distribution Algorithms

## Source:

Based on: [ccvt](https://github.com/Atrix256/ccvt)

## Description

A clean implementation of stippling algorithms for artistic point distribution visualization, featuring both **CCVT (Capacity-Constrained Voronoi Tessellation)** and **Lloyd** algorithms with optimized visualization output.

## 🎯 Overview

This project implements two high-quality point distribution algorithms:
- **CCVT**: Capacity-constrained algorithm from **Balzer2009: Capacity-constrained Point Distributions**
- **Lloyd**: Standard Lloyd relaxation algorithm for comparison

Both algorithms produce identical output formats with clean visualizations and comprehensive coordinate data.

## 🚀 Quick Start

### Using Preset Configurations

```bash
# CCVT algorithm examples
python run.py configs/ccvt_plant_config.yaml      # Plant image (20,000 points)
python run.py configs/ccvt_buildings_config.yaml  # Buildings image (3,000 points)
python run.py configs/ccvt_f_in_config.yaml       # F_in image (15,000 points)

# Lloyd algorithm examples  
python run.py configs/lloyd_plant_config.yaml     # Plant image (20,000 points)
python run.py configs/lloyd_buildings_config.yaml # Buildings image (3,000 points)
python run.py configs/lloyd_f_in_config.yaml      # F_in image (15,000 points)

# Quick demo
python demo_stippling.py
```

### Creating Custom Configurations

```bash
# Generate template
python run.py --create-sample

# Edit sample_config.yaml with your settings
python run.py sample_config.yaml
```

## 📁 Project Structure

```
PointDistributions/
├── run.py                          # Main runner script (YAML config)
├── demo_stippling.py               # Quick demonstration script
├── input/                          # Input images
│   ├── Plant.png                   # Reference plant image
│   ├── Buildings.png               # Reference buildings image
│   ├── F_in.png                    # Sample input image
│   └── S_in.png                    # Additional sample image
├── output/                         # Generated results
│   ├── *_ccvt_clean_stippling.png     # CCVT: Pure black dots on white
│   ├── *_ccvt_points_visualization.png # CCVT: Yellow dots over image
│   ├── *_ccvt_centers_visualization.png # CCVT: Voronoi centers (red/cyan)
│   ├── *_lloyd_clean_stippling.png    # Lloyd: Pure black dots on white
│   ├── *_lloyd_points_visualization.png # Lloyd: Yellow dots over image  
│   ├── *_lloyd_centers_visualization.png # Lloyd: Voronoi centers (blue/green)
│   ├── *_ccvt_points.txt              # CCVT point coordinates
│   ├── *_lloyd_points.txt             # Lloyd point coordinates
│   ├── *_ccvt_centroids.txt           # CCVT centroid coordinates
│   ├── *_lloyd_centroids.txt          # Lloyd centroid coordinates
│   ├── *_density.npy                  # CCVT cached density maps
│   ├── *_lloyd_density.npy            # Lloyd cached density maps
│   └── *_metadata.txt                 # Generation metadata
├── configs/                        # YAML configuration files
│   ├── ccvt_plant_config.yaml      # CCVT: Plant preset (20,000 points)
│   ├── ccvt_buildings_config.yaml  # CCVT: Buildings preset (3,000 points)
│   ├── ccvt_f_in_config.yaml       # CCVT: F_in preset (15,000 points)
│   ├── lloyd_plant_config.yaml     # Lloyd: Plant preset (20,000 points)
│   ├── lloyd_buildings_config.yaml # Lloyd: Buildings preset (3,000 points)
│   ├── lloyd_f_in_config.yaml      # Lloyd: F_in preset (15,000 points)
│   └── sample_config.yaml          # Template configuration
├── algorithms/
│   ├── capacity_constrained/
│   │   ├── paper_accurate_stippling.py # CCVT algorithm implementation
│   │   └── lloyd_stippling.py          # Lloyd algorithm implementation
│   └── standard_lloyd/
│       └── standard_lloyd.py           # Core Lloyd algorithm
└── Expected/                       # Reference results from paper
    ├── BuildingRes.png
    └── PlantRes.png
```

## ⚙️ Configuration Format

YAML configuration files control all algorithm parameters:

```yaml
image: "input/Plant.png"           # Input image path
algorithm: "ccvt"                  # Algorithm: "ccvt" or "lloyd"
points: 20000                      # Number of stippling points
output_dir: "output"               # Output directory
description: "Plant stippling"     # Optional description
```

## 🎨 Algorithm Features

### CCVT (Capacity-Constrained Voronoi Tessellation)
- **Paper-Accurate**: Based on Balzer2009 research
- **Rejection Sampling**: Density-based point placement with high quality
- **Aggressive Processing**: Precise density extraction and contrast enhancement
- **Unique Distribution**: Creates distinctive point patterns

### Lloyd Algorithm  
- **Centroidal Voronoi**: Classic Lloyd relaxation for smooth distributions
- **Energy Minimization**: Iterative optimization for even spacing
- **Fast Convergence**: Efficient algorithm with predictable results
- **Comparative Baseline**: Standard algorithm for comparison

### Unified Output Quality
- **Consistent Formats**: Both algorithms produce identical file types
- **Color-Coded Visualizations**: Algorithm-specific colors for easy distinction
- **High Resolution**: 300 DPI output for publication quality
- **Coordinate Export**: Normalized coordinates for further processing

## 📊 Performance

| Algorithm | Image | Points | Generation Time | Characteristics |
|-----------|-------|--------|----------------|----------------|
| **CCVT** | Plant.png | 20,000 | ~6 seconds | High-quality distribution, 64% acceptance |
| **CCVT** | Buildings.png | 3,000 | ~2 seconds | Dense detail areas, 78% acceptance |
| **CCVT** | F_in.png | 15,000 | ~6 seconds | Complex patterns, 64% acceptance |
| **Lloyd** | Plant.png | 20,000 | ~18 seconds | Smooth convergence, 3 iterations |
| **Lloyd** | Buildings.png | 3,000 | ~8 seconds | Fast convergence, even distribution |
| **Lloyd** | F_in.png | 15,000 | ~18 seconds | Consistent results, energy optimization |

## � Command Reference

```bash
# List available presets
python run.py --list-presets

# Create sample configuration  
python run.py --create-sample

# Run with configuration
python run.py config.yaml

# Quick demonstration
python demo_stippling.py
```

## 📝 Output Files

Each algorithm run generates a complete set of outputs with algorithm-specific naming:

### CCVT Algorithm Outputs:
1. **`*_ccvt_clean_stippling.png`** - Pure black dots on white background
2. **`*_ccvt_points_visualization.png`** - Yellow dots over original image  
3. **`*_ccvt_centers_visualization.png`** - Voronoi centers (red dots, cyan cells)
4. **`*_ccvt_points.txt`** - Point coordinates (x, y) in [0,1] range
5. **`*_ccvt_centroids.txt`** - Centroid coordinates (same as points)
6. **`*_ccvt_metadata.txt`** - Generation metadata and parameters
7. **`*_density.npy`** - CCVT cached density map for reuse

### Lloyd Algorithm Outputs:
1. **`*_lloyd_clean_stippling.png`** - Pure black dots on white background
2. **`*_lloyd_points_visualization.png`** - Yellow dots over original image
3. **`*_lloyd_centers_visualization.png`** - Voronoi centers (blue dots, green cells)
4. **`*_lloyd_points.txt`** - Point coordinates (x, y) in [0,1] range
5. **`*_lloyd_centroids.txt`** - Centroid coordinates (same as points)
6. **`*_lloyd_metadata.txt`** - Generation metadata and parameters
7. **`*_lloyd_density.npy`** - Lloyd cached density map for reuse

## 🎯 Algorithm Comparison

### CCVT Algorithm
Based on **Balzer2009: Capacity-constrained Point Distributions**, this implementation reproduces paper-accurate results:
- **Figure 7c**: Plant stippling with 20,000 points
- **Figure 8**: Buildings detail with 3,000 points

**Key Parameters:**
- Gaussian blur (σ=1.0)
- Aggressive threshold (0.35) 
- Power contrast enhancement (γ=0.2)
- Rejection sampling for point placement

### Lloyd Algorithm
Standard Centroidal Voronoi Tessellation for comparison:
- Energy-based optimization
- Smooth, even distribution
- Fast convergence (typically 3-5 iterations)
- Predictable, consistent results

## 🚀 Installation

```bash
# Clone repository  
git clone https://github.com/OfirGiladBGU/PointDistributions.git
cd PointDistributions

# Install dependencies
pip install -r requirements.txt

# Test with sample configurations
python run.py configs/ccvt_f_in_config.yaml
python run.py configs/lloyd_f_in_config.yaml
```

## 📚 Dependencies

- `numpy` - Numerical computations
- `matplotlib` - Visualization and plotting
- `pillow` - Image processing  
- `scipy` - Gaussian filtering
- `pyyaml` - Configuration file parsing

## 🎨 Usage Examples

## 🎨 Usage Examples

### Basic Algorithm Comparison
```bash
# Compare both algorithms on the same image
python run.py configs/ccvt_plant_config.yaml
python run.py configs/lloyd_plant_config.yaml
```

### Custom Images
```yaml
# my_config.yaml
image: "input/my_image.png"
algorithm: "ccvt"                   # or "lloyd"
points: 5000
output_dir: "output"
description: "Custom stippling"
```

```bash
python run.py my_config.yaml
```

### Batch Processing
```bash
# Process with both algorithms
python run.py configs/ccvt_f_in_config.yaml
python run.py configs/lloyd_f_in_config.yaml

# Compare results in output/ directory:
# F_in_ccvt_points_visualization.png vs F_in_lloyd_points_visualization.png
```

This system provides a clean, efficient workflow for generating high-quality stippling results with dual algorithm support and comprehensive output options for easy comparison.

## 📚 Dependencies

- **numpy** - Numerical computations and array operations
- **matplotlib** - Visualization and plotting for all output images
- **pillow** - Image processing and loading
- **scipy** - Scientific computing (Gaussian filtering, Voronoi diagrams)
- **pyyaml** - YAML configuration file parsing

## 🏆 Key Features

- **Dual Algorithm Support**: Compare CCVT and Lloyd algorithms side-by-side
- **Unified Output Format**: Consistent file naming and visualization styles
- **Color-Coded Results**: Visual distinction between algorithm outputs
- **High-Quality Visualizations**: 300 DPI publication-ready images
- **Clean Configuration**: Simple YAML-based parameter control
- **Comprehensive Output**: Points, visualizations, and metadata for each run

## 📄 License

MIT License