# Point Distribution Algorithms

A comprehensive implementation of point distribution algorithms with Voronoi tessellation export and **image stippling capabilities**.

## Features

- **Standard Lloyd Algorithm**: Classic Centroidal Voronoi Tessellation (CVT)
- **Capacity-Constrained Distribution Algorithm**: Blue noise point distribution with capacity constraints
- **Exact Paper Algorithm**: Direct implementation of the heap-based algorithm from research paper
- **🎨 Image Stippling**: Convert images to artistic point distributions (NEW!)
- **Export Capabilities**: 
  - Point positions as TXT files
  - Voronoi diagrams as high-quality PNG files
  - Stippling visualizations for artistic effects

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the tool:**
   ```bash
   python run_algorithms.py
   ```

3. **🆕 For image stippling:**
   ```bash
   python image_stippling.py
   ```

## 🎨 Image Stippling

Transform any image into artistic stippling effects, similar to the results shown in the research paper!

**What is Image Stippling?**
- **Dark areas** receive more points (high density)
- **Light areas** receive fewer points (low density)  
- Overall visual structure is preserved through intelligent point placement
- Results can be used for artistic effects, printing, or further processing

**Key Features:**
- **Supported formats**: PNG, JPG, JPEG, BMP, TIFF, GIF
- **Multiple algorithms**: Lloyd, Capacity-Constrained, Optimized Paper
- **Customizable parameters**: Point density, blur, color inversion
- **High-quality output**: Both coordinates and visual results

**Quick Start:**
```bash
# Method 1: From main interface
python run_algorithms.py
# Choose option 6: Image Stippling

# Method 2: Direct stippling tool
python image_stippling.py

# Method 3: Create test images first
python create_sample_images.py
python image_stippling.py
```

**Algorithm Recommendations:**
- **Standard Lloyd**: Even distribution, smooth results
- **Capacity-Constrained**: Blue noise effects, artistic variation  
- **Optimized Paper**: Best balance of quality and speed (recommended)

**Parameter Guidelines:**
- **Points**: 2,000-5,000 (simple), 5,000-15,000 (detailed), 15,000+ (very detailed)
- **Iterations**: 30-50 (balanced), 50-100 (higher quality)
- **Blur sigma**: 0 (no smoothing), 0.5-1.0 (light), 1.0-3.0 (heavy smoothing)

## Usage

The main tool provides a menu interface:

1. **Standard Lloyd Algorithm** - Classic CVT algorithm
2. **Capacity-Constrained Distribution** - Blue noise algorithm  
3. **Exact Paper Algorithm (Optimized)** - Fast heap-based algorithm
4. **Original Paper Algorithm (Slow)** - Research comparison version
5. **All Fast Algorithms** - Run algorithms 1, 2, and 3 with same parameters
6. **🆕 Image Stippling** - Create artistic stippling from images
0. **Exit**

### Parameters

- **Number of generators**: How many points to distribute
- **Number of iterations**: Algorithm iterations (default: 50)
- **Density function**: Choose from uniform, gaussian, multi-gaussian, or linear
- **Blue noise weight**: Balance between spatial quality and capacity uniformity (capacity-constrained only)

## Output Files

Results are automatically organized into directories:

- **`output/`**: Algorithm results (TXT coordinates + PNG Voronoi diagrams)
- **`sample_output/`**: Sample images, demos, and test results

**File Types:**
- **TXT files**: Point coordinates (tab-separated, ready for import)
- **PNG files**: High-quality visualizations (300 DPI)
- **Stippling files**: Side-by-side original and stippling comparison

**File Naming:**
- `algorithm_density_points.txt` - Point coordinates
- `algorithm_density_points_voronoi.png` - Voronoi diagram  
- `stippling_algorithm_image_points_stippling.png` - Stippling visualization

## Project Structure

```
PointDistributions/
├── 🐍 Main Scripts
│   ├── run_algorithms.py          # Main interface for all algorithms
│   ├── image_stippling.py          # Image stippling interface  
│   ├── create_sample_images.py     # Generate test images
│   └── organize_project.py         # Keep project tidy
│
├── 📁 Core Implementation
│   ├── algorithms/                 # Algorithm implementations
│   ├── utils/                      # Utility modules & image processing
│   └── tests/                      # Test files
│
├── 📁 Generated Content
│   ├── output/                     # Algorithm results (TXT + PNG)
│   └── sample_output/              # Test images and demos
│
└── 📋 Documentation & Config
    ├── README.md                   # This file
    └── requirements.txt            # Python dependencies
```

**Directory Organization:**
- **Clean project root**: Only main scripts visible
- **Automatic organization**: Run `python organize_project.py` to tidy up
- **Separated concerns**: Code vs. data vs. results
- **Git-friendly**: Proper separation of generated files

## Algorithms

### Standard Lloyd Algorithm
- Classic Centroidal Voronoi Tessellation
- Minimizes energy through centroidal movement
- Ideal for energy-focused applications

### Capacity-Constrained Distribution Algorithm  
- Blue noise point distribution
- Balances spatial regularity with capacity constraints
- Configurable blue noise weight parameter
- Superior distribution quality for rendering applications

### Exact Paper Algorithm (Optimized)
- Direct implementation of heap-based algorithm from research paper
- Fast performance with 5-10x speed improvement
- Best balance of quality and speed
- Recommended for most applications

### Original Paper Algorithm (Slow)
- Exact research implementation for comparison
- Very slow but matches paper exactly
- Use only for research validation

## Tips & Best Practices

**For Image Stippling:**
- Use high contrast images for best results
- Start with 5,000 points for testing, increase for detail
- Try different algorithms for different artistic effects
- Adjust blur sigma to control detail level

**For General Point Distribution:**
- Use Standard Lloyd for energy minimization
- Use Capacity-Constrained for blue noise effects
- Use Optimized Paper for best overall quality

**Project Maintenance:**
- Run `python organize_project.py` to keep directories clean
- Sample images are stored in `sample_output/`
- Results automatically go to `output/`

## Dependencies

- **numpy** - Numerical computations
- **matplotlib** - Plotting and visualization  
- **scipy** - Scientific computing (Voronoi, spatial operations)
- **pillow** - Image processing for stippling
- **scikit-learn** - Blue noise analysis

## License

MIT License