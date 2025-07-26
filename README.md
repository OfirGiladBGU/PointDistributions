# Point Distribution Algorithms

A clean implementation of point distribution algorithms with Voronoi tessellation export.

## Features

- **Standard Lloyd Algorithm**: Classic Centroidal Voronoi Tessellation (CVT)
- **Capacity-Constrained Distribution Algorithm**: Blue noise point distribution with capacity constraints
- **Export Capabilities**: 
  - Point positions as TXT files
  - Voronoi diagrams as high-quality PNG files

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the tool:**
   ```bash
   python run_algorithms.py
   ```

## Usage

The tool provides a simple menu interface:

1. **Standard Lloyd Algorithm** - Run classic CVT algorithm
2. **Capacity-Constrained Distribution** - Run blue noise algorithm  
3. **Both Algorithms** - Run both with same parameters
4. **Exit**

### Parameters

- **Number of generators**: How many points to distribute
- **Number of iterations**: Algorithm iterations (default: 50)
- **Density function**: Choose from uniform, gaussian, multi-gaussian, or linear
- **Blue noise weight**: Balance between spatial quality and capacity uniformity (capacity-constrained only)

## Output Files

All results are saved to the `output/` directory:

- **TXT files**: Point coordinates (tab-separated)
- **PNG files**: Voronoi tessellation diagrams (300 DPI)

### File naming convention:
- `algorithm_density_points_voronoi.png` - Voronoi diagram
- `algorithm_density_points.txt` - Point coordinates

## Project Structure

```
PointDistributions/
├── algorithms/           # Core algorithm implementations
├── utils/               # Utility functions and density definitions
├── output/              # Generated results (git-ignored)
├── run_algorithms.py    # Main tool
└── requirements.txt     # Dependencies
```

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

## Dependencies

- numpy
- matplotlib  
- scipy
- scikit-learn (for blue noise analysis)

## License

MIT License