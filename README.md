# Lloyd Algorithm for Capacity-Constrained Point Distributions

This project implements Lloyd's algorithm for Centroidal Voronoi Tessellations (CVT) and its capacity-constrained variant for Blue Noise Point Distributions. The implementation includes two focused algorithms:

1. **Standard Lloyd Algorithm** - Classic CVT for energy minimization
2. **Capacity-Constrained Lloyd Algorithm** - Blue noise characteristics with equal capacity constraints

Based on the research paper: *"Capacity-Constrained Point Distributions: A Variant of Lloyd's Method"*

## 🎯 **Clean Algorithm Separation**

The algorithms are clearly separated into focused, independent modules:

### **📁 Project Structure**

```
PointDistributions/
├── algorithms/                      # Algorithm implementations
│   ├── standard_lloyd/             # Standard Lloyd Algorithm (CVT)
│   │   ├── standard_lloyd.py       # CVT implementation  
│   │   ├── example_standard_lloyd.py # Algorithm-specific examples
│   │   └── __init__.py
│   ├── capacity_constrained/       # Capacity-Constrained Algorithm (Blue Noise + Equal Capacity)
│   │   ├── capacity_constrained_lloyd.py  # Blue Noise implementation
│   │   ├── example_capacity_constrained.py # Algorithm-specific examples
│   │   └── __init__.py
│   ├── README.md                   # Algorithm documentation and usage
│   └── __init__.py
├── utils/                          # Shared utilities
│   ├── base.py                     # Base classes and Voronoi utilities
│   ├── density_functions.py        # Example density functions
│   ├── visualization.py            # Plotting and visualization tools
│   └── __init__.py
├── tests/                          # Testing and comparison scripts
│   ├── compare_algorithms.py       # Algorithm comparison tool
│   ├── test_corrected_algorithm.py # Implementation verification
│   ├── benchmark.py               # Performance benchmarking
│   └── README.md                  # Testing documentation
├── examples/                       # General examples and demonstrations
│   ├── quick_demo.py              # Quick demonstration of both algorithms
│   ├── custom_densities.py        # Custom density function examples
│   └── README.md                  # Examples documentation
├── run_examples.py                # Main menu-driven interface
├── CORRECTED_ALGORITHM_ANALYSIS.md # Analysis of algorithm corrections
└── [configuration files...]       # Requirements, README, etc.
```

## 🔬 **Algorithms**

### **1. Standard Lloyd Algorithm** (`algorithms/standard_lloyd/`)

**Class:** `StandardLloydAlgorithm`

The classic Lloyd algorithm for Centroidal Voronoi Tessellations (CVT):

- **Purpose**: Minimizes energy through iterative centroid computation
- **Energy Function**: E = Σᵢ ∫_{Vᵢ} ρ(x)||x - pᵢ||² dx
- **Best For**: Energy-focused applications, smooth density functions
- **Features**:
  - Fast convergence for most cases
  - Support for arbitrary density functions
  - Optimal energy minimization
  - Well-distributed point sets

**Algorithm Steps:**
1. Compute Voronoi regions for current generators
2. Calculate density-weighted centroids of each region
3. Move generators to their region centroids
4. Repeat until convergence

### **2. Capacity-Constrained Lloyd Algorithm** (`algorithms/capacity_constrained/`)

**Class:** `CapacityConstrainedDistributionAlgorithm`

The improved algorithm from the paper *"Capacity-Constrained Point Distributions: A Variant of Lloyd's Method"*:

- **Purpose**: Enforces uniform capacity distribution across Voronoi cells
- **Target Capacity**: C_target = (∫_Ω ρ(x) dx) / n
- **Best For**: Applications requiring equal capacity per cell
- **Features**:
  - Adaptive weight mechanism for capacity balancing
  - Weighted Voronoi tessellations
  - Superior uniformity for equal-capacity applications
  - Capacity variance tracking and optimization

**Algorithm Steps:**
1. Compute current Voronoi regions and their capacities
2. Update weights based on capacity error: w_i = max(0.1, 1 + α × (C_target - C_i)/C_target)
3. Compute weighted Voronoi tessellation using adapted weights
4. Move generators to weighted centroids of new regions
5. Repeat until convergence

## 🚀 **Quick Start**

### **Installation**

```bash
# Clone the repository
git clone <repository-url>
cd PointDistributions

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### **Basic Usage**

#### **Algorithm-Specific Examples (Recommended)**

For focused testing of individual algorithms:

```bash
# Test Standard Lloyd Algorithm
cd algorithms/standard_lloyd
python example_standard_lloyd.py

# Test Capacity-Constrained Algorithm  
cd algorithms/capacity_constrained
python example_capacity_constrained.py
```

#### **Standard Lloyd Algorithm**

```python
from algorithms.standard_lloyd import StandardLloydAlgorithm
from utils import get_example_density_functions

# Setup
domain = (0, 1, 0, 1)  # (xmin, xmax, ymin, ymax)
algorithm = StandardLloydAlgorithm(domain)

# Generate initial points
n_generators = 20
initial_points = algorithm.generate_random_points(n_generators, seed=42)

# Get density function
density_funcs = get_example_density_functions()
density_func = density_funcs['gaussian']

# Run algorithm
final_points, energy_history = algorithm.run(
    initial_points,
    n_iterations=50,
    density_func=density_func
)

print(f"Final energy: {energy_history[-1]:.6f}")
```

#### **Capacity-Constrained Algorithm**

```python
from algorithms.capacity_constrained import CapacityConstrainedDistributionAlgorithm

# Setup
algorithm = CapacityConstrainedDistributionAlgorithm(domain)

# Run with capacity constraints and blue noise optimization
final_points, energy_history, capacity_variance_history, combined_energy_history = algorithm.run(
    initial_points,
    density_func,  # Required for capacity computation
    n_iterations=50,
    blue_noise_weight=0.5  # Balance spatial quality vs capacity uniformity
)

print(f"Final capacity variance: {capacity_variance_history[-1]:.6f}")

# Analyze capacity distribution and blue noise quality
analysis = algorithm.analyze_capacity_distribution(final_points, density_func)
print(f"Blue noise quality score: {analysis['blue_noise_quality']:.6f}")
print(f"Capacity uniformity score: {analysis['capacity_uniformity_score']:.6f}")
```

### **Running Examples**

#### **Algorithm-Specific Examples (Recommended)**
```bash
# Standard Lloyd Algorithm examples
cd algorithms/standard_lloyd
python example_standard_lloyd.py           # Full examples
python example_standard_lloyd.py --quick   # Quick test

# Capacity-Constrained Algorithm examples  
cd algorithms/capacity_constrained
python example_capacity_constrained.py           # Full examples
python example_capacity_constrained.py --quick   # Quick test
python example_capacity_constrained.py --compare # Compare with standard Lloyd
```

#### **General Examples and Demonstrations**
```bash
# From project root
python examples/quick_demo.py               # Quick demo of both algorithms
python examples/custom_densities.py        # Custom density function examples
```

#### **Tests and Comparisons**
```bash
# Algorithm comparison and testing
python tests/compare_algorithms.py         # Side-by-side algorithm comparison
python tests/test_corrected_algorithm.py   # Test corrected implementation
python tests/benchmark.py                  # Performance benchmarking
```

#### **Menu-Driven Interface**
```bash
python run_examples.py                     # Interactive menu system
```

## 📊 **Algorithm Comparison**

| **Aspect** | **Standard Lloyd** | **Capacity-Constrained** |
|------------|-------------------|-------------------------|
| **Primary Goal** | Energy minimization | Capacity uniformity |
| **Energy Performance** | ⭐⭐⭐⭐⭐ Optimal | ⭐⭐⭐⭐ Good |
| **Capacity Uniformity** | ⭐⭐⭐ Moderate | ⭐⭐⭐⭐⭐ Excellent |
| **Convergence Speed** | ⭐⭐⭐⭐⭐ Fast | ⭐⭐⭐⭐ Good |
| **Computational Cost** | ⭐⭐⭐⭐⭐ Low | ⭐⭐⭐⭐ Moderate |
| **Best Applications** | General CVT, energy-focused | Stippling, mesh generation, uniform sampling |

## 🎨 **Density Functions**

The implementation includes several example density functions:

- **Uniform**: Constant density across domain
- **Gaussian**: Single peak at domain center
- **Multi-Gaussian**: Multiple high-density regions (challenging case)
- **Linear**: Linear gradient density
- **Custom**: Build your own using `DensityFunctionBuilder`

## 📈 **Visualization Features**

- **Voronoi Diagrams**: Visualize tessellations with generators
- **Density Backgrounds**: Show density functions with overlaid points
- **Convergence Curves**: Track energy and capacity variance over iterations
- **Comparison Plots**: Side-by-side algorithm comparisons
- **Capacity Analysis**: Histograms and statistical analysis

## 🔧 **Key Parameters**

### **Standard Lloyd Parameters**
- `n_iterations`: Maximum iterations (default: 100)
- `tolerance`: Convergence tolerance (default: 1e-6)
- `density_func`: Optional density function
- `sample_density`: Integration sample points (default: 10,000)

### **Capacity-Constrained Parameters**
- `adaptation_rate`: Weight adaptation rate (0.1-0.5 recommended)
- `target_capacity`: Automatically computed as total_capacity / n_generators
- All standard parameters plus capacity-specific settings

## 📚 **Applications**

### **Standard Lloyd Algorithm**
- Basic point distribution problems
- Energy minimization applications
- Smooth density function cases
- General Voronoi tessellation needs

### **Capacity-Constrained Algorithm**
- **Computer Graphics**: Stippling, halftoning with density constraints
- **Computational Geometry**: Mesh generation with area constraints
- **Sampling**: Problems requiring uniform capacity distribution
- **Geographic Information Systems**: Facility location with equal service areas
- **Image Processing**: Pixel clustering with uniform regions

## 🧪 **Testing and Examples**

### **Algorithm-Specific Testing**
```bash
# Test individual algorithms in their own folders
cd algorithms/standard_lloyd && python example_standard_lloyd.py
cd algorithms/capacity_constrained && python example_capacity_constrained.py
```

### **Comprehensive Testing**
```bash
# From project root - test functionality and comparisons
python tests/test_corrected_algorithm.py   # Verify corrected implementation
python tests/compare_algorithms.py         # Algorithm comparison
python tests/benchmark.py                  # Performance analysis
```

### **General Examples**
```bash
# Educational and demonstration examples
python examples/quick_demo.py              # Quick overview
python examples/custom_densities.py        # Custom density functions
python run_examples.py                     # Interactive menu
```

## 📖 **Mathematical Background**

### **Energy Function**
Both algorithms minimize the CVT energy functional:
```
E = Σᵢ ∫_{Vᵢ} ρ(x)||x - pᵢ||² dx
```
Where:
- Vᵢ is the Voronoi region of generator pᵢ
- ρ(x) is the density function
- ||·|| is the Euclidean distance

### **Capacity Constraint**
The capacity-constrained algorithm enforces:
```
C_i ≈ C_target = (∫_Ω ρ(x) dx) / n
```
Where C_i is the capacity of region i and n is the number of generators.

## 🔄 **Algorithm Differences**

| **Feature** | **Standard Lloyd** | **Capacity-Constrained** |
|-------------|-------------------|-------------------------|
| **Voronoi Assignment** | Standard distance: d_ij | Weighted distance: d_ij / w_i |
| **Weight Update** | None | w_i = max(0.1, 1 + α × error_i) |
| **Optimization Target** | Energy only | Energy + capacity uniformity |
| **Convergence Metric** | Point movement | Point movement + capacity variance |

## 🏗️ **Development and Contribution**

### **Code Organization Benefits**
- **Clear Separation**: Easy to distinguish between algorithms
- **Modular Design**: Independent development and testing
- **Shared Utilities**: Common functionality in utils package
- **Extensible**: Easy to add new algorithms or density functions

### **Contributing**
- Add new density functions in `utils/density_functions.py`
- Implement new algorithms following the base class pattern
- Improve visualization capabilities in `utils/visualization.py`
- Add comprehensive tests for new features

## 📄 **References**

- **Paper**: "Capacity-Constrained Point Distributions: A Variant of Lloyd's Method"
- **Standard Lloyd**: Classic algorithm for Centroidal Voronoi Tessellations
- **Voronoi Diagrams**: Spatial partitioning based on proximity
- **CVT Theory**: Energy minimization through iterative centroidal updates

## 📋 **Dependencies**

```
numpy>=1.21.0
matplotlib>=3.5.0
scipy>=1.7.0
scikit-learn>=1.0.0
```

## 📝 **License**

This project is open source. Please cite appropriately if used in research.

---

## 🎯 **Summary**

This implementation provides:
1. **Clear algorithm separation** for better understanding
2. **Two distinct algorithms** for different use cases
3. **Comprehensive examples** and visualizations
4. **Modular design** for easy extension
5. **Well-documented code** with mathematical background

Choose **Standard Lloyd** for energy-focused applications and **Capacity-Constrained Lloyd** for uniformity-critical applications!
