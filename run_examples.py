"""
Interactive menu system for demonstrations of the Lloyd algorithms.
This script provides a menu-driven interface to explore the two main algorithms:
1. Standard Lloyd Algorithm (CVT)
2. Capacity-Constrained Point Distribution Algorithm (Blue Noise + Equal Capacity)
"""

import sys
import os
import traceback

def print_menu():
    """Print the main menu options."""
    print("\n" + "="*70)
    print("LLOYD ALGORITHM DEMONSTRATION SUITE")
    print("="*70)
    print("1. Standard Lloyd Algorithm Examples")
    print("2. Capacity-Constrained Point Distribution Examples")
    print("3. Algorithm Comparison")
    print("4. Quick Demo - Both Algorithms")
    print("5. Custom Density Examples")
    print("6. Test Corrected Implementation")
    print("7. Generate Lloyd Results (Comprehensive)")
    print("8. Generate Capacity-Constrained Results (Comprehensive)")
    print("9. Generate Comparison Results (Comprehensive)")
    print("10. Run Algorithm with Voronoi Output ⭐")
    print("11. Run All Examples")
    print("0. Exit")
    print("="*70)

def run_script(script_name, description):
    """Run a Python script and handle errors."""
    print(f"\n{'='*70}")
    print(f"RUNNING: {description}")
    print('='*70)
    
    try:
        # Import and run the appropriate module
        if script_name == "standard_lloyd_example":
            print("Running Standard Lloyd Algorithm example...")
            os.system(f"cd algorithms\\standard_lloyd && {sys.executable} example_standard_lloyd.py")
        elif script_name == "capacity_constrained_example":
            print("Running Capacity-Constrained Point Distribution example...")
            os.system(f"cd algorithms\\capacity_constrained && {sys.executable} example_capacity_constrained_distribution.py")
        elif script_name == "compare_algorithms":
            sys.path.append('tests')
            import compare_algorithms
            compare_algorithms.compare_algorithms()
        elif script_name == "quick_demo":
            import examples.quick_demo
            examples.quick_demo.quick_demo()
        elif script_name == "custom_densities":
            import examples.custom_densities
            examples.custom_densities.demonstrate_custom_densities()
        elif script_name == "test_corrected":
            sys.path.append('tests')
            import test_corrected_algorithm
            test_corrected_algorithm.test_corrected_algorithm()
        elif script_name == "generate_lloyd_results":
            import generate_lloyd_results
            generate_lloyd_results.generate_lloyd_results()
        elif script_name == "generate_capacity_results":
            import generate_capacity_results
            generate_capacity_results.generate_capacity_constrained_results()
        elif script_name == "generate_comparison_results":
            import generate_comparison_results
            generate_comparison_results.generate_comparison_results()
        elif script_name == "run_algorithm_with_voronoi":
            import run_algorithm_with_voronoi
            run_algorithm_with_voronoi.interactive_menu()
        else:
            print(f"Unknown script: {script_name}")
            return False
            
        print(f"✓ Successfully completed: {description}")
        return True
        
    except Exception as e:
        print(f"✗ Error running {description}: {str(e)}")
        traceback.print_exc()
        return False
    
    return True

def main():
    """Main menu interface."""
    print("Lloyd Algorithm Implementation - Clean Structure")
    print("Two focused algorithms: Standard Lloyd and Capacity-Constrained Point Distribution")
    
    while True:
        print_menu()
        
        try:
            choice = input("\nEnter your choice (0-11): ").strip()
            
            if choice == "0":
                print("\nGoodbye!")
                break
                
            elif choice == "1":
                run_script("standard_lloyd_example", "Standard Lloyd Algorithm Examples")
                
            elif choice == "2":
                run_script("capacity_constrained_example", "Capacity-Constrained Point Distribution Examples")
                
            elif choice == "3":
                run_script("compare_algorithms", "Algorithm Comparison")
                
            elif choice == "4":
                run_script("quick_demo", "Quick Demo - Both Algorithms")
                
            elif choice == "5":
                run_script("custom_densities", "Custom Density Examples")
                
            elif choice == "6":
                run_script("test_corrected", "Test Corrected Implementation")
                
            elif choice == "7":
                run_script("generate_lloyd_results", "Generate Lloyd Results (Comprehensive)")
                
            elif choice == "8":
                run_script("generate_capacity_results", "Generate Capacity-Constrained Results (Comprehensive)")
                
            elif choice == "9":
                run_script("generate_comparison_results", "Generate Comparison Results (Comprehensive)")
                
            elif choice == "10":
                run_script("run_algorithm_with_voronoi", "Run Algorithm with Voronoi Output")
                
            elif choice == "11":
                print("\n" + "="*70)
                print("RUNNING ALL EXAMPLES")
                print("="*70)
                print("This will run all available examples...")
                print("Note: This may take several minutes and generate multiple plots.")
                
                try:
                    confirm = input("Continue? (y/N): ").strip().lower()
                except EOFError:
                    print("\nCancelled due to end of input.")
                    confirm = "n"
                    
                if confirm == 'y' or confirm == 'yes':
                    scripts = [
                        ("standard_lloyd_example", "Standard Lloyd Algorithm Examples"),
                        ("capacity_constrained_example", "Capacity-Constrained Point Distribution Examples"),
                        ("compare_algorithms", "Algorithm Comparison"),
                        ("quick_demo", "Quick Demo"),
                        ("custom_densities", "Custom Density Examples"),
                        ("generate_lloyd_results", "Generate Lloyd Results"),
                        ("generate_capacity_results", "Generate Capacity-Constrained Results"),
                        ("generate_comparison_results", "Generate Comparison Results")
                    ]
                    
                    success_count = 0
                    for script, desc in scripts:
                        if run_script(script, desc):
                            success_count += 1
                        try:
                            input("\nPress Enter to continue to next example...")
                        except EOFError:
                            print("\nContinuing automatically...")
                            pass
                    
                    print(f"\n{'='*70}")
                    print(f"SUMMARY: {success_count}/{len(scripts)} examples completed successfully")
                    print("="*70)
                else:
                    print("Cancelled.")
                    
            else:
                print("Invalid choice. Please enter a number between 0 and 10.")
                
        except EOFError:
            print("\n\nExiting due to end of input...")
            break
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Returning to menu...")

def quick_info():
    """Print quick information about the implementation."""
    info = """
LLOYD ALGORITHM IMPLEMENTATION - CLEAN STRUCTURE
===============================================

This implementation features two focused algorithms:

📁 PROJECT STRUCTURE:
├── algorithms/
│   ├── standard_lloyd/                          # Standard Lloyd Algorithm (CVT)
│   │   ├── standard_lloyd.py                    # Implementation
│   │   ├── example_standard_lloyd.py            # Algorithm-specific examples
│   │   └── __init__.py
│   ├── capacity_constrained/                    # Capacity-Constrained Point Distribution
│   │   ├── capacity_constrained_distribution.py # Implementation (Blue Noise + Equal Capacity)
│   │   ├── example_capacity_constrained_distribution.py # Algorithm-specific examples
│   │   └── __init__.py
│   └── __init__.py
├── utils/                                       # Shared utilities
│   ├── base.py                                  # Base classes and Voronoi utils
│   ├── density_functions.py                     # Example density functions
│   ├── visualization.py                         # Plotting utilities
│   └── __init__.py
├── tests/                                       # Testing and comparison
│   ├── compare_algorithms.py                    # Algorithm comparison
│   ├── test_corrected_algorithm.py              # Implementation verification
│   └── benchmark.py                             # Performance analysis
├── examples/                                    # General examples
│   ├── quick_demo.py                            # Quick demonstration
│   └── custom_densities.py                      # Custom density examples
├── output/                                      # Generated results (git ignored)
├── generate_lloyd_results.py                    # Comprehensive Lloyd results
├── generate_capacity_results.py                 # Comprehensive Capacity-Constrained results
├── generate_comparison_results.py               # Comprehensive algorithm comparison
└── run_examples.py                              # This menu interface

🎯 TWO MAIN ALGORITHMS:

1. STANDARD LLOYD ALGORITHM
   • Classic CVT algorithm for energy minimization
   • Fast convergence, optimal for energy-focused applications
   • Located in: algorithms/standard_lloyd/

2. CAPACITY-CONSTRAINED POINT DISTRIBUTION ALGORITHM
   • Paper implementation: "Capacity-Constrained Point Distributions"
   • Blue noise characteristics + equal capacity constraints
   • Fundamentally different approach from traditional Lloyd's method
   • Located in: algorithms/capacity_constrained/

🚀 USAGE RECOMMENDATIONS:
• Algorithm-specific examples: cd algorithms/[algorithm]/ && python example_[algorithm].py
• Quick overview: python examples/quick_demo.py
• Algorithm comparison: python tests/compare_algorithms.py
• Comprehensive results: python generate_*_results.py

📊 KEY BENEFITS:
• Clean separation of two focused algorithms
• No legacy code clutter
• Algorithm-specific examples in their own folders
• Clear distinction between Standard Lloyd and Capacity-Constrained approaches
• Comprehensive result generation scripts
• All outputs saved to git-ignored output/ directory
• Accurate naming reflecting the fundamental algorithmic differences
• Easier to understand, modify, and extend

For detailed documentation, see README.md
"""
    print(info)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--info":
        quick_info()
    else:
        main()
