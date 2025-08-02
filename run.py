#!/usr/bin/env python3
"""
Main Runner for Point Distribution Algorithms
Supports YAML configuration files for easy parameter management
"""

import yaml
import argparse
import os
import sys
from pathlib import Path

# Import our working algorithms
sys.path.append(os.path.join(os.path.dirname(__file__), 'algorithms', 'capacity_constrained'))
from paper_accurate_stippling import run_paper_accurate_stippling
from lloyd_stippling import run_lloyd_stippling

def load_config(config_path):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"ERROR: Could not load config file '{config_path}': {e}")
        return None

def validate_config(config):
    """Validate configuration parameters"""
    required_fields = ['image', 'algorithm', 'points']
    
    for field in required_fields:
        if field not in config:
            print(f"ERROR: Missing required field '{field}' in config")
            return False
    
    # Check if image file exists
    if not os.path.exists(config['image']):
        print(f"ERROR: Image file '{config['image']}' not found")
        return False
    
    # Validate algorithm
    valid_algorithms = ['ccvt', 'lloyd']
    if config['algorithm'] not in valid_algorithms:
        print(f"ERROR: Unknown algorithm '{config['algorithm']}'. Available: {', '.join(valid_algorithms)}")
        return False
    
    # Validate points
    if not isinstance(config['points'], int) or config['points'] <= 0:
        print(f"ERROR: 'points' must be a positive integer, got {config['points']}")
        return False
    
    return True

def run_algorithm(config):
    """Run the specified algorithm with given configuration"""
    
    print("🚀 POINT DISTRIBUTION ALGORITHM RUNNER")
    print("=" * 50)
    print(f"📁 Image: {config['image']}")
    print(f"🎯 Algorithm: {config['algorithm']}")
    print(f"📊 Points: {config['points']}")
    
    # Get output directory (default: 'output')
    output_dir = config.get('output_dir', 'output')
    print(f"📂 Output: {output_dir}")
    
    if config['algorithm'] == 'ccvt':
        return run_paper_accurate_stippling(
            image_path=config['image'],
            num_points=config['points'],
            output_dir=output_dir
        )
    elif config['algorithm'] == 'lloyd':
        return run_lloyd_stippling(
            image_path=config['image'],
            num_points=config['points'],
            output_dir=output_dir
        )
    else:
        print(f"ERROR: Algorithm '{config['algorithm']}' not implemented")
        return None

def create_sample_config(output_path='sample_config.yaml'):
    """Create a sample configuration file"""
    
    sample_config = {
        'image': 'input/Plant.png',
        'algorithm': 'ccvt',
        'points': 20000,
        'output_dir': 'output',
        'description': 'CCVT stippling for Plant.png (Figure 7c)'
    }
    
    with open(output_path, 'w') as file:
        yaml.dump(sample_config, file, default_flow_style=False, indent=2)
    
    print(f"✅ Sample configuration created: {output_path}")
    return output_path

def main():
    """Command line interface"""
    
    parser = argparse.ArgumentParser(description='Run point distribution algorithms with YAML config')
    parser.add_argument('config', nargs='?', help='YAML configuration file path')
    parser.add_argument('--create-sample', action='store_true', 
                       help='Create a sample configuration file')
    parser.add_argument('--list-presets', action='store_true',
                       help='List available preset configurations')
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_config()
        return
    
    if args.list_presets:
        print("📋 Available Preset Configurations:")
        print("=" * 40)
        print("CCVT Algorithm:")
        print("  Plant (Figure 7c):    20,000 points")
        print("  Buildings (Figure 8):  3,000 points")
        print("  F_in:                 15,000 points")
        print("Lloyd Algorithm:")
        print("  Plant:                20,000 points")
        print("  Buildings:             3,000 points")
        print("  F_in:                 15,000 points")
        print("\\nAlgorithms: ccvt, lloyd")
        print("Use --create-sample to generate a template config file")
        return
    
    if not args.config:
        print("ERROR: No configuration file specified")
        print("\\nUsage:")
        print("  python run.py config.yaml              # Run with config")
        print("  python run.py --create-sample          # Create sample config")
        print("  python run.py --list-presets           # Show preset options")
        return
    
    # Load and validate configuration
    config = load_config(args.config)
    if config is None:
        return
    
    if not validate_config(config):
        return
    
    # Run the algorithm
    result = run_algorithm(config)
    
    if result is not None:
        print("\\n🎉 SUCCESS! Algorithm completed successfully.")
    else:
        print("\\n❌ FAILED: Algorithm execution failed.")

if __name__ == "__main__":
    main()
