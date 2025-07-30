#!/usr/bin/env python3
"""
Project Directory Organizer

This script helps keep the project directory clean by organizing files
into appropriate subdirectories.
"""

import os
import shutil
import glob


def organize_project_directory():
    """Organize the project directory structure."""
    print("🧹 ORGANIZING PROJECT DIRECTORY")
    print("="*40)
    
    # Define directory structure
    directories = {
        'sample_output': 'Sample images and demonstration outputs',
        'output': 'Algorithm results and generated files',
        'algorithms': 'Algorithm implementations',
        'utils': 'Utility modules and helper functions',
        'tests': 'Test files and validation scripts',
        'examples': 'Example scripts and tutorials'
    }
    
    # Create directories if they don't exist
    for dir_name, description in directories.items():
        os.makedirs(dir_name, exist_ok=True)
        print(f"📁 {dir_name:<15} - {description}")
    
    # Move files to appropriate directories
    moves_made = 0
    
    # Move any stray image files to sample_output
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.gif']
    for ext in image_extensions:
        for file in glob.glob(ext):
            if not file.startswith('sample_output') and not file.startswith('output'):
                dest = os.path.join('sample_output', file)
                try:
                    shutil.move(file, dest)
                    print(f"  📄 Moved {file} → sample_output/")
                    moves_made += 1
                except Exception as e:
                    print(f"  ⚠️  Could not move {file}: {e}")
    
    # Move any demo or test files to sample_output
    demo_patterns = ['demo_*.py', 'test_*.png', 'sample_*.png']
    for pattern in demo_patterns:
        for file in glob.glob(pattern):
            if not file.startswith('sample_output'):
                dest = os.path.join('sample_output', file)
                try:
                    shutil.move(file, dest)
                    print(f"  📄 Moved {file} → sample_output/")
                    moves_made += 1
                except Exception as e:
                    print(f"  ⚠️  Could not move {file}: {e}")
    
    print(f"\n✅ Organization complete! Moved {moves_made} files.")
    
    # Show current directory structure
    print(f"\n📊 CURRENT PROJECT STRUCTURE:")
    print("="*40)
    
    main_files = [f for f in os.listdir('.') if os.path.isfile(f) and not f.startswith('.')]
    main_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and not d.startswith('.') and d != '__pycache__']
    
    # Show main Python files
    python_files = [f for f in main_files if f.endswith('.py')]
    if python_files:
        print("🐍 Main Python Scripts:")
        for file in sorted(python_files):
            print(f"   {file}")
    
    # Show main config files
    config_files = [f for f in main_files if f.endswith(('.md', '.txt', '.yml', '.yaml', '.json', '.toml'))]
    if config_files:
        print("\n📋 Configuration & Documentation:")
        for file in sorted(config_files):
            print(f"   {file}")
    
    # Show directories with file counts
    print("\n📁 Directories:")
    for dir_name in sorted(main_dirs):
        try:
            file_count = len([f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))])
            dir_count = len([d for d in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, d))])
            
            content_info = []
            if file_count > 0:
                content_info.append(f"{file_count} files")
            if dir_count > 0:
                content_info.append(f"{dir_count} subdirs")
            
            content_str = ", ".join(content_info) if content_info else "empty"
            print(f"   {dir_name:<15} ({content_str})")
        except PermissionError:
            print(f"   {dir_name:<15} (access denied)")
    
    print(f"\n🎯 TIPS FOR KEEPING PROJECT CLEAN:")
    print("="*40)
    print("• Use 'sample_output/' for test images and demos")
    print("• Use 'output/' for algorithm results")
    print("• Keep only main scripts in project root")
    print("• Run this organizer script when needed")
    print("• Add large files to .gitignore")


def clean_pycache():
    """Remove Python cache directories."""
    print("\n🗑️  CLEANING PYTHON CACHE")
    print("="*30)
    
    cache_dirs = []
    for root, dirs, files in os.walk('.'):
        if '__pycache__' in dirs:
            cache_dirs.append(os.path.join(root, '__pycache__'))
    
    if cache_dirs:
        for cache_dir in cache_dirs:
            try:
                shutil.rmtree(cache_dir)
                print(f"   Removed: {cache_dir}")
            except Exception as e:
                print(f"   Failed to remove {cache_dir}: {e}")
        print(f"✅ Cleaned {len(cache_dirs)} cache directories")
    else:
        print("✅ No cache directories found")


def show_directory_sizes():
    """Show directory sizes for cleanup guidance."""
    print(f"\n📏 DIRECTORY SIZES")
    print("="*25)
    
    for item in os.listdir('.'):
        if os.path.isdir(item) and not item.startswith('.'):
            try:
                size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(item)
                    for filename in filenames
                )
                
                # Convert to human readable
                if size > 1024*1024:
                    size_str = f"{size/(1024*1024):.1f} MB"
                elif size > 1024:
                    size_str = f"{size/1024:.1f} KB"
                else:
                    size_str = f"{size} B"
                
                print(f"   {item:<15} {size_str:>8}")
                
            except Exception as e:
                print(f"   {item:<15} (error: {e})")


if __name__ == "__main__":
    try:
        organize_project_directory()
        clean_pycache()
        show_directory_sizes()
        
        print(f"\n🎉 Project organization complete!")
        print("Run this script anytime to keep your project tidy.")
        
    except KeyboardInterrupt:
        print("\n❌ Organization cancelled.")
    except Exception as e:
        print(f"\n❌ Error during organization: {e}")
