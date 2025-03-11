import os
import argparse

def create_directory(directory_path):
    """Create directory if it doesn't exist."""
    os.makedirs(directory_path, exist_ok=True)
    print(f"Created directory: {directory_path}")

def create_file(file_path, content=""):
    """Create file with optional content."""
    with open(file_path, 'w') as f:
        f.write(content)
    print(f"Created file: {file_path}")

def setup_project_structure(base_dir):
    """Set up the project directory structure."""
    # Main directories
    directories = [
        os.path.join(base_dir, "data", "raw"),
        os.path.join(base_dir, "data", "processed"),
        os.path.join(base_dir, "notebooks"),
        os.path.join(base_dir, "src", "data"),
        os.path.join(base_dir, "src", "features"),
        os.path.join(base_dir, "src", "visualization"),
        os.path.join(base_dir, "src", "models"),
        os.path.join(base_dir, "tests"),
        os.path.join(base_dir, "web", "static", "css"),
        os.path.join(base_dir, "web", "static", "js"),
        os.path.join(base_dir, "web", "static", "images"),
        os.path.join(base_dir, "web", "templates"),
        os.path.join(base_dir, "docs"),
    ]
    
    for directory in directories:
        create_directory(directory)
    
    # Create __init__.py files in all src subdirectories
    init_files = [
        os.path.join(base_dir, "src", "__init__.py"),
        os.path.join(base_dir, "src", "data", "__init__.py"),
        os.path.join(base_dir, "src", "features", "__init__.py"),
        os.path.join(base_dir, "src", "visualization", "__init__.py"),
        os.path.join(base_dir, "src", "models", "__init__.py"),
        os.path.join(base_dir, "tests", "__init__.py"),
    ]
    
    for init_file in init_files:
        create_file(init_file)
    
    # Create placeholder files
    placeholder_files = [
        (os.path.join(base_dir, "src", "data", "preprocessing.py"), ""),
        (os.path.join(base_dir, "src", "data", "validation.py"), ""),
        (os.path.join(base_dir, "src", "features", "build_features.py"), ""),
        (os.path.join(base_dir, "src", "visualization", "visualize.py"), ""),
        (os.path.join(base_dir, "src", "models", "train_model.py"), ""),
        (os.path.join(base_dir, "tests", "test_preprocessing.py"), ""),
        (os.path.join(base_dir, "notebooks", "01_data_exploration.ipynb"), ""),
        (os.path.join(base_dir, "web", "app.py"), ""),
    ]
    
    for file_path, content in placeholder_files:
        create_file(file_path, content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up project directory structure")
    parser.add_argument("--base-dir", default=".", help="Base directory for the project")
    args = parser.parse_args()
    
    setup_project_structure(args.base_dir)
    print("Project structure setup complete!")