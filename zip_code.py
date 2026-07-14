"""
Zip the source code, configs, and required files for Kaggle Dataset upload.
Excludes large files, datasets, caches, and checkpoints.
"""

import os
import zipfile
from pathlib import Path

# Paths to include
INCLUDES = ["src", "configs", "requirements.txt", "tests", "create_subset.py"]
OUTPUT_ZIP = "biohub_code.zip"

# Patterns/folders to exclude
EXCLUDE_PATTERNS = [
    "__pycache__",
    ".git",
    ".gemini",
    ".agents",
    "checkpoints",
    "scratch",
    "test_out",
    ".ipynb_checkpoints",
]

def should_exclude(path: Path) -> bool:
    for pattern in EXCLUDE_PATTERNS:
        if pattern in path.parts or any(pattern in part for part in path.parts):
            return True
    return False

def zip_project():
    root_dir = Path(".")
    zip_path = root_dir / OUTPUT_ZIP
    
    print(f"Creating zip archive at: {zip_path.resolve()}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for item in INCLUDES:
            item_path = root_dir / item
            if not item_path.exists():
                print(f"Warning: {item} does not exist, skipping.")
                continue
                
            if item_path.is_file():
                if not should_exclude(item_path):
                    zipf.write(item_path, item_path.relative_to(root_dir))
                    print(f"Added file: {item}")
            else:
                for file_path in item_path.rglob("*"):
                    if file_path.is_file() and not should_exclude(file_path):
                        archive_name = file_path.relative_to(root_dir)
                        zipf.write(file_path, archive_name)
                        
    print(f"Successfully created {OUTPUT_ZIP} ({zip_path.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    zip_project()
