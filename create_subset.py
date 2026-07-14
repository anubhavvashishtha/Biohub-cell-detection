import argparse
import random
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Create a random subset of the Biohub training data.")
    parser.add_argument("--input_dir", type=str, default="/kaggle/input/competitions/biohub-cell-tracking-during-development/train", help="Path to the original train directory (containing .zarr folders)")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/train_subset", help="Path to the output subset directory")
    parser.add_argument("--num_samples", type=int, default=70, help="Number of samples to extract")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for selection")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        print("Please point --input_dir to your actual Kaggle dataset directory.")
        return
        
    print(f"Scanning '{input_dir}' for .zarr directories...")
    # Find all .zarr directories
    zarr_paths = [p for p in input_dir.rglob("*.zarr") if p.is_dir()]
    
    if not zarr_paths:
        print(f"No .zarr directories found in {input_dir}.")
        return
        
    print(f"Found {len(zarr_paths)} .zarr directories.")
    
    # Shuffle and select
    random.seed(args.seed)
    num_to_select = min(args.num_samples, len(zarr_paths))
    selected = random.sample(zarr_paths, num_to_select)
    
    print(f"Randomly selected {len(selected)} samples. Copying to '{output_dir}'...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, zarr_path in enumerate(selected):
        # We try to maintain the relative folder structure 
        # (in case there are subfolders before the .zarr)
        try:
            rel_path = zarr_path.relative_to(input_dir)
        except ValueError:
            rel_path = Path(zarr_path.name)
            
        dest_path = output_dir / rel_path
        
        print(f"[{i+1}/{len(selected)}] Copying {zarr_path.name}...")
        
        if dest_path.exists():
            print(f"  -> {dest_path} already exists. Skipping.")
        else:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(zarr_path, dest_path, dirs_exist_ok=True)
            
    print(f"\nSuccess! Created a subset with {len(selected)} samples in '{output_dir}'.")
    print(f"You can now point your config.data.data_root to '{output_dir}' to test locally.")

if __name__ == "__main__":
    main()
