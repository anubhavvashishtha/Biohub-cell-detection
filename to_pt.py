import os
import zipfile
from pathlib import Path

def reconstruct_pt(folder_name):
    # Notice: Removed the leading slash so it looks in the local 'src' directory
    folder_path = Path(f"src/{folder_name}")
    target_path = Path(f"src/{folder_name}.pt")
    
    # 1. Check if the folder exists
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"[-] Folder '{folder_path}' does not exist. Skipping.")
        return

    # 2. If target .pt already exists, delete it
    if target_path.exists():
        print(f"[*] '{target_path.name}' already exists. Deleting it...")
        target_path.unlink()

    print(f"[+] Found unzipped checkpoint folder at '{folder_path}'")
    
    # PyTorch unzips to a specific structure where `data.pkl` is inside an internal directory.
    # We find that directory so we zip it up perfectly.
    true_root = folder_path
    for p in folder_path.rglob("data.pkl"):
        true_root = p.parent
        break
        
    print(f"    Repacking it into strict PyTorch format at '{target_path}'...")
    
    # 3. Create the new .pt file (which is just an uncompressed zip containing an 'archive' folder)
    with zipfile.ZipFile(target_path, 'w', compression=zipfile.ZIP_STORED) as zf:
        for root, _, files in os.walk(true_root):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Ensure Linux-style forward slashes for the archive internal paths
                arcname = os.path.relpath(file_path, true_root).replace('\\', '/')
                
                # PyTorch expects all files to be inside a root directory (usually named 'archive/')
                arcname = "archive/" + arcname
                zf.write(file_path, arcname)
                
    print(f"[+] Successfully created '{target_path}'!\n")

if __name__ == "__main__":
    # Repack 'last' folder to 'last.pt'
    reconstruct_pt("last")
    
    # Repack 'best' folder to 'best.pt'
    reconstruct_pt("best")