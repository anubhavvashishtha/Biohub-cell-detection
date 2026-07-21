import os
import zipfile

def zip_project(output_filename="biohub_code.zip"):
    print(f"Creating {output_filename}...")
    
    # Files and directories to include
    targets = [
        "src",
        "configs",
        "requirements.txt"
    ]
    
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for target in targets:
            if not os.path.exists(target):
                print(f"Warning: {target} not found, skipping.")
                continue
                
            if os.path.isfile(target):
                zipf.write(target, target)
                print(f"Added file: {target}")
            else:
                for root, dirs, files in os.walk(target):
                    # Skip __pycache__
                    if "__pycache__" in root:
                        continue
                        
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Explicitly ensure we include our new .pt files
                        zipf.write(file_path, file_path)
                        print(f"Added: {file_path}")
                        
    print(f"Successfully created {output_filename}")

if __name__ == "__main__":
    zip_project()
