import torch
from pathlib import Path

def check_checkpoint(file_path):
    path = Path(file_path)
    if not path.exists():
        print(f"[-] {path} does not exist.")
        return
        
    print(f"\n{'='*50}")
    print(f"Loading: {path.name}")
    print(f"{'='*50}")
    
    try:
        # Load to CPU so it works instantly without needing GPU memory
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        
        print("Available Keys:")
        for key, value in ckpt.items():
            # If it's a dictionary (like model or optimizer state), just print its type/size
            if isinstance(value, dict):
                print(f"  - {key}: dict with {len(value)} keys")
            # If it's a tensor, print its shape
            elif torch.is_tensor(value):
                print(f"  - {key}: Tensor of shape {value.shape}")
            # If it's a normal primitive (like epoch, val_loss, etc), print the actual value!
            else:
                print(f"  - {key}: {value}")
                
    except Exception as e:
        print(f"[!] Failed to load {path.name}: {e}")

if __name__ == "__main__":
    check_checkpoint("src/last.pt")
    check_checkpoint("src/best.pt")