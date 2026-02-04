import os
import shutil
from PIL import Image
from tqdm import tqdm
import torchvision.datasets as datasets

def setup_cifar10(output_dir: str, num_samples: int = 1000):
    """Downloads CIFAR-10 and dumps `num_samples` test images to output_dir."""
    print(f"Setting up CIFAR-10 in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    if len(os.listdir(output_dir)) >= num_samples:
        print("Directory not empty, skipping download.")
        return

    # Download to temp
    cifar = datasets.CIFAR10(root='./temp_data', train=False, download=True)
    
    print(f"Extracting {num_samples} images...")
    for i in tqdm(range(min(num_samples, len(cifar)))):
        img, label = cifar[i]
        # Upscale to 224x224
        img = img.resize((224, 224), Image.BICUBIC)
        img.save(os.path.join(output_dir, f"cifar_{i:04d}_cls{label}.png"))
        
    print("Done.")

def setup_flowers102(output_dir: str, num_samples: int = 1000):
    """
    Downloads Flowers-102 dataset (fine-grained) and dumps images.
    Requires 'scipy' for .mat loading usually, but pytorch might handle it.
    """
    print(f"Setting up Flowers-102 in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    if len(os.listdir(output_dir)) >= num_samples:
        print("Directory not empty, skipping download.")
        return
        
    # Flowers102 needs split='test' usually
    try:
        flowers = datasets.Flowers102(root='./temp_data', split='test', download=True)
    except ImportError:
        print("Error: Flowers102 requires scipy. Please install it: pip install scipy")
        return

    print(f"Extracting {num_samples} images...")
    for i in tqdm(range(min(num_samples, len(flowers)))):
        img, label = flowers[i]
        # Flowers are variable size, Resize to 224x224
        img = img.resize((224, 224), Image.BICUBIC)
        img.save(os.path.join(output_dir, f"flower_{i:04d}_cls{label}.jpg"))
        
    print("Done.")

DATASET_REGISTRY = {
    "cifar10": setup_cifar10,
    "flowers102": setup_flowers102
}
