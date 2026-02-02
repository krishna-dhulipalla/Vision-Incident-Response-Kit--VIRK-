import os
import shutil
from PIL import Image
import torchvision.datasets as datasets
from tqdm import tqdm

def setup_cifar10(output_dir: str, num_samples: int = 1000):
    """
    Downloads CIFAR-10 and dumps `num_samples` test images to output_dir.
    Useful for file-based evaluation pipeline.
    """
    print(f"Setting up CIFAR-10 in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if empty, if not, assume done
    if len(os.listdir(output_dir)) >= num_samples:
        print("Directory not empty, skipping download.")
        return

    # Download to temp
    cifar = datasets.CIFAR10(root='./temp_data', train=False, download=True)
    
    print(f"Extracting {num_samples} images...")
    for i in tqdm(range(min(num_samples, len(cifar)))):
        img, label = cifar[i]
        # CIFAR images are 32x32. ResNet needs bigger usually (224). 
        # But we can eval on 32 or upscale.
        # Let's upscale to 224 to match standard backbone expectations and reduce "scale" artifact confusion,
        # OR just eval on small images if extractor supports it (timm usually adapts).
        # Let's upscale slightly to 64 or 128 to make corruptions visible. 224 is standard.
        img = img.resize((224, 224), Image.BICUBIC)
        
        img.save(os.path.join(output_dir, f"cifar_{i:04d}_cls{label}.png"))
        
    print("Done.")

if __name__ == "__main__":
    setup_cifar10("cifar10_eval")
