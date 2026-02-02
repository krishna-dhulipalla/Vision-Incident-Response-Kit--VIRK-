from PIL import Image, ImageFilter
import numpy as np

def motion_blur(img: Image.Image, severity: int = 3) -> Image.Image:
    """Simulate motion blur using a directional kernel."""
    # Simple kernel approximation (horizontal motion)
    # Severity maps to kernel size
    k_size = severity * 2 + 1
    kernel = np.zeros((k_size, k_size))
    kernel[k_size // 2, :] = 1 / k_size
    
    # We need a custom filter
    # PIL Kernel filter takes a flat list
    flat_kernel = kernel.flatten().tolist()
    print(f"Severity: {severity}, Kernel Size: {k_size}, Flat Kernel Len: {len(flat_kernel)}")
    print(f"Kernel: {flat_kernel}")
    return img.filter(ImageFilter.Kernel((k_size, k_size), flat_kernel))

if __name__ == "__main__":
    # Create dummy image
    img = Image.new('RGB', (100, 100), color = 'red')
    
    try:
        res = motion_blur(img, severity=3)
        res.save("debug_blur_out.jpg")
        print("Success")
    except Exception as e:
        print(f"Failed: {e}")
