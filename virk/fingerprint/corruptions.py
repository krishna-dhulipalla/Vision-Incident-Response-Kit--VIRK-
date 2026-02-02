import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
from typing import List, Callable, Dict

class Corruptions:
    """Collection of image corruptions to simulate common shifts."""
    
    @staticmethod
    def brightness(img: Image.Image, severity: int = 3) -> Image.Image:
        """Adjust brightness. Severity 1-5."""
        # severity 1: 0.9, 5: 0.5 (darker)
        factor = 1.0 - (severity * 0.1)
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)

    @staticmethod
    def gaussian_blur(img: Image.Image, severity: int = 3) -> Image.Image:
        """Apply Gaussian Blur. Severity 1-5."""
        radius = severity * 0.5
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

    @staticmethod
    def gaussian_noise(img: Image.Image, severity: int = 3) -> Image.Image:
        """Add Gaussian Noise. Severity 1-5."""
        # This is slower in pure python/PIL, keeping it simple
        np_img = np.array(img, dtype=np.float32)
        sigma = severity * 10
        noise = np.random.normal(0, sigma, np_img.shape)
        noisy_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)
    
    @staticmethod
    def defocus_blur(img: Image.Image, severity: int = 3) -> Image.Image:
         # Simulating defocus with box blur for speed/simplicity
         radius = severity
         return img.filter(ImageFilter.BoxBlur(radius))

    @staticmethod
    def jpeg_compression(img: Image.Image, severity: int = 3) -> Image.Image:
        """Apply JPEG Compression artifacts. Severity 1-5."""
        # Severity 1: Quality 90, Severity 5: Quality 10
        quality = int(100 - (severity * 18))
        quality = max(5, min(95, quality))
        
        import io
        buffer = io.BytesIO()
        img.save(buffer, "JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert('RGB')
        
    @staticmethod
    def motion_blur(img: Image.Image, severity: int = 3) -> Image.Image:
        """Simulate motion blur using accumulation of shifted images."""
        import numpy as np
        
        # Severity maps to blur length
        length = severity * 4  # Length 4, 8, 12, 16, 20 pixels
        if length < 1:
            return img
            
        # Horizontal motion
        arr = np.array(img, dtype=np.float32)
        acc = np.zeros_like(arr)
        
        # Simple box blur along x-axis
        # We accumulate shifted versions
        # This is strictly a 'motion smear'
        count = 0
        for i in range(length):
            # Shift i pixels to the right
            # We use roll, but handle boundaries by simple wrap (good enough for eval)
            # or better: we just accept the artifact at edge
            shifted = np.roll(arr, shift=i, axis=1)
            acc += shifted
            count += 1
            
        avg = acc / count
        return Image.fromarray(avg.astype(np.uint8))

    @staticmethod
    def scale(img: Image.Image, severity: int = 3) -> Image.Image:
        """Downsample and upsample to lose detail."""
        # Severity 1: 0.9x, Severity 5: 0.5x
        factor = 1.0 - (severity * 0.1)
        w, h = img.size
        new_w, new_h = int(w * factor), int(h * factor)
        
        downsampled = img.resize((new_w, new_h), Image.BILINEAR)
        return downsampled.resize((w, h), Image.NEAREST) # Nearest to keep blocks visible

    @staticmethod
    def get_all() -> Dict[str, Callable]:
        return {
            "brightness": Corruptions.brightness,
            "gaussian_blur": Corruptions.gaussian_blur,
            "gaussian_noise": Corruptions.gaussian_noise,
            "defocus_blur": Corruptions.defocus_blur,
            "jpeg_compression": Corruptions.jpeg_compression,
            "motion_blur": Corruptions.motion_blur,
            "scale": Corruptions.scale
        }
