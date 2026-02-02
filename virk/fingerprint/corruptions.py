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
    def get_all() -> Dict[str, Callable]:
        return {
            "brightness": Corruptions.brightness,
            "gaussian_blur": Corruptions.gaussian_blur,
            "gaussian_noise": Corruptions.gaussian_noise,
            "defocus_blur": Corruptions.defocus_blur
        }
