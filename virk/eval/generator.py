from typing import List, Tuple, Dict
import os
import shutil
import numpy as np
from PIL import Image
from virk.fingerprint.corruptions import Corruptions

class ShiftGenerator:
    """Generates shifted versions of a dataset for evaluation."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        
    def generate(self, image_paths: List[str], shifts: List[str] = None, severities: List[int] = [3]) -> Dict[str, List[str]]:
        """
        Generate shifted versions of the input images.
        
        Args:
            image_paths: List of paths to source images.
            shifts: List of shift types to apply (e.g. ['brightness', 'gaussian_blur']). 
                   If None, uses all available.
            severities: List of severity levels (1-5) to generate.
            
        Returns:
            Dict mapping "shift_type_severity" to list of generated image paths.
        """
        available_shifts = Corruptions.get_all()
        if shifts is None:
            shifts = list(available_shifts.keys())
            
        results = {}
        
        for shift_name in shifts:
            if shift_name not in available_shifts:
                print(f"Warning: Shift '{shift_name}' not found. Skipping.")
                continue
                
            func = available_shifts[shift_name]
            
            for severity in severities:
                shift_key = f"{shift_name}_s{severity}"
                shift_dir = os.path.join(self.output_dir, shift_key)
                os.makedirs(shift_dir, exist_ok=True)
                
                generated_paths = []
                for p in image_paths:
                    filename = os.path.basename(p)
                    target_path = os.path.join(shift_dir, filename)
                    
                    try:
                        img = Image.open(p).convert('RGB')
                        shifted_img = func(img, severity=severity)
                        shifted_img.save(target_path)
                        generated_paths.append(target_path)
                    except Exception as e:
                        print(f"Error generating {shift_key} for {p}: {e}")
                
                results[shift_key] = generated_paths
                
        return results
