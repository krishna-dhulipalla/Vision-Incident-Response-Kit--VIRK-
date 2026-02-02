import os
import json
import shutil
import zipfile
from typing import List, Dict, Any
from pathlib import Path
from virk.core.types import Incident

class ReproBundler:
    """Generates the incident reproduction package."""
    
    def __init__(self, output_dir: str = "incidents"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _create_replay_script(self, target_dir: Path):
        """Writes the replay.py script to the bundle."""
        script_content = """
import json
import argparse
import sys
import random
import numpy as np
import torch
from pathlib import Path

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Replay VIRK incident")
    parser.add_argument("--incident", default="manifest.json", help="Path to incident manifest")
    args = parser.parse_args()
    
    # 1. Enforce Determinism
    set_seed(42)
    print("Determinism enforced: Random Seed 42")
    
    manifest_path = Path(args.incident)
    if not manifest_path.exists():
        print(f"Error: {manifest_path} not found.")
        sys.exit(1)
        
    with open(manifest_path, 'r') as f:
        data = json.load(f)
        
    print(f"Replaying Incident: {data['incident_id']}")
    print(f"Computed Data Hash: {data.get('data_hash', 'N/A')}")
    print("-" * 30)
    
    print("Drift Detected:", data['drift_profile']['is_drift_detected'])
    print("Magnitude:", data['drift_profile']['drift_magnitude'])
    
    print("-" * 30)
    print("Shift Fingerprint (Probable Causes):")
    for cause, score in data['fingerprint']['shift_types'].items():
        if score > 0.01:
            print(f"  - {cause}: {score:.2f}")
            
    print("-" * 30)
    print("Top Contributing Slices:")
    for sl in data['top_slices']:
        print(f"  - {sl['slice_name']}: {sl['contribution_score']:.2f}")

if __name__ == "__main__":
    main()
"""
        with open(target_dir / "replay.py", "w") as f:
            f.write(script_content)

    def bundle(self, incident: Incident, image_paths: List[str]) -> str:
        """
        Create a zip bundle for the incident.
        """
        import hashlib
        
        bundle_name = f"incident_{incident.incident_id}"
        bundle_path = self.output_dir / bundle_name
        
        # 1. Create temporary directory structure
        if bundle_path.exists():
            shutil.rmtree(bundle_path)
        bundle_path.mkdir()
        
        img_dir = bundle_path / "images"
        img_dir.mkdir()
        
        # 2. Copy images and Compute Hash
        # Deterministic sort to ensure hash consistency independent of FS order
        image_paths = sorted(image_paths)
        
        hasher = hashlib.sha256()
        
        # Cap at 20 images
        saved_count = 0
        for img_p in image_paths[:20]:
            try:
                shutil.copy2(img_p, img_dir)
                # Update hash with file content
                with open(img_p, "rb") as f:
                    chunk = f.read()
                    hasher.update(chunk)
                saved_count += 1
            except Exception as e:
                print(f"Warning: could not copy {img_p}: {e}")
        
        data_hash = hasher.hexdigest()
        incident.metadata["data_hash"] = data_hash
        incident.metadata["bundle_version"] = "1.0.0"
        incident.metadata["sample_count"] = saved_count
                
        # 3. Write manifest
        with open(bundle_path / "manifest.json", "w") as f:
            json.dump(incident.to_dict(), f, indent=2)
            
        # 4. Create replay script
        self._create_replay_script(bundle_path)
        
        # 5. Zip it up
        zip_file = self.output_dir / f"{bundle_name}.zip"
        with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(bundle_path):
                # Sort files to ensure zip binary determinism (roughly)
                for file in sorted(files):
                    file_path = Path(root) / file
                    archive_name = file_path.relative_to(bundle_path)
                    zf.write(file_path, arcname=archive_name)
                    
        # Cleanup
        shutil.rmtree(bundle_path)
        
        return str(zip_file)
