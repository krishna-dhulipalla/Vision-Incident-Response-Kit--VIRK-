from abc import ABC, abstractmethod
from pathlib import Path
import shutil
import os
from typing import Optional

class ArtifactStorage(ABC):
    """Abstract base class for incident bundle storage."""
    
    @abstractmethod
    def save(self, local_path: Path, remote_key: str) -> str:
        """Save local file to storage, return accessible URI."""
        pass

class LocalStorage(ArtifactStorage):
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def save(self, local_path: Path, remote_key: str) -> str:
        target = self.output_dir / remote_key
        shutil.copy2(local_path, target)
        return str(target.absolute())

class S3Storage(ArtifactStorage):
    """
    S3 Storage backend. 
    Requires 'boto3' to be installed and credentials configured in env.
    """
    def __init__(self, bucket: str, prefix: str = "incidents"):
        self.bucket = bucket
        self.prefix = prefix
        try:
            import boto3
            self.s3 = boto3.client('s3')
            self.boto_available = True
        except ImportError:
            print("Warning: boto3 not installed. S3Storage disabled.")
            self.boto_available = False
            
    def save(self, local_path: Path, remote_key: str) -> str:
        if not self.boto_available:
            raise ImportError("boto3 required for S3Storage")
            
        key = f"{self.prefix}/{remote_key}"
        try:
            self.s3.upload_file(str(local_path), self.bucket, key)
            return f"s3://{self.bucket}/{key}"
        except Exception as e:
            print(f"S3 Upload failed: {e}")
            return f"failed_upload://{key}"
