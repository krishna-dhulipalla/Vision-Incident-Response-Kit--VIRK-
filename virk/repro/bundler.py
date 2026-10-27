import os
import json
import shutil
import zipfile
from typing import List
from pathlib import Path
from virk.core.types import Incident


class ReproBundler:
    """Generates the incident reproduction package."""

    def __init__(
        self, output_dir: str = "incidents", replay_config=None, model_state_path=None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.replay_config = replay_config or {}
        self.model_state_path = Path(model_state_path) if model_state_path else None
        if self.model_state_path is not None and not self.model_state_path.is_file():
            raise ValueError(
                f"Model state file does not exist: {self.model_state_path}"
            )

    def _create_replay_script(self, target_dir: Path):
        """Copy the standalone replay utility into the bundle."""
        shutil.copy2(Path(__file__).with_name("replay.py"), target_dir / "replay.py")

    def bundle(self, incident: Incident, image_paths: List[str], evidence=None) -> str:
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
        bundled_images = []
        used_names = set()
        for img_p in image_paths[:20]:
            try:
                source = Path(img_p)
                filename = source.name
                suffix = 1
                while filename.casefold() in used_names:
                    filename = f"{source.stem}_{suffix}{source.suffix}"
                    suffix += 1
                target = img_dir / filename
                shutil.copy2(img_p, target)
                # Update hash with file content
                with open(target, "rb") as f:
                    chunk = f.read()
                    hasher.update(chunk)
                used_names.add(filename.casefold())
                bundled_images.append(filename)
                saved_count += 1
            except Exception as e:
                print(f"Warning: could not copy {img_p}: {e}")

        data_hash = hasher.hexdigest()
        incident.metadata["data_hash"] = data_hash
        incident.metadata["bundle_version"] = "1.0.0"  # Bundle format version
        incident.metadata["schema_version"] = "1.0.0"  # Manifest schema version
        incident.metadata["sample_count"] = saved_count
        incident.metadata["bundled_images"] = bundled_images
        incident.metadata["replay_config"] = self.replay_config

        if evidence is not None:
            import numpy as np

            np.save(
                bundle_path / "reference_embeddings.npy",
                evidence.reference_embeddings,
                allow_pickle=False,
            )
            np.save(
                bundle_path / "current_embeddings.npy",
                evidence.current_embeddings,
                allow_pickle=False,
            )
            artifact_names = ["reference_embeddings.npy", "current_embeddings.npy"]
            if self.model_state_path is not None:
                shutil.copy2(self.model_state_path, bundle_path / "model_state.pt")
                artifact_names.append("model_state.pt")
            context = {
                "format_version": "1.0.0",
                "configuration": self.replay_config,
                "detector": {
                    "statistic": "biased_rbf_mmd_squared",
                    "gamma": evidence.gamma,
                    "threshold": evidence.threshold,
                    "comparison": ">",
                },
                "reference_shape": list(evidence.reference_embeddings.shape),
                "current_shape": list(evidence.current_embeddings.shape),
                "artifact_hashes": {
                    name: hashlib.sha256((bundle_path / name).read_bytes()).hexdigest()
                    for name in artifact_names
                },
                "scope": "Exact embedding-level drift replay. Capped sample images do not reconstruct the full traffic window.",
                "model_state_included": self.model_state_path is not None,
            }
            (bundle_path / "replay_context.json").write_text(
                json.dumps(context, indent=2), encoding="utf-8"
            )
            incident.metadata["replay_context"] = "replay_context.json"

        # Capture installed versions without requiring pip in the runtime.
        try:
            from importlib.metadata import distributions

            reqs = (
                "\n".join(
                    sorted(
                        {
                            f"{dist.metadata['Name']}=={dist.version}"
                            for dist in distributions()
                            if dist.metadata["Name"]
                        }
                    )
                )
                + "\n"
            )
            with open(bundle_path / "requirements.txt", "w") as f:
                f.write(reqs)
        except Exception as e:
            print(f"Warning: Could not capture requirements: {e}")

        # 4. Write manifest
        with open(bundle_path / "manifest.json", "w") as f:
            json.dump(incident.to_dict(), f, indent=2)

        # 4. Create replay script
        self._create_replay_script(bundle_path)

        # 5. Zip it up
        zip_file = self.output_dir / f"{bundle_name}.zip"
        with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(bundle_path):
                # Sort files to ensure zip binary determinism (roughly)
                for file in sorted(files):
                    file_path = Path(root) / file
                    archive_name = file_path.relative_to(bundle_path)
                    zf.write(file_path, arcname=archive_name)

        # Cleanup
        shutil.rmtree(bundle_path)

        return str(zip_file)
