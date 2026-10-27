"""Portable bundle inspection and exact embedding-level drift replay.

Copied into bundles as replay.py; the default inspection needs only Python.
"""

import argparse
import hashlib
import json
from pathlib import Path


def inspect_bundle(manifest_path, verify_drift=False):
    manifest_path = Path(manifest_path)
    root = manifest_path.parent
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = data.get("metadata", {})
    print(f"Inspecting Incident: {data['incident_id']}")
    expected = metadata.get("data_hash", data.get("data_hash", "N/A"))
    print(f"Recorded Data Hash: {expected}")
    if "bundled_images" in metadata:
        hasher = hashlib.sha256()
        for name in metadata["bundled_images"]:
            hasher.update((root / "images" / name).read_bytes())
        if hasher.hexdigest() != expected:
            raise ValueError("bundled image hash does not match manifest")
        print("Bundled image hash verified.")
    print("Drift Detected:", data["drift_profile"]["is_drift_detected"])
    print("Magnitude:", data["drift_profile"]["drift_magnitude"])
    print("Shift Fingerprint:", data["fingerprint"]["shift_types"])
    print("Top Contributing Slices:", data["top_slices"])
    if not verify_drift:
        print("Manifest inspection only; model inference is not rerun.")
        return data

    context_path = root / "replay_context.json"
    if not context_path.exists():
        raise ValueError("Bundle has no recorded drift replay context")
    context = json.loads(context_path.read_text(encoding="utf-8"))
    for name, expected_digest in context["artifact_hashes"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected_digest:
            raise ValueError(f"Replay artifact hash does not match: {name}")
    import numpy as np
    from sklearn.metrics.pairwise import rbf_kernel

    reference = np.load(root / "reference_embeddings.npy", allow_pickle=False)
    current = np.load(root / "current_embeddings.npy", allow_pickle=False)
    gamma = context["detector"]["gamma"]
    score = max(
        0.0,
        float(
            rbf_kernel(reference, reference, gamma=gamma).mean()
            + rbf_kernel(current, current, gamma=gamma).mean()
            - 2 * rbf_kernel(reference, current, gamma=gamma).mean()
        ),
    )
    threshold = context["detector"]["threshold"]
    expected_score = data["drift_profile"]["drift_magnitude"]
    if not np.isclose(score, expected_score, rtol=1e-5, atol=1e-8):
        raise ValueError(
            f"Replayed drift score {score} differs from recorded {expected_score}"
        )
    if (score > threshold) != data["drift_profile"]["is_drift_detected"]:
        raise ValueError("Replayed drift decision differs from recorded decision")
    print(f"Drift replay verified: score={score:.8f}, threshold={threshold:.8f}")
    return {"score": score, "is_drift_detected": score > threshold}


def main():
    parser = argparse.ArgumentParser(description="Inspect or replay a VIRK incident")
    parser.add_argument("--incident", default="manifest.json")
    parser.add_argument(
        "--verify-drift",
        action="store_true",
        help="Recompute drift using the exact saved embeddings",
    )
    args = parser.parse_args()
    try:
        inspect_bundle(args.incident, args.verify_drift)
    except (ValueError, OSError, ImportError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
