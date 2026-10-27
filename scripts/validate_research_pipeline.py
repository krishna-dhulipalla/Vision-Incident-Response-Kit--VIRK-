"""Offline functional validation; synthetic images and untrained weights.

Run from the repository root in the pinned environment. Results exercise the
reporting pipeline and must not be interpreted as model-quality benchmarks.
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from virk.eval.engine import EvalEngine
from virk.eval.reporter import ReportGenerator
from virk.monitor.extractor import TimmExtractor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="eval_report_research_validation")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory(prefix="virk_research_validation_") as directory:
        for i in range(80):
            Image.fromarray(rng.integers(0, 256, (48, 48, 3), dtype=np.uint8)).save(
                Path(directory) / f"{i:03d}.png"
            )
        engine = EvalEngine(
            directory,
            str(output),
            batch_size=4,
            window_size=8,
            seed=42,
            sample_interval_seconds=0.1,
            benchmark_inference=True,
            extractor=TimmExtractor("resnet18", pretrained=False),
        )
        try:
            report = engine.run()
            report["experiment"][
                "validation_scope"
            ] = "Functional smoke test with synthetic images and randomly initialized ResNet-18; not a research performance benchmark."
            (output / "report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
            )
            ReportGenerator().generate(report, str(output / "report.html"))
        finally:
            engine.cleanup()
    print(f"Offline research pipeline validated. Reports: {output.resolve()}")


if __name__ == "__main__":
    main()
