"""Uncertainty estimates for independent clean evaluation batches."""

from math import sqrt
from statistics import NormalDist

import numpy as np


def false_positive_summary(scores, threshold: float, confidence: float = 0.95):
    """Wilson score interval, conditional on independent Bernoulli batch outcomes.

    Reference: https://itl.nist.gov/div898/handbook/prc/section2/prc241.htm
    Do not apply this interval to overlapping sliding-window checks.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("scores must be a nonempty finite vector")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    count = len(scores)
    positives = int(np.count_nonzero(scores > threshold))
    rate = positives / count
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    denominator = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / denominator
    half_width = (
        z * sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / denominator
    )
    return {
        "num_batches": count,
        "false_positives": positives,
        "fpr": rate,
        "confidence": confidence,
        "ci_lower": max(0.0, center - half_width),
        "ci_upper": min(1.0, center + half_width),
        "interval_method": "wilson",
        "assumption": "Independent representative clean batches; no guarantee on future data.",
    }
