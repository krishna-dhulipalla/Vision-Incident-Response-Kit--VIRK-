import numpy as np
import pytest

from virk.eval.metrics import MetricCalculator


def test_detection_metrics_use_frozen_threshold_and_strict_comparison():
    result = MetricCalculator.evaluate_detection(
        [0.01, 0.02, 0.03], [0.05, 0.07], threshold=0.02
    )
    assert result["fpr_at_threshold"] == pytest.approx(1 / 3)
    assert result["tpr_at_threshold"] == 1
    assert result["auroc"] == 1


@pytest.mark.parametrize(
    "clean,shifted",
    [([], [1]), ([0], []), ([float("nan")], [1]), ([0], [float("inf")])],
)
def test_invalid_detection_scores_are_not_fabricated_as_auc(clean, shifted):
    with pytest.raises(ValueError):
        MetricCalculator.evaluate_detection(clean, shifted)


def test_single_class_confusion_is_well_defined():
    result = MetricCalculator.evaluate_fingerprinting(["blur"] * 2, ["blur"] * 2)
    assert result == {"accuracy": 1.0, "confusion_matrix": [[2]], "labels": ["blur"]}
