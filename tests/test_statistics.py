import pytest

from virk.eval.statistics import false_positive_summary


def test_zero_observed_false_positives_does_not_imply_zero_uncertainty():
    summary = false_positive_summary([0.01] * 100, threshold=0.02)
    assert summary["fpr"] == 0
    assert summary["ci_lower"] == pytest.approx(0, abs=1e-12)
    assert summary["ci_upper"] == pytest.approx(0.0369935, abs=1e-6)


def test_wilson_interval_known_proportion():
    summary = false_positive_summary([0.1] * 5 + [0.0] * 95, threshold=0.02)
    assert summary["false_positives"] == 5
    assert summary["ci_lower"] == pytest.approx(0.0215437, abs=1e-6)
    assert summary["ci_upper"] == pytest.approx(0.1117505, abs=1e-6)


@pytest.mark.parametrize(
    "scores,confidence", [([], 0.95), ([float("nan")], 0.95), ([1], 1), ([1], 0)]
)
def test_invalid_interval_inputs(scores, confidence):
    with pytest.raises(ValueError):
        false_positive_summary(scores, 0.02, confidence)
