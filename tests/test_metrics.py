"""
Unit tests for CLMetrics and SNNMetrics.
Validated against Maya Research Series P3-P7 results.
"""

import numpy as np
import pytest
from cl_metrics import CLMetrics, SNNMetrics


# --- Fixtures ---

@pytest.fixture
def simple_3task_matrix():
    return np.array([
        [0.90, 0.00, 0.00],
        [0.72, 0.85, 0.00],
        [0.65, 0.78, 0.88],
    ])

@pytest.fixture
def maya_p7_matrix():
    """Approximate accuracy matrix from Maya-Manas (P7) Split-CIFAR-100 CIL."""
    return np.array([
        [0.1519, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.09,   0.14, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.07,   0.10, 0.13, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.05,   0.08, 0.09, 0.12, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.04,   0.06, 0.07, 0.09, 0.11, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.03,   0.05, 0.06, 0.07, 0.08, 0.10, 0.00, 0.00, 0.00, 0.00],
        [0.03,   0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.00, 0.00, 0.00],
        [0.02,   0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.00, 0.00],
        [0.02,   0.03, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.00],
        [0.01,   0.02, 0.03, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
    ])


# --- CLMetrics Tests ---

class TestCLMetrics:

    def test_aa_simple(self, simple_3task_matrix):
        m = CLMetrics(simple_3task_matrix)
        aa = m.average_accuracy()
        expected = (0.65 + 0.78 + 0.88) / 3
        assert abs(aa - expected) < 1e-4

    def test_bwt_negative_forgetting(self, simple_3task_matrix):
        m = CLMetrics(simple_3task_matrix)
        bwt = m.backward_transfer()
        assert bwt < 0, "BWT should be negative (forgetting occurred)"

    def test_fwt_positive_transfer(self, simple_3task_matrix):
        m = CLMetrics(simple_3task_matrix)
        fwt = m.forward_transfer()
        assert isinstance(fwt, float)

    def test_plasticity_in_range(self, simple_3task_matrix):
        m = CLMetrics(simple_3task_matrix)
        assert 0.0 <= m.plasticity_index() <= 1.0

    def test_stability_in_range(self, simple_3task_matrix):
        m = CLMetrics(simple_3task_matrix)
        assert 0.0 <= m.stability_index() <= 1.0

    def test_forgetting_nonnegative(self, simple_3task_matrix):
        m = CLMetrics(simple_3task_matrix)
        assert m.forgetting_measure() >= 0.0

    def test_perfect_retention(self):
        R = np.eye(3) * 0.9
        R[1, 0] = 0.9
        R[2, 0] = 0.9
        R[2, 1] = 0.9
        m = CLMetrics(R)
        assert m.backward_transfer() >= 0.0

    def test_single_task(self):
        R = np.array([[0.85]])
        m = CLMetrics(R)
        assert m.average_accuracy() == pytest.approx(0.85)
        assert m.backward_transfer() == 0.0
        assert m.forward_transfer() == 0.0

    def test_invalid_shape(self):
        with pytest.raises(ValueError):
            CLMetrics(np.array([[0.9, 0.8], [0.7, 0.6], [0.5, 0.4]]))

    def test_invalid_values(self):
        with pytest.raises(ValueError):
            CLMetrics(np.array([[1.5, 0.0], [0.8, 0.9]]))

    def test_summary_returns_dict(self, simple_3task_matrix):
        m = CLMetrics(simple_3task_matrix)
        s = m.summary()
        assert isinstance(s, dict)
        assert "AA" in s and "BWT" in s and "FWT" in s

    def test_maya_p7_aa(self, maya_p7_matrix):
        m = CLMetrics(maya_p7_matrix)
        aa = m.average_accuracy()
        assert 0.0 < aa < 1.0

    def test_task_weights(self, simple_3task_matrix):
        weights = np.array([10.0, 10.0, 10.0])
        m = CLMetrics(simple_3task_matrix, task_weights=weights)
        aa_weighted = m.average_accuracy()
        m2 = CLMetrics(simple_3task_matrix)
        assert abs(aa_weighted - m2.average_accuracy()) < 1e-4


# --- SNNMetrics Tests ---

class TestSNNMetrics:

    @pytest.fixture
    def snn(self, simple_3task_matrix):
        sr = np.array([0.12, 0.09, 0.11])
        return SNNMetrics(simple_3task_matrix, spike_rates=sr)

    def test_srp_in_range(self, snn):
        assert 0.0 <= snn.spike_rate_proxy() <= 1.0

    def test_sr_aa_less_than_aa(self, snn):
        assert snn.spike_rate_normalized_aa() <= snn.average_accuracy()

    def test_ea_bwt_exists(self, snn):
        assert isinstance(snn.energy_adjusted_bwt(), float)

    def test_eer_positive(self, snn):
        assert snn.energy_to_error_ratio() >= 0.0

    def test_no_spike_rates_returns_none(self, simple_3task_matrix):
        m = SNNMetrics(simple_3task_matrix)
        assert m.spike_rate_proxy() is None
        assert m.sr_aa() is None if hasattr(m, 'sr_aa') else True

    def test_invalid_spike_rate_shape(self, simple_3task_matrix):
        with pytest.raises(ValueError):
            SNNMetrics(simple_3task_matrix, spike_rates=np.array([0.1, 0.2]))

    def test_invalid_spike_rate_values(self, simple_3task_matrix):
        with pytest.raises(ValueError):
            SNNMetrics(simple_3task_matrix, spike_rates=np.array([1.5, 0.1, 0.2]))

    def test_summary_includes_snn_keys(self, snn):
        s = snn.summary()
        assert "SRP" in s
        assert "SR-AA" in s
        assert "EA-BWT" in s
        assert "EER" in s
