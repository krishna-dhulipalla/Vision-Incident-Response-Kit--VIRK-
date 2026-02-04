import pytest
import sys
from unittest.mock import MagicMock
from virk.integration.metrics import get_backend, NoOpBackend, PrometheusBackend

def test_metrics_fallback_to_prometheus(monkeypatch, caplog):
    # 1. Simulate OTel is MISSING
    monkeypatch.setitem(sys.modules, 'opentelemetry', None)
    
    # 2. Simulate Prometheus is INSTALLED
    mock_prom = MagicMock()
    monkeypatch.setitem(sys.modules, 'prometheus_client', mock_prom)
    
    # Request OTel
    backend = get_backend("test_svc", "otel")
    
    # Validation: Should warn and return PrometheusBackend
    assert "OpenTelemetry requested but not installed" in caplog.text
    # Note: checks class name because class import might be mocked differently
    assert backend.__class__.__name__ == "PrometheusBackend"

def test_metrics_fallback_to_noop(monkeypatch, caplog):
    # 1. Simulate OTel MISSING
    monkeypatch.setitem(sys.modules, 'opentelemetry', None)
    # 2. Simulate Prometheus MISSING
    monkeypatch.setitem(sys.modules, 'prometheus_client', None)
    
    backend = get_backend("test_svc", "otel")
    
    assert "Falling back to Prometheus" in caplog.text
    assert "Prometheus requested but not installed" in caplog.text
    assert isinstance(backend, NoOpBackend)
