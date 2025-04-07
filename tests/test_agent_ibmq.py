import pytest
import torch
from unittest.mock import MagicMock, patch

from agent_ibmq import (
    build_ansatz,
    calculate_total_params,
    IBMQPolicyNet
)

@pytest.fixture
def dummy_backend():
    backend = MagicMock()
    backend.name = "fake_backend"
    return backend

@pytest.fixture
def dummy_estimator():
    estimator = MagicMock()
    dummy_result = MagicMock()
    dummy_result.result.return_value = [MagicMock(data=MagicMock(evs=0.75))]
    estimator.run.return_value = dummy_result
    return estimator

def test_build_ansatz_shape():
    """Check if ansatz circuit has correct number of parameters."""
    num_qubits = 4
    layers = 3
    total_params = calculate_total_params(num_qubits, layers)
    from qiskit.circuit import Parameter
    params = [Parameter(f"θ{i}") for i in range(total_params)]
    circuit = build_ansatz(num_qubits, params)
    assert isinstance(circuit.num_qubits, int)
    assert circuit.num_qubits == num_qubits
    assert circuit.depth() > 0

def test_calculate_total_params():
    assert calculate_total_params(4, 3) == 12
    assert calculate_total_params(2, 2) == 4

@patch("agent_ibmq.submit_ibm_job", return_value="fake_job_id")
@patch("agent_ibmq.wait_for_ibm_result", return_value=0.75)
def test_ibmq_policy_net_forward(mock_wait, mock_submit, dummy_estimator, dummy_backend):
    model = IBMQPolicyNet(n_actions=2, estimator=dummy_estimator, backend=dummy_backend)
    x = torch.tensor([[0.5]], dtype=torch.float32)
    out = model(x)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 2)

def test_ibmq_policy_net_parameters(dummy_estimator, dummy_backend):
    model = IBMQPolicyNet(n_actions=3, estimator=dummy_estimator, backend=dummy_backend)
    params = list(model.parameters())
    assert any(p.requires_grad for p in params), "No trainable parameters found"
    assert model.q_params.shape[0] == model.num_qubits * model.layers
