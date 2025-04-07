import pytest
import torch
import numpy as np
from unittest.mock import patch

from agent_qiskit import run_qiskit_circuit, QiskitPolicyNet


def test_run_qiskit_circuit_probability():
    """Test that the circuit returns a valid probability in [0, 1]."""
    angle = np.pi / 2  # RY(pi/2) puts qubit into equal superposition
    prob = run_qiskit_circuit(angle)
    assert 0.0 <= prob <= 1.0, "Probability must be between 0 and 1"
    assert isinstance(prob, float)


@patch("agent_qiskit.run_qiskit_circuit", return_value=0.75)
def test_qiskit_policy_net_forward_shape(mock_circuit):
    """Test QiskitPolicyNet forward pass output shape."""
    model = QiskitPolicyNet(n_actions=3)
    input_tensor = torch.tensor([0.5], dtype=torch.float32)
    output = model(input_tensor)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 3)


@patch("agent_qiskit.run_qiskit_circuit", side_effect=[0.25, 0.75])
def test_qiskit_policy_net_different_input_output(mock_circuit):
    """Test that different quantum results yield different outputs."""
    model = QiskitPolicyNet(n_actions=2)

    input_1 = torch.tensor([0.1], dtype=torch.float32)
    out_1 = model(input_1)

    input_2 = torch.tensor([0.9], dtype=torch.float32)
    out_2 = model(input_2)

    assert not torch.allclose(out_1, out_2), "Outputs should differ with different quantum results"


def test_model_has_trainable_parameters():
    """Ensure model has trainable weights."""
    model = QiskitPolicyNet(3)
    params = list(model.parameters())
    assert any(p.requires_grad for p in params), "Model should have trainable parameters"
