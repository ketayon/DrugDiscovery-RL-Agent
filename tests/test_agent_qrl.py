import pytest
import torch
from unittest.mock import patch
from agent_qrl import QuantumPolicyNet, quantum_policy


def test_quantum_policy_output_range():
    """Test that quantum_policy returns a float between -1 and 1."""
    inputs = torch.tensor([0.5], dtype=torch.float32)
    weights = torch.randn((2, 1), dtype=torch.float32)
    output = quantum_policy(inputs, weights)

    assert isinstance(output, torch.Tensor)
    assert output.ndim == 0
    assert -1.0 <= output.item() <= 1.0


def test_quantum_policy_net_forward_shape():
    """Ensure QuantumPolicyNet forward output has correct shape."""
    model = QuantumPolicyNet(n_actions=2)
    input_tensor = torch.tensor([0.25], dtype=torch.float32)
    output = model(input_tensor)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 2)


@patch("agent_qrl.quantum_policy", return_value=torch.tensor(0.7))
def test_mocked_quantum_policy_net_output(mock_qc):
    """Test QuantumPolicyNet forward when quantum_policy is mocked."""
    model = QuantumPolicyNet(n_actions=3)
    input_tensor = torch.tensor([0.5], dtype=torch.float32)
    output = model(input_tensor)

    assert output.shape == (1, 3)
    mock_qc.assert_called_once()


def test_model_has_trainable_parameters():
    """Ensure the quantum weights and linear layer are trainable."""
    model = QuantumPolicyNet(n_actions=2)
    params = list(model.parameters())

    assert any(p.requires_grad for p in params), "Model should have trainable parameters"
    assert model.weights.shape == (2, 1)
