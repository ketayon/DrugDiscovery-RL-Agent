import pytest
import torch
from agent_crl import ClassicalPolicyNet


def test_model_initialization():
    """Test that the model initializes correctly."""
    model = ClassicalPolicyNet(n_actions=3)
    assert isinstance(model, ClassicalPolicyNet)
    assert hasattr(model, "net")


def test_forward_output_shape():
    """Test that the output shape matches the expected number of actions."""
    n_actions = 3
    model = ClassicalPolicyNet(n_actions)
    input_tensor = torch.tensor([[0.5]], dtype=torch.float32)  # shape (1, 1)
    output = model(input_tensor)

    assert output.shape == (1, n_actions), f"Expected output shape (1, {n_actions}), got {output.shape}"


def test_forward_values_change_with_input():
    """Test that different inputs give different outputs."""
    model = ClassicalPolicyNet(2)
    input_1 = torch.tensor([[0.1]], dtype=torch.float32)
    input_2 = torch.tensor([[0.9]], dtype=torch.float32)

    output_1 = model(input_1)
    output_2 = model(input_2)

    assert not torch.allclose(output_1, output_2), "Model output should change with different inputs"


def test_forward_on_batch():
    """Test forward pass on a batch of inputs."""
    model = ClassicalPolicyNet(4)
    batch_input = torch.tensor([[0.1], [0.5], [0.9]], dtype=torch.float32)  # shape (3, 1)
    output = model(batch_input)

    assert output.shape == (3, 4), f"Expected output shape (3, 4), got {output.shape}"


def test_invalid_input_dimension():
    """Test model behavior when input dimension is incorrect."""
    model = ClassicalPolicyNet(3)
    bad_input = torch.tensor([1.0, 2.0, 3.0])  # wrong shape

    with pytest.raises(RuntimeError):
        model(bad_input)
