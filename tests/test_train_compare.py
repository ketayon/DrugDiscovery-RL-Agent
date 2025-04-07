import pytest
import logging
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from agent_crl import ClassicalPolicyNet
from drug_env import DrugDiscoveryEnv
from train_compare import evaluate_model


@pytest.fixture
def dummy_env():
    smiles = ["CCO", "C1=CC=CC=C1", "CC(=O)O"]
    return DrugDiscoveryEnv(smiles)


@pytest.fixture
def dummy_model(dummy_env):
    return ClassicalPolicyNet(n_actions=dummy_env.action_space.n)


def test_training_step_logic(dummy_env):
    """Test training logic without full loop."""
    model = ClassicalPolicyNet(n_actions=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    obs, _ = dummy_env.reset()
    x = torch.tensor(obs, dtype=torch.float32)

    logits = model(x)
    probs = torch.softmax(logits, dim=0)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    log_prob = dist.log_prob(action)

    _, reward, *_ = dummy_env.step(action.item())

    loss = -log_prob * reward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert isinstance(reward, float)
    assert isinstance(loss.item(), float)
