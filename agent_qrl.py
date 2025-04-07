import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np
import logging


log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

n_qubits = 1
n_layers = 2
dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev, interface="torch")
def quantum_policy(inputs, weights):
    qml.RY(inputs[0], wires=0)
    qml.templates.BasicEntanglerLayers(weights, wires=[0])
    return qml.expval(qml.PauliZ(0))


class QuantumPolicyNet(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.weights = nn.Parameter(torch.randn((n_layers, n_qubits)))
        self.out = nn.Linear(1, n_actions)
        log.info("Initialized QuantumPolicyNet with %d actions", n_actions)

    def forward(self, x):
        x_scaled = x * np.pi
        q_out = quantum_policy(x_scaled, self.weights)
        q_out = torch.tensor([[q_out]], dtype=torch.float32)
        log.debug("Quantum output: %.4f", q_out.item())
        return self.out(q_out)
