import torch
import torch.nn as nn
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
import numpy as np
import logging


log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

simulator = AerSimulator()


def run_qiskit_circuit(angle):
    qc = QuantumCircuit(1)
    qc.ry(angle, 0)
    qc.measure_all()
    transpiled_qc = transpile(qc, simulator)
    result = simulator.run(transpiled_qc, shots=1024).result()
    counts = result.get_counts()
    prob = counts.get('1', 0) / 1024
    return prob


class QiskitPolicyNet(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.linear_out = nn.Linear(1, n_actions)
        log.info("Initialized Qiskit-based policy network.")

    def forward(self, x):
        angle = float(x[0]) * np.pi  # scale input to angle
        prob = run_qiskit_circuit(angle)
        q_out = torch.tensor([[prob]], dtype=torch.float32)
        log.debug("Qiskit output (prob of 1): %.4f", prob)
        return self.linear_out(q_out)
