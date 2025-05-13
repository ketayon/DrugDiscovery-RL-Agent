import time
import logging
import torch
import torch.nn as nn
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator


log = logging.getLogger(__name__)
job_store = {}
IBM_TOKEN = None


def conv_layer(num_qubits, label, params):
    qc = QuantumCircuit(num_qubits, name=label)
    for i in range(num_qubits):
        qc.rx(params[i], i)
    return qc

def pool_layer(qubits, label):
    qc = QuantumCircuit(len(qubits), name=label)
    for i in range(0, len(qubits) - 1, 2):
        qc.cx(qubits[i], qubits[i + 1])
    return qc

def build_ansatz(num_qubits, params):
    ansatz = QuantumCircuit(num_qubits, name="Ansatz")
    layer_params = iter(params)

    for layer in range(1, 4):
        conv_params = [next(layer_params) for _ in range(num_qubits)]
        ansatz.compose(conv_layer(num_qubits, f"c{layer}", conv_params), range(num_qubits), inplace=True)
        ansatz.compose(pool_layer(range(num_qubits), f"p{layer}"), range(num_qubits), inplace=True)
    return ansatz

def calculate_total_params(num_qubits, layers=3):
    return num_qubits * layers


def connect_ibm_backend(token):
    global IBM_TOKEN
    IBM_TOKEN = token  # ✅ Save for later use

    if not token:
        raise EnvironmentError("IBM Quantum token is required.")

    service = QiskitRuntimeService(
        channel="ibm_quantum",
        instance="ibm-q/open/main",
        token=token
    )
    backend = service.least_busy(operational=True, simulator=False)
    estimator = Estimator(mode=backend)
    log.info(f"🔗 Connected to IBM backend: {backend.name}")
    return estimator, backend


def submit_ibm_job(features, estimator, backend, max_pending=3, wait_time=60):
    service = QiskitRuntimeService(
        channel="ibm_quantum",
        instance="ibm-q/open/main",
        token=IBM_TOKEN
    )
    while True:
        try:
            recent_jobs = service.jobs(limit=20)
            pending_jobs = [
                job for job in recent_jobs
                if str(job.backend()) == backend.name and str(job.status()) in ["QUEUED", "RUNNING"]
            ]

            if len(pending_jobs) < max_pending:
                break

            log.warning(f"⏳ Max pending jobs ({max_pending}) reached. Waiting {wait_time}s...")
            time.sleep(wait_time)

        except Exception as e:
            log.warning(f"⚠️ Could not check job queue: {e}. Retrying...")
            time.sleep(wait_time)


    num_qubits = 18
    layers = 3
    total_params = calculate_total_params(num_qubits, layers)
    if len(features) != total_params:
        raise ValueError(f"Expected {total_params} features, got {len(features)}")

    params = [Parameter(f"θ{i}") for i in range(total_params)]
    circuit = build_ansatz(num_qubits, params)
    observable = SparsePauliOp("Z" * num_qubits)

    pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=1)
    transpiled = pass_manager.run(circuit)
    observable = observable.apply_layout(transpiled.layout)

    job = estimator.run([(transpiled, observable, [features])])
    job_store[job.job_id()] = job
    log.info(f"🚀 Submitted IBM Quantum job: {job.job_id()}")
    return job.job_id()


def wait_for_ibm_result(job_id, timeout=600, poll_interval=5):
    job = job_store.get(job_id)

    if job is None:
        log.warning("🔁 Job not found locally. Trying to retrieve from IBM service...")
        try:
            service = QiskitRuntimeService(
                channel="ibm_quantum",
                instance="ibm-q/open/main",
                token=IBM_TOKEN  # ✅ use saved token
            )
            job = service.job(job_id)
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve job from IBM service: {e}")

    log.info(f"⏳ Waiting for IBM Quantum job {job_id} to complete...")
    start_time = time.time()

    while True:
        try:
            job_status = str(job.status())
        except Exception as e:
            raise RuntimeError(f"Failed to fetch job status: {e}")

        log.info(f"🌀 Job status: {job_status}")

        if job_status == "DONE":
            break
        elif job_status in {"CANCELLED", "ERROR", "FAILED"}:
            raise RuntimeError(f"IBM Quantum job {job_id} failed with status: {job_status}")
        elif time.time() - start_time > timeout:
            raise TimeoutError(f"IBM Quantum job {job_id} timed out after {timeout} seconds.")

        time.sleep(poll_interval)

    try:
        result = job.result()
        value = float(result[0].data.evs)
    except Exception as e:
        raise RuntimeError(f"Failed to get result from IBM Quantum job: {e}")

    log.info(f"✅ IBM Quantum job {job_id} result: {value:.4f}")
    return value


class IBMQPolicyNet(nn.Module):
    def __init__(self, n_actions, estimator, backend):
        super().__init__()
        self.estimator = estimator
        self.backend = backend
        self.num_qubits = 18
        self.layers = 3
        self.total_params = self.num_qubits * self.layers

        self.q_params = nn.Parameter(torch.randn(self.total_params))
        self.linear = nn.Linear(1, n_actions)

    def forward(self, x):
        try:
            features = self.q_params.detach().cpu().numpy().tolist()
            job_id = submit_ibm_job(features, self.estimator, self.backend)
            result_value = wait_for_ibm_result(job_id)
        except Exception as e:
            log.warning(f"⚠️ Quantum job failed or timed out: {str(e)}. Using fallback.")
            result_value = 0.5

        q_out = torch.tensor([[result_value]], dtype=torch.float32)
        return self.linear(q_out)
