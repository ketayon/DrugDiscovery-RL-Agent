# ⚛️ Quantum Reinforcement Learning for Molecular Optimization

This project applies **Reinforcement Learning (RL)** to molecular design using real chemical structures — powered by **Quantum Circuits**, **IBM Quantum**, and **Classical Neural Networks**.

Agents learn to modify molecules (SMILES) to improve chemical properties like **logP** using OpenAI Gym, RDKit, and Quantum backends like **PennyLane**, **Qiskit Aer**, and **IBM Quantum**.

---

## 🚀 What This Project Can Do

- ✅ Load real molecules from the **Tox21** dataset
- 🧪 Define a Gym-compatible molecule editing environment
- ⚛️ Train policy networks using:
  - Classical neural nets
  - Quantum circuits (PennyLane)
  - Quantum circuit simulation (Qiskit Aer)
  - Real IBM Quantum hardware 💻🔁⚛️
- 📈 Plot reward history and chemical property distributions
- 🧬 Modify molecules using **REINFORCE policy gradients**
- 🔬 Evaluate model performance by logP improvement
- 💾 Save/load trained models

---

## 📦 Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```
## 🧪 Available Agent Types

| Mode      | Description                          | Backend         |
|-----------|--------------------------------------|-----------------|
| classical | Simple PyTorch neural network        | CPU/GPU         |
| quantum   | PennyLane quantum circuit            | PennyLane (sim) |
| aer       | Qiskit AerSimulator quantum circuit  | Qiskit Aer      |
| ibm       | Real quantum backend from IBM Quantum | IBM Q Runtime   |

---

## 🧠 Actions the Agent Can Take

| Action | Modification             |
|--------|--------------------------|
| 0      | Do nothing               |
| 1      | Add methyl group (–CH₃)  |
| 2      | Add hydroxyl group (–OH) |

---

## 🛠️ How to Use

### 1. Clone this repo

```bash
git clone https://github.com/yourusername/DrugDiscovery-RL-Agent.git
cd DrugDiscovery-RL-Agent
```

### 2. Run training

```bash
python train_compare.py
```

#### You will be prompted to select an agent type:

```bash
🤖 What type of agent would you like to train?
  - classical
  - quantum
  - aer
  - ibm
    - If you choose ibm, you will be prompted for your IBM Quantum API token.
```

### ❗️❗️❗️ Pay attention **Note on IBM Quantum Access**

If you choose the `ibm` agent (real quantum backend), please be aware:

- ⏱️ **Free plan**: Jobs are queued behind others and may take **~10 minutes** to run.
- 💳 **Paid plans**: Provide **priority access**, but **may incur charges** depending on your usage and account limits.

You will be prompted to enter your **IBM Quantum API token** at runtime. You can manage your plan and quota at [https://quantum.ibm.com](https://quantum.ibm.com).


## 💾 Saved Model Paths

Trained models are saved automatically in the `/models/` directory:

| Agent     | Saved Path              |
|-----------|-------------------------|
| Classical | `models/model_crl.pth`  |
| Quantum   | `models/model_qrl.pth`  |
| Aer       | `models/model_aer.pth`  |
| IBM       | `models/model_ibm.pth`  |


### 📊 Evaluation Metrics

```bash
After training, the model is evaluated over 50 new episodes:
  - ✅ Average logP improvement
  - 📈 Histogram of reward (logP) values
  - 🔬 Summary of improved vs. unchanged vs. worse molecules
```

### 🌐 IBM Quantum Integration

``` bash
To use a real quantum backend:
  1. Get an API token from IBM Quantum
  2. Paste it into the prompt when choosing the ibm agent
  3. The program will:
    - Submit jobs
    - Wait for real quantum execution
    - Handle retry/timeout logic
    - ⚠️ IBM allows only a few pending jobs at a time. The code will wait/retry if queue is full.
```

### 🧪 Run Tests

```bash
pytest -v
```

### 🧬 Inspired By
  - PennyLane
  - Qiskit
  - DeepChem
  - RDKit


## 📜 License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

© 2024–2025 Maksym Husarov

