# Quantum Reinforcement Learning for Molecular Optimization

This project explores **Quantum-enhanced Reinforcement Learning (QRL)** for **drug discovery** using real molecular data from the Tox21 dataset.

Agents (classical and quantum) learn to modify molecules to **improve chemical properties** (e.g., logP). The environment is built using `RDKit`, and quantum circuits are simulated using `PennyLane`.

---

## 🔬 What This Project Does

- Loads real-world SMILES data from the **Tox21** dataset
- Defines a custom **OpenAI Gym environment** for molecule editing
- Uses **Quantum** or **Classical** policy networks
- Trains an agent using the **REINFORCE policy gradient** algorithm
- Renders molecules and logs chemical property rewards (e.g., logP)

---

## 📦 Dependencies

Install all required libraries:

```bash
pip install rdkit gym torch pandas matplotlib pennylane deepchem tensorflow
```
## 🚀 How to Run
Clone or download this repo

Run the training script:

```bash
python train_compare.py
```
- ✨ The script auto-downloads tox21.csv.gz on first run

## 🧠 Quantum vs Classical Mode

Edit the train_compare.py file:
```bash
use_quantum = True  # Set to False to use classical agent
```
True → Quantum policy (via PennyLane)

False → Classical neural net policy

## 🧪 Molecule Actions

Action ID	Modification
0	        Do nothing
1	        Add methyl group
2	        Add hydroxyl group
