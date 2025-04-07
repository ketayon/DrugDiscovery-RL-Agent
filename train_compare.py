import os
import urllib.request
import pandas as pd
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from rdkit import Chem

from drug_env import DrugDiscoveryEnv
from agent_qrl import QuantumPolicyNet
from agent_crl import ClassicalPolicyNet
from agent_qiskit import QiskitPolicyNet


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger()


def get_data():
    if not os.path.exists("tox21.csv.gz"):
        url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
        urllib.request.urlretrieve(url, "tox21.csv.gz")
        log.info("Downloaded tox21.csv.gz")

    df = pd.read_csv("tox21.csv.gz")
    raw_smiles = df["smiles"].dropna().unique().tolist()
    valid_smiles = [s for s in raw_smiles if Chem.MolFromSmiles(s)]
    smiles = valid_smiles[:100]
    log.info(f"Loaded {len(smiles)} valid SMILES molecules")
    return smiles


def get_agent(use_agent, n_actions, ibm_token=None):
    if use_agent == "quantum":
        model = QuantumPolicyNet(n_actions)
        save_path = "models/model_qrl.pth"
    elif use_agent == "aer":
        model = QiskitPolicyNet(n_actions)
        save_path = "models/model_aer.pth"
    elif use_agent == "ibm":
        from agent_ibmq import IBMQPolicyNet, connect_ibm_backend
        estimator, backend = connect_ibm_backend(ibm_token)
        model = IBMQPolicyNet(n_actions, estimator, backend)
        save_path = "models/model_ibm.pth"
    else:
        model = ClassicalPolicyNet(n_actions)
        save_path = "models/model_crl.pth"

    return model, save_path


def train(model, env, save_path, episodes=100):
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    if os.path.exists(save_path):
        try:
            model.load_state_dict(torch.load(save_path))
            log.info(f"Loaded model from {save_path}")
        except RuntimeError as e:
            log.warning(f"⚠️ Failed to load saved model: {e}. Starting from scratch.")

    reward_history = []

    for episode in range(episodes):
        state, _ = env.reset()
        state_tensor = torch.tensor(state, dtype=torch.float32)

        logits = model(state_tensor)
        probs = torch.softmax(logits, dim=0)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        _, reward, _, _, _ = env.step(action.item())

        loss = -log_prob * reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        reward_history.append(reward)
        log.info(f"Episode {episode+1}: Action {action.item()}, Reward {reward:.2f}")

        if (episode + 1) % 20 == 0:
            log.info("🧪 Molecule after action:")
            env.render()

    torch.save(model.state_dict(), save_path)
    log.info(f"Saved model to {save_path}")

    plt.plot(reward_history)
    plt.title("Reward Curve (QRL vs CRL)")
    plt.xlabel("Episode")
    plt.ylabel("Reward (logP)")
    plt.grid(True)
    plt.show()


def evaluate_model(model, env, episodes=50, render=True):
    log.info("=== Evaluating Model ===")
    model.eval()
    total_rewards = []
    improved, unchanged, worse = 0, 0, 0

    for ep in range(episodes):
        state, _ = env.reset()
        initial_logp = env._get_logp()
        state_tensor = torch.tensor(state, dtype=torch.float32)

        with torch.no_grad():
            logits = model(state_tensor)
            probs = torch.softmax(logits, dim=0)
            action = torch.argmax(probs).item()

        _, final_logp, _, _, _ = env.step(action)
        total_rewards.append(final_logp)

        if final_logp > initial_logp:
            improved += 1
        elif final_logp == initial_logp:
            unchanged += 1
        else:
            worse += 1

        if render and ep % 10 == 0:
            log.info(f"[Eval Ep {ep+1}] logP: {initial_logp:.2f} → {final_logp:.2f}, Action: {action}")
            env.render()

    avg_reward = sum(total_rewards) / len(total_rewards)
    log.info("📊 Evaluation complete.")
    log.info(f"Average logP (reward): {avg_reward:.2f}")
    log.info(f"Improved: {improved}, Unchanged: {unchanged}, Worse: {worse}")

    plt.figure()
    plt.hist(total_rewards, bins=10, color='skyblue', edgecolor='black')
    plt.title("Distribution of logP (Reward) During Evaluation")
    plt.xlabel("logP Score")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    print("\n🤖 What type of agent would you like to train?")
    print("Enter one of the following options:")
    print("  - classical     (simple neural network)")
    print("  - quantum       (PennyLane quantum circuit)")
    print("  - aer           (Qiskit AerSimulator circuit)")
    print("  - ibm           (Run on real IBM Quantum backend)\n")

    use_agent = input("Your choice [classical | quantum | aer | ibm]: ").strip().lower()
    while use_agent not in ["classical", "quantum", "aer", "ibm"]:
        use_agent = input("❌ Invalid choice. Please enter classical, quantum, aer, or ibm: ").strip().lower()

    ibm_token = None
    if use_agent == "ibm":
        ibm_token = input("🔐 Enter your IBM Quantum API token: ").strip()
        while not ibm_token:
            ibm_token = input("❌ Token cannot be empty. Please enter your IBM token: ").strip()

    smiles = get_data()
    env = DrugDiscoveryEnv(smiles)
    n_actions = env.action_space.n

    os.makedirs("models", exist_ok=True)
    model, save_path = get_agent(use_agent, n_actions, ibm_token)

    train(model, env, save_path)
    evaluate_model(model, env)


if __name__ == "__main__":
    main()
