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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        # logging.FileHandler("training.log")
    ]
)
log = logging.getLogger()


if not os.path.exists("tox21.csv.gz"):
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
    urllib.request.urlretrieve(url, "tox21.csv.gz")
    log.info("Downloaded tox21.csv.gz")

df = pd.read_csv("tox21.csv.gz")
raw_smiles = df["smiles"].dropna().unique().tolist()

valid_smiles = []
for s in raw_smiles:
    try:
        if Chem.MolFromSmiles(s):
            valid_smiles.append(s)
    except:
        continue

smiles = valid_smiles[:100]
log.info(f"Loaded {len(smiles)} valid SMILES molecules")

use_quantum = True  # Toggle this
os.makedirs("models", exist_ok=True)
save_path = "models/model_qrl.pth" if use_quantum else "models/model_crl.pth"


env = DrugDiscoveryEnv(smiles)
n_actions = env.action_space.n
model = QuantumPolicyNet(n_actions) if use_quantum else ClassicalPolicyNet(n_actions)
optimizer = optim.Adam(model.parameters(), lr=0.01)


if os.path.exists(save_path):
    model.load_state_dict(torch.load(save_path))
    log.info(f"Loaded model from {save_path}")


reward_history = []
for episode in range(100):
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

# === Save Model ===
torch.save(model.state_dict(), save_path)
log.info(f"Saved model to {save_path}")

# === Plot Reward Curve ===
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

    # === Plot reward distribution ===
    plt.figure()
    plt.hist(total_rewards, bins=10, color='skyblue', edgecolor='black')
    plt.title("Distribution of logP (Reward) During Evaluation")
    plt.xlabel("logP Score")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

evaluate_model(model, env, episodes=50, render=True)
