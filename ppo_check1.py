import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ============================
# Simple Policy and Value Networks
# ============================
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


# ============================
# PPO Loss Functions
# ============================
def compute_ppo_loss(ratio, adv, eps=0.2):
    return torch.min(ratio * adv, torch.clamp(ratio, 1 - eps, 1 + eps) * adv)

def compute_sched1_loss(I, L_max, L_min):
    return I * L_max + (1 - I) * L_min

def compute_sched2_loss(I, L_max, L_min):
    return I * L_max + (1 - I) * (L_min + L_max)


# ============================
# Training Function with Reporting
# ============================
def train_variant(env_name, variant, total_episodes=200, warmup=50, alpha=0.1):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNet(state_dim, action_dim)
    value = ValueNet(state_dim)
    optimizer = optim.Adam(list(policy.parameters()) + list(value.parameters()), lr=3e-3)

    report_data = []

    for episode in range(total_episodes):
        state, _ = env.reset()
        log_probs, rewards, values = [], [], []
        done = False

        while not done:
            s = torch.tensor(state, dtype=torch.float32)
            probs = policy(s)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            log_probs.append(dist.log_prob(action))
            rewards.append(reward)
            values.append(value(s))
            state = next_state

        # Compute returns and advantages
        R, adv = 0, []
        returns = []
        for r in reversed(rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        values = torch.cat(values)
        adv = returns - values.detach()

        ratio = torch.exp(torch.stack(log_probs) - torch.stack(log_probs).detach())

        # Define Lmax and Lmin
        L_max = torch.max(
            0.5 * (ratio * adv + torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * adv),
            alpha * (torch.clamp(ratio, -1, 1) * adv)
        )
        L_min = torch.min(
            0.5 * (ratio * adv + torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * adv),
            alpha * (torch.clamp(ratio, -1, 1) * adv)
        )

        I = 1.0 if episode < warmup else 0.0

        # Variant selection
        if variant == "original":
            loss = -compute_ppo_loss(ratio, adv).mean()
        elif variant == "sched1":
            loss = -compute_sched1_loss(I, L_max, L_min).mean()
        elif variant == "sched2":
            loss = -compute_sched2_loss(I, L_max, L_min).mean()
        else:
            raise ValueError("Unknown variant")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_reward = sum(rewards)
        report_data.append({
            "Episode": episode,
            "Variant": variant,
            "Reward": episode_reward,
            "Loss": loss.item(),
            "Advantage_Mean": adv.mean().item(),
            "I(t)": I
        })

        print(f"[{variant}] Ep {episode+1}/{total_episodes} | Reward={episode_reward:.2f} | "
              f"Loss={loss.item():.4f} | Adv={adv.mean().item():.3f} | I={I}")

    env.close()
    return pd.DataFrame(report_data)


# ============================
# Run and Report All Variants
# ============================
if __name__ == "__main__":
    variants = ["original", "sched1", "sched2"]
    all_reports = []

    for v in variants:
        print(f"\n=== Running Variant: {v} ===")
        df = train_variant("CartPole-v1", v)
        all_reports.append(df)

    full_report = pd.concat(all_reports, ignore_index=True)

    # Save reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_report.to_csv(f"PPO_variants_report_{timestamp}.csv", index=False)

    # Write summary text file
    with open(f"PPO_variants_summary_{timestamp}.txt", "w") as f:
        f.write("PPO VARIANT COMPARISON REPORT\n")
        f.write("==============================\n\n")
        for v in variants:
            dfv = full_report[full_report["Variant"] == v]
            f.write(f"Variant: {v}\n")
            f.write(f"  Avg Reward: {dfv['Reward'].mean():.2f}\n")
            f.write(f"  Max Reward: {dfv['Reward'].max():.2f}\n")
            f.write(f"  Final Reward: {dfv['Reward'].iloc[-1]:.2f}\n")
            f.write(f"  Avg Loss: {dfv['Loss'].mean():.4f}\n")
            f.write(f"  Avg Advantage: {dfv['Advantage_Mean'].mean():.4f}\n")
            f.write("\n")

    # Plot reward curves
    plt.figure(figsize=(8, 5))
    for v in variants:
        dfv = full_report[full_report["Variant"] == v]
        plt.plot(dfv["Episode"], dfv["Reward"], label=v)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("PPO Variant Comparison - Reward Progress")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
