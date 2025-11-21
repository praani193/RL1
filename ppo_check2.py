import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ============= Networks =============
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

# ============= Loss Variants =============
def L_original(ratio, adv, eps=0.2):
    return torch.min(ratio * adv, torch.clamp(ratio, 1 - eps, 1 + eps) * adv)

def L_sched2(I, L_max, L_min):
    return I * L_max + (1 - I) * (L_min + L_max)

def L_kl_penalty(ratio, adv, kl, beta=0.5):
    """KL-PPO"""
    return ratio * adv - beta * kl

def L_entropy_bonus(ratio, adv, entropy, beta=0.01):
    """Entropy regularized PPO"""
    return torch.min(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv) + beta * entropy

def L_dynamic_clip(ratio, adv, kl, base_eps=0.2):
    """Adaptive clipping based on KL"""
    eps = torch.clamp(base_eps * (1 + kl.mean()), 0.1, 0.4)
    return torch.min(ratio * adv, torch.clamp(ratio, 1 - eps, 1 + eps) * adv)

def L_mirror_descent(ratio, adv, kl, lr=0.1):
    """Mirror Descent PPO"""
    return adv * torch.exp(-kl / lr)

def L_gae(ratio, adv, lam=0.95):
    """Generalized Advantage PPO"""
    return torch.min(ratio * adv * lam, torch.clamp(ratio, 0.8, 1.2) * adv * lam)

# ============= Training Function =============
def train_variant(env_name, variant, total_episodes=200, warmup=50):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNet(state_dim, action_dim)
    value = ValueNet(state_dim)
    optimizer = optim.Adam(list(policy.parameters()) + list(value.parameters()), lr=3e-3)

    report = []
    alpha = 0.1

    for ep in range(total_episodes):
        state, _ = env.reset()
        log_probs, rewards, values, entropies = [], [], [], []
        done = False

        while not done:
            s = torch.tensor(state, dtype=torch.float32)
            probs = policy(s)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
            rewards.append(reward)
            values.append(value(s))
            state = next_state

        # GAE advantage calculation (simplified)
        R, returns = 0, []
        for r in reversed(rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        values = torch.cat(values)
        adv = returns - values.detach()
        ratio = torch.exp(torch.stack(log_probs) - torch.stack(log_probs).detach())

        # KL divergence approx
        kl = (torch.stack(log_probs) - torch.stack(log_probs).detach()) ** 2
        entropy = torch.stack(entropies).mean()

        # L_max and L_min (for sched2)
        L_max = torch.max(0.5 * (ratio * adv + torch.clamp(ratio, 0.8, 1.2) * adv),
                          alpha * torch.clamp(ratio, -1, 1) * adv)
        L_min = torch.min(0.5 * (ratio * adv + torch.clamp(ratio, 0.8, 1.2) * adv),
                          alpha * torch.clamp(ratio, -1, 1) * adv)

        I = 1.0 if ep < warmup else 0.0

        # Select variant
        if variant == "original": loss = -L_original(ratio, adv).mean()
        elif variant == "sched2": loss = -L_sched2(I, L_max, L_min).mean()
        elif variant == "ppo_kl": loss = -L_kl_penalty(ratio, adv, kl).mean()
        elif variant == "ppo_entropy": loss = -L_entropy_bonus(ratio, adv, entropy).mean()
        elif variant == "ppo_dynamic": loss = -L_dynamic_clip(ratio, adv, kl).mean()
        elif variant == "ppo_md": loss = -L_mirror_descent(ratio, adv, kl).mean()
        elif variant == "ppo_gae": loss = -L_gae(ratio, adv).mean()
        else: raise ValueError("Unknown variant")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        report.append({
            "Episode": ep,
            "Variant": variant,
            "Reward": sum(rewards),
            "Loss": loss.item(),
            "AdvMean": adv.mean().item(),
            "Entropy": entropy.item(),
            "KL": kl.mean().item(),
            "I(t)": I
        })

        print(f"[{variant}] Ep {ep+1} | Reward={sum(rewards):.1f} | Loss={loss.item():.4f} | KL={kl.mean():.4f}")

    env.close()
    return pd.DataFrame(report)

# ============= Run and Save Report =============
if __name__ == "__main__":
    variants = ["original", "sched2", "ppo_kl", "ppo_entropy", "ppo_dynamic", "ppo_md", "ppo_gae"]
    all_reports = []
    for v in variants:
        print(f"\n=== Running {v} ===")
        df = train_variant("CartPole-v1", v)
        all_reports.append(df)

    full_report = pd.concat(all_reports, ignore_index=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_report.to_csv(f"PPO_variants_full_report_{ts}.csv", index=False)

    # Summary
    with open(f"PPO_variants_summary_{ts}.txt", "w") as f:
        for v in variants:
            sub = full_report[full_report["Variant"] == v]
            f.write(f"{v}:\n")
            f.write(f"  Avg Reward: {sub['Reward'].mean():.2f}\n")
            f.write(f"  Final Reward: {sub['Reward'].iloc[-1]:.2f}\n")
            f.write(f"  Avg Loss: {sub['Loss'].mean():.4f}\n")
            f.write(f"  Avg KL: {sub['KL'].mean():.4f}\n\n")

    # Plot
    plt.figure(figsize=(10,6))
    for v in variants:
        sub = full_report[full_report["Variant"] == v]
        plt.plot(sub["Episode"], sub["Reward"], label=v)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("PPO Variants Comparison (CartPole-v1)")
    plt.legend()
    plt.grid(True)
    plt.show()
