import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================
# Policy and Value Networks
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
# PPO Loss Variants
# ============================
def compute_ppo_clip(ratio, adv, eps=0.2):
    return torch.min(ratio * adv, torch.clamp(ratio, 1 - eps, 1 + eps) * adv)

def compute_sched2_loss(I, L_max, L_min):
    return I * L_max + (1 - I) * (L_min + L_max)

def compute_kl_divergence(p_old, p_new):
    # KL(p_old || p_new)
    return (p_old * (p_old.log() - p_new.log())).sum(dim=-1).mean()

def compute_entropy_bonus(probs):
    return -(probs * probs.log()).sum(dim=-1).mean()


# ============================
# PPO Training Function
# ============================
def train_variant(env_name, variant, total_episodes=150, warmup=50):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNet(state_dim, action_dim)
    value = ValueNet(state_dim)
    optimizer = optim.Adam(list(policy.parameters()) + list(value.parameters()), lr=3e-3)

    report_data = []

    for episode in range(total_episodes):
        state, _ = env.reset()
        log_probs_old, rewards, values, states, actions = [], [], [], [], []
        done = False
        total_reward = 0

        # 1️⃣ Collect trajectory
        while not done:
            s = torch.tensor(state, dtype=torch.float32)
            probs = policy(s)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            total_reward += reward

            log_probs_old.append(dist.log_prob(action))
            rewards.append(reward)
            values.append(value(s))
            states.append(s)
            actions.append(action)
            state = next_state

        # 2️⃣ Compute advantages
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        values = torch.cat(values)
        adv = returns - values.detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # 3️⃣ New policy after collecting data
        log_probs_new, probs_old_list, probs_new_list = [], [], []
        for s, a in zip(states, actions):
            probs_old = policy(s).detach()
            probs_new = policy(s)
            dist_new = torch.distributions.Categorical(probs_new)
            log_probs_new.append(dist_new.log_prob(a))

            probs_old_list.append(probs_old)
            probs_new_list.append(probs_new)

        # 4️⃣ Compute PPO ratio properly
        log_probs_new = torch.stack(log_probs_new)
        log_probs_old = torch.stack(log_probs_old)
        ratio = torch.exp(log_probs_new - log_probs_old.detach())

        # Define L_max and L_min (for schedule)
        L_max = torch.max(ratio * adv, torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * adv)
        L_min = torch.min(ratio * adv, torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * adv)

        # Compute metrics
        p_old = torch.stack(probs_old_list)
        p_new = torch.stack(probs_new_list)
        kl = compute_kl_divergence(p_old, p_new)
        entropy = compute_entropy_bonus(p_new)

        # Warmup schedule indicator
        I = 1.0 if episode < warmup else 0.0

        # 5️⃣ Variant-specific loss
        if variant == "original":
            loss = -compute_ppo_clip(ratio, adv).mean()

        elif variant == "sched2":
            loss = -compute_sched2_loss(I, L_max, L_min).mean()

        elif variant == "kl_penalty":
            loss = -(compute_ppo_clip(ratio, adv).mean() - 0.5 * kl)

        elif variant == "adaptive_kl":
            beta = 1.0 + 0.1 * (episode / total_episodes)
            loss = -(compute_ppo_clip(ratio, adv).mean() - beta * kl)

        elif variant == "entropy_bonus":
            loss = -(compute_ppo_clip(ratio, adv).mean() + 0.01 * entropy)

        elif variant == "dynamic_clip":
            eps = 0.2 * (1 - episode / total_episodes)
            loss = -torch.min(ratio * adv, torch.clamp(ratio, 1 - eps, 1 + eps) * adv).mean()

        elif variant == "smooth_clip":
            smooth_ratio = torch.tanh(ratio - 1)
            loss = -torch.min(smooth_ratio * adv, torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * adv).mean()

        else:
            raise ValueError("Unknown variant name")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        report_data.append({
            "Episode": episode,
            "Variant": variant,
            "Reward": total_reward,
            "Loss": loss.item(),
            "Entropy": entropy.item(),
            "KL": kl.item()
        })

    env.close()
    return pd.DataFrame(report_data)


# ============================
# Run and Compare All Variants
# ============================
if __name__ == "__main__":
    variants = [
        "original",
        "sched2",
        "kl_penalty",
        "adaptive_kl",
        "entropy_bonus",
        "dynamic_clip",
        "smooth_clip"
    ]

    all_reports = []
    for v in variants:
        print(f"Running variant: {v}")
        df = train_variant("CartPole-v1", v)
        all_reports.append(df)

    final_report = pd.concat(all_reports, ignore_index=True)
    final_report.to_csv("ppo_variant_report.csv", index=False)
    print("\n✅ Saved full report to ppo_variant_report.csv")

    # === Summary Table ===
    summary = final_report.groupby("Variant")[["Reward", "Loss", "KL", "Entropy"]].mean()
    summary["Std_Reward"] = final_report.groupby("Variant")["Reward"].std()
    summary["Final_Score"] = (
        0.5*summary["Reward"] + summary["Loss"]+ 0.1 * summary["Entropy"]
    )
    summary = summary.sort_values("Final_Score", ascending=False)
    print("\n📊 PPO Variant Summary:\n")
    print(summary)

    best_variant = summary.index[0]
    print(f"\n🏆 Best variant: {best_variant}")
    print("Reason: Highest reward, low KL (stability), balanced entropy (exploration).")

    # === Plot Reward Curves ===
    plt.figure(figsize=(10, 6))
    for v in variants:
        rewards = final_report[final_report["Variant"] == v]["Reward"].rolling(10).mean()
        plt.plot(rewards, label=v)
    plt.xlabel("Episode")
    plt.ylabel("Reward (Moving Avg)")
    plt.title("PPO Variant Comparison on CartPole-v1")
    plt.legend()
    plt.grid(True)
    plt.show()
