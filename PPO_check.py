import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

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
# PPO Core Logic
# ============================
def compute_ppo_loss(ratio, adv, eps=0.2):
    return torch.min(ratio * adv, torch.clamp(ratio, 1 - eps, 1 + eps) * adv)

def compute_sched1_loss(I, L_max, L_min):
    return I * L_max + (1 - I) * L_min

def compute_sched2_loss(I, L_max, L_min):
    return I * L_max + (1 - I) * (L_min + L_max)

# ============================
# Training Function
# ============================
def train_variant(env_name, variant, total_episodes=400, warmup=100):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNet(state_dim, action_dim)
    value = ValueNet(state_dim)
    optimizer = optim.Adam(list(policy.parameters()) + list(value.parameters()), lr=3e-3)

    all_rewards = []

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
        original = torch.min(ratio * adv, torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * adv)
        L_max = torch.max(0.5 * (ratio * adv + torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * adv),0.1*(torch.clamp(ratio, -1, 1 ) * adv))
        L_min = torch.min(0.5 * (ratio * adv + torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * adv),0.1*(torch.clamp(ratio, -1, 1 ) * adv))

        # Warmup schedule indicator
        I = 1.0 if episode < warmup else 0.0

        # Select variant
        if variant == "original":
            loss = -original.mean()
        elif variant == "sched1":
            loss = -compute_sched1_loss(I, L_max, L_min).mean()
        elif variant == "sched2":
            loss = -compute_sched2_loss(I, L_max, L_min).mean()
        else:
            raise ValueError("Unknown variant")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_rewards.append(sum(rewards))

    env.close()
    return all_rewards


# ============================
# Run All Variants
# ============================
if __name__ == "__main__":
    variants = ["original", "sched1", "sched2"]
    results = {}

    for v in variants:
        print(f"Running variant: {v}")
        rewards = train_variant("CartPole-v1", v)
        results[v] = rewards

    # Plot results
    plt.figure(figsize=(8,5))
    for v, r in results.items():
        plt.plot(r, label=v)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("PPO Variant Comparison (CartPole-v1)")
    plt.legend()
    plt.grid(True)
    plt.show()
