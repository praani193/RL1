# ppo_mujoco.py
import os
import time
import argparse
import pickle
from copy import deepcopy

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence

import mujoco
import numpy as np
import transforms3d as tf3
import ray

from pathlib import Path

# ----------------------------
# PPO Storage
# ----------------------------
class PPOBuffer:
    def __init__(self, state_dim, action_dim, gamma, lam, size):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.gamma = gamma
        self.lam = lam
        self.size = size

    def store(self, state, action, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def finish_path(self, last_val=0):
        # Compute returns with GAE
        returns = []
        gae = 0
        values = self.values + [last_val]
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + self.gamma * values[step + 1] * (1 - self.dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - self.dones[step]) * gae
            returns.insert(0, gae + values[step])
        self.returns = returns

    def get_data(self):
        return {
            'states': torch.tensor(np.array(self.states), dtype=torch.float),
            'actions': torch.tensor(np.array(self.actions), dtype=torch.float),
            'returns': torch.tensor(np.array(self.returns), dtype=torch.float),
            'values': torch.tensor(np.array(self.values), dtype=torch.float)
        }

# ----------------------------
# PPO Policy (Gaussian Feedforward)
# ----------------------------
class Gaussian_FF_Actor(torch.nn.Module):
    def __init__(self, state_dim, action_dim, init_std=0.1, learn_std=True):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_dim)
        )
        self.log_std = torch.nn.Parameter(torch.ones(action_dim) * np.log(init_std)) if learn_std else torch.ones(action_dim) * np.log(init_std)

    def forward(self, x, deterministic=False):
        mean = self.fc(x)
        std = self.log_std.exp()
        if deterministic:
            return mean
        return mean + std * torch.randn_like(mean)

    def distribution(self, x):
        mean = self.fc(x)
        std = self.log_std.exp()
        return torch.distributions.Normal(mean, std)

# ----------------------------
# PPO Critic
# ----------------------------
class FF_V(torch.nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.fc(x).squeeze(-1)

# ----------------------------
# PPO Algorithm
# ----------------------------
class PPO:
    def __init__(self, env_fn, args):
        self.gamma = args.gamma
        self.lam = args.lam
        self.lr = args.lr
        self.eps = args.eps
        self.ent_coeff = args.entropy_coeff
        self.clip = args.clip
        self.minibatch_size = args.minibatch_size
        self.epochs = args.epochs
        self.max_traj_len = args.max_traj_len
        self.recurrent = False  # simplified
        self.n_proc = args.num_procs
        self.grad_clip = args.max_grad_norm

        self.batch_size = self.n_proc * self.max_traj_len
        self.total_steps = 0
        self.iteration_count = 0

        obs_dim = env_fn().observation_space.shape[0]
        action_dim = env_fn().action_space.shape[0]

        self.policy = Gaussian_FF_Actor(obs_dim, action_dim, init_std=args.std_dev, learn_std=True)
        self.old_policy = deepcopy(self.policy)
        self.critic = FF_V(obs_dim)

    @staticmethod
    @ray.remote
    @torch.no_grad()
    def sample(env_fn, policy, critic, gamma, lam, iteration_count, max_steps, max_traj_len, deterministic):
        env = env_fn()
        memory = PPOBuffer(policy.fc[0].in_features, policy.fc[-1].out_features, gamma, lam, size=max_traj_len*2)
        memory_full = False
        while not memory_full:
            state = torch.tensor(env.reset(), dtype=torch.float)
            done = False
            traj_len = 0
            while not done and traj_len < max_traj_len:
                action = policy(state, deterministic=deterministic)
                value = critic(state)
                next_state, reward, done, info = env.step(action.numpy().copy())
                memory.store(state, action, reward, value, done)
                memory_full = (len(memory.states) >= max_steps)
                state = torch.tensor(next_state, dtype=torch.float)
                traj_len += 1
            memory.finish_path(last_val=(not done)*critic(state))
        return memory.get_data()

    def sample_parallel(self, env_fn, deterministic=False):
        ray.init(ignore_reinit_error=True)
        worker_args = (self.gamma, self.lam, self.iteration_count, self.batch_size//self.n_proc, self.max_traj_len, deterministic)
        workers = [self.sample.remote(env_fn, self.policy, self.critic, *worker_args) for _ in range(self.n_proc)]
        result = ray.get(workers)
        keys = result[0].keys()
        aggregated_data = {k: torch.cat([r[k] for r in result]) for k in keys}
        class Data: pass
        data = Data()
        for k, v in aggregated_data.items(): setattr(data, k, v)
        return data

    def update_actor_critic(self, obs_batch, action_batch, return_batch, advantage_batch, mask=1):
        pdf = self.policy.distribution(obs_batch)
        log_probs = pdf.log_prob(action_batch).sum(-1, keepdim=True)
        old_pdf = self.old_policy.distribution(obs_batch)
        old_log_probs = old_pdf.log_prob(action_batch).sum(-1, keepdim=True)
        ratio = (log_probs - old_log_probs).exp()
        cpi_loss = ratio * advantage_batch * mask
        clip_loss = ratio.clamp(1.0 - self.clip, 1.0 + self.clip) * advantage_batch * mask
        actor_loss = -torch.min(cpi_loss, clip_loss).mean()
        values = self.critic(obs_batch)
        critic_loss = F.mse_loss(return_batch, values)
        entropy_penalty = -(pdf.entropy() * mask).mean()
        self.actor_optimizer.zero_grad()
        (actor_loss + self.ent_coeff*entropy_penalty).backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()
        return actor_loss.item(), critic_loss.item(), entropy_penalty.item()

    def train(self, env_fn, n_itr):
        self.actor_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr, eps=self.eps)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr, eps=self.eps)
        for itr in range(n_itr):
            self.iteration_count = itr
            batch = self.sample_parallel(env_fn)
            advantages = batch.returns - batch.values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            self.old_policy.load_state_dict(self.policy.state_dict())
            for epoch in range(self.epochs):
                indices = np.arange(len(batch.states))
                np.random.shuffle(indices)
                for start in range(0, len(indices), self.minibatch_size):
                    end = start + self.minibatch_size
                    batch_idx = indices[start:end]
                    self.update_actor_critic(batch.states[batch_idx],
                                             batch.actions[batch_idx],
                                             batch.returns[batch_idx],
                                             advantages[batch_idx])
            print(f"Iteration {itr} done, total steps: {self.total_steps}")

# ----------------------------
# Evaluation
# ----------------------------
def print_reward(ep_rewards):
    mean_rewards = {k:0 for k in ep_rewards[0].keys()}
    for r in ep_rewards:
        for k, v in r.items():
            mean_rewards[k] += v
    for k in mean_rewards: mean_rewards[k]/=len(ep_rewards)
    print("Mean rewards per step:", mean_rewards)

def run_eval(env, policy, max_steps=2000, sync=False):
    observation = env.reset()
    viewer = getattr(env, 'viewer', None)
    ts = 0
    ep_rewards = []
    done = False
    while ts < max_steps:
        with torch.no_grad():
            action = policy(torch.tensor(observation, dtype=torch.float), deterministic=True).numpy()
        observation, _, done, info = env.step(action)
        if isinstance(info, (int, float)):
            ep_rewards.append({'reward': info})
        else:
            ep_rewards.append(info)
        if viewer:
            env.render()
        ts += 1
        if done:
            break
    print_reward(ep_rewards)

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="train PPO")
    parser.add_argument("--eval", type=str, help="path to trained actor.pt")
    parser.add_argument("--env", type=str, required=True, help="Env class path, e.g., LHW.envs.myenv:MyEnv")
    args = parser.parse_args()

    # dynamic import of environment
    module_path, class_name = args.env.split(":")
    env_module = __import__(module_path, fromlist=[class_name])
    env_cls = getattr(env_module, class_name)
    env = env_cls()

    if args.train:
        class Args:
            gamma=0.99; lam=0.95; lr=3e-4; eps=1e-5; entropy_coeff=0.01; clip=0.2
            minibatch_size=64; epochs=10; max_traj_len=200; num_procs=1; max_grad_norm=0.5
            std_dev=0.1
        ppo = PPO(lambda: env, Args())
        ppo.train(lambda: env, n_itr=10)
    elif args.eval:
        policy = torch.load(args.eval)
        policy.eval()
        run_eval(env, policy)
    else:
        print("Specify --train or --eval")

if __name__=='__main__':
    main()
