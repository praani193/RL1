#!/usr/bin/env python3
"""
SLM + PPO variants experiment.

Supports variants:
["original", "sched2", "kl_penalty", "adaptive_kl", "entropy_bonus", "dynamic_clip", "smooth_clip"]

Outputs:
- metrics CSV per run (metrics per epoch)
- generated sample outputs
- combined comparison report (CSV + PNG)

Note: This is a compact, educational implementation — suitable for small experiments.
"""

import os
import argparse
import math
import time
import random
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from transformers import GPT2Config, GPT2TokenizerFast, GPT2LMHeadModel
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

# ---------------------------
# Utilities & Toy Dataset
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

def make_toy_dataset(tokenizer, n_samples=1000, seq_len=16):
    """
    Create a small synthetic dataset of prompt -> target sequences.
    Each sample: prompt "Question: X" -> "Answer: Y"
    Replace with your dataset for real experiments.
    Returns list of prompts (strings) and target continuations (strings).
    """
    prompts, targets = [], []
    for i in range(n_samples):
        a = i % 50
        prompt = f"Question: What's {a} plus {a}?"
        answer = f" Answer: {a + a}."
        prompts.append(prompt)
        targets.append(answer)
    return prompts, targets

# ---------------------------
# Small LM (policy) creation
# ---------------------------
def create_small_gpt2(tokenizer, n_embd=128, n_layer=2, n_head=4):
    cfg = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=128,
        n_ctx=128,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
    )
    model = GPT2LMHeadModel(cfg)
    return model

# ---------------------------
# PPO Helper functions
# ---------------------------
def flatten_logits_and_compute_logprobs(model, input_ids, attention_mask, action_ids):
    """
    Compute logprobs of action_ids under the model conditioned on input_ids.
    input_ids: [B, T_prompt]
    action_ids: [B, T_gen] (the generated continuation tokens)
    We'll forward whole sequence (prompt + action) and compute token-wise logprobs
    for the action positions.
    """
    device = input_ids.device
    concat = torch.cat([input_ids, action_ids], dim=1)  # [B, T_prompt + T_gen]
    outputs = model(concat, attention_mask=torch.ones_like(concat))
    logits = outputs.logits  # [B, T, V]
    # we want logprobs at positions corresponding to action tokens
    # compute logprobs for tokens at those action positions
    log_probs = F.log_softmax(logits, dim=-1)  # [B, T, V]
    # action positions are last T_gen positions
    action_logits = log_probs[:, -action_ids.size(1):, :]  # [B, T_gen, V]
    # gather
    action_ids_expanded = action_ids.unsqueeze(-1)  # [B, T_gen, 1]
    token_log_probs = torch.gather(action_logits, dim=-1, index=action_ids_expanded).squeeze(-1)  # [B, T_gen]
    # return sum over tokens as sequence logprob
    seq_logprob = token_log_probs.sum(dim=1)  # [B]
    # also return mean per token logprob
    mean_token_logprob = token_log_probs.mean(dim=1)
    return seq_logprob, mean_token_logprob, token_log_probs  # seq_logprob for trajectory-level PPO

def compute_advantages(rewards, gamma=0.99, lam=0.95):
    """
    rewards: list/1D tensor of rewards per timestep in an episode.
    This is a simplified GAE where value function is zero (or we can use a separate critic).
    For simplicity, we treat return-to-go as advantage (works for short episodes).
    """
    R = 0.0
    adv = []
    for r in reversed(rewards):
        R = r + gamma * R
        adv.insert(0, R)
    adv = torch.tensor(adv, dtype=torch.float32)
    # normalize advantages
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv

# ---------------------------
# Reward function
# ---------------------------
def simple_reward_fn(generated_texts: List[str], targets: List[str]) -> List[float]:
    """
    Simple reward: token overlap fraction between generated and target.
    Replace with a learned reward model for real RLHF experiments.
    """
    rewards = []
    for gen, tgt in zip(generated_texts, targets):
        # basic reward: proportion of matching characters / exact match bonus
        match = sum(1 for a, b in zip(gen, tgt) if a == b)
        frac = match / max(1, len(tgt))
        bonus = 1.0 if gen.strip() == tgt.strip() else 0.0
        rewards.append(frac + bonus)
    return rewards

# ---------------------------
# PPO Trainer (supports variants)
# ---------------------------
class PPOTrainer:
    def __init__(self,
                 model: nn.Module,
                 tokenizer: GPT2TokenizerFast,
                 lr=1e-4,
                 clip_eps=0.2,
                 device=None,
                 variant="original",
                 scheduler_params=None,
                 kl_target=0.01,
                 kl_beta_init=0.2,
                 entropy_coef=0.01):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.clip_eps = clip_eps
        self.variant = variant
        self.scheduler_params = scheduler_params or {}
        self.kl_target = kl_target
        self.kl_beta = kl_beta_init
        self.entropy_coef = entropy_coef
        # track steps to use in sched2
        self.global_step = 0

    def generate_actions(self, prompts: List[str], max_gen_len=8, temperature=1.0):
        """
        Sample continuations from current policy.
        Returns action_ids tensor [B, T_gen] and generated_texts list.
        """
        self.model.eval()
        tok = self.tokenizer
        batch_input = tok(prompts, padding=True, return_tensors="pt").to(self.device)
        input_ids = batch_input["input_ids"]
        B = input_ids.size(0)
        generated = []
        action_ids_list = []
        with torch.no_grad():
            for i in range(B):
                seq = input_ids[i:i+1]
                cur = seq
                out_ids = []
                for _ in range(max_gen_len):
                    logits = self.model(cur).logits[:, -1, :] / temperature
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)  # sample
                    out_ids.append(next_token.item())
                    cur = torch.cat([cur, next_token], dim=1)
                action_ids_list.append(torch.tensor(out_ids, dtype=torch.long, device=self.device).unsqueeze(0))
                gen_text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
                generated.append(gen_text)
        action_ids = torch.cat(action_ids_list, dim=0)  # [B, T_gen]
        return input_ids, action_ids, generated

    def ppo_update(self, batch_prompts, batch_targets, epochs=4, gamma=0.99):
        """
        One PPO update on the batch.
        Steps:
         - sample actions from current policy (we already did in collect)
         - compute old logprobs (from old policy) — but in this simple impl we compute them before update
         - compute rewards -> advantages
         - perform several gradient steps with PPO loss depending on variant
        """
        tok = self.tokenizer
        B = len(batch_prompts)
        # Collect rollouts (sample under current policy)
        input_ids, action_ids, gen_texts = self.generate_actions(batch_prompts)
        # compute rewards
        rewards = simple_reward_fn(gen_texts, batch_targets)
        advantages = compute_advantages(rewards, gamma=gamma)  # [T] where T=B (one timestep per sample)
        advantages = advantages.to(self.device)

        # compute old logprobs under current policy (we'll treat these as pi_old for PPO)
        self.model.eval()
        with torch.no_grad():
            old_seq_logprob, old_mean_logprob, _ = flatten_logits_and_compute_logprobs(self.model, input_ids, None, action_ids)
        old_seq_logprob = old_seq_logprob.detach()

        # For reporting: compute KL between old policy and current policy during updates
        # We'll compute approximate KL by generating logits and comparing distributions token-wise.
        metrics = {"avg_reward": float(sum(rewards)/len(rewards)), "batch_size": B}

        # Perform update epochs
        for epoch in range(epochs):
            self.model.train()
            # forward to compute new logprobs
            new_seq_logprob, new_mean_logprob, token_log_probs = flatten_logits_and_compute_logprobs(self.model, input_ids, None, action_ids)
            # ratio
            ratio = (new_seq_logprob - old_seq_logprob).exp().clamp(1e-8, 1e8)  # [B]

            # surrogate losses
            # using sequence-level advantages
            adv = advantages

            # Standard clipped surrogate
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
            ppo_clip_loss = -torch.mean(torch.min(surr1, surr2))

            # KL between old and new: approximate using mean token-wise logprobs difference
            # This approximation: E_old[log pi_old - log pi_new]
            approx_kl = torch.mean(old_seq_logprob - new_seq_logprob).item()

            # Entropy (avg token entropy)
            # approximate using token_log_probs (we have log probs of chosen tokens only) -> can't compute full entropy cheaply here.
            # Instead compute approximate entropy from model logits by sampling a batch of tokens.
            # For simplicity, compute token-level entropy from logits at generation time
            # (here we compute entropy for the last action token positions)
            with torch.no_grad():
                # rerun to get logits
                concat = torch.cat([input_ids, action_ids], dim=1)
                logits = self.model(concat).logits[:, -action_ids.size(1):, :]  # [B, T_gen, V]
                logp = F.log_softmax(logits, dim=-1)
                ent = -(logp * logp.exp()).sum(dim=-1).mean().item()

            # Base loss
            loss = ppo_clip_loss

            # Apply variant-specific modifications
            if self.variant == "original":
                # nothing else
                info = {"ppo_clip_loss": ppo_clip_loss.item()}
            elif self.variant == "sched2":
                # sched2: scheduled clip eps that decays with global_step or uses a cyclical schedule
                base_eps = self.clip_eps
                # example schedule: shrink eps linearly then ramp up slightly
                step = self.global_step
                max_steps = self.scheduler_params.get("max_steps", 10000)
                eps = base_eps * max(0.1, 1.0 - step / max_steps)  # decays toward 0.1*base
                # optionally add small dynamic KL penalty for large KL
                kl_penalty_coeff = self.scheduler_params.get("kl_penalty_coeff", 0.0)
                clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * adv
                loss = -torch.mean(torch.min(ratio * adv, clipped))
                if kl_penalty_coeff > 0:
                    loss = loss + kl_penalty_coeff * approx_kl
                info = {"eps": eps, "approx_kl": approx_kl}
            elif self.variant == "kl_penalty":
                beta = self.scheduler_params.get("beta", 0.2)
                loss = ppo_clip_loss + beta * approx_kl
                info = {"beta": beta, "approx_kl": approx_kl}
            elif self.variant == "adaptive_kl":
                # increase beta if KL > target, else decrease
                target = self.kl_target
                beta = self.kl_beta
                # update beta multiplicatively based on KL (common practice)
                if approx_kl > target * 1.5:
                    beta *= 1.5
                elif approx_kl < target / 1.5:
                    beta *= 0.67
                beta = max(1e-4, min(beta, 1000.0))
                self.kl_beta = beta
                loss = ppo_clip_loss + beta * approx_kl
                info = {"beta": beta, "approx_kl": approx_kl}
            elif self.variant == "entropy_bonus":
                ent_coef = self.entropy_coef
                loss = ppo_clip_loss - ent_coef * ent
                info = {"entropy": ent, "ent_coef": ent_coef}
            elif self.variant == "dynamic_clip":
                # use KL to adapt clip epsilon per step
                base = self.clip_eps
                eps = base * (1.0 + approx_kl * 10.0)  # if KL large, widen clip
                eps = min(eps, 1.0)
                clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * adv
                loss = -torch.mean(torch.min(ratio * adv, clipped))
                info = {"eps": eps, "approx_kl": approx_kl}
            elif self.variant == "smooth_clip":
                # instead of min clip, apply L2 penalty when ratio outside [1-eps,1+eps]
                eps = self.clip_eps
                inside = ((ratio >= (1 - eps)) & (ratio <= (1 + eps))).float()
                outside = 1.0 - inside
                # penalty magnitude proportional to squared deviation
                l2_penalty = torch.mean(outside * (ratio - ratio.clamp(1 - eps, 1 + eps)) ** 2)
                smooth_coeff = self.scheduler_params.get("smooth_coeff", 10.0)
                loss = ppo_clip_loss + smooth_coeff * l2_penalty
                info = {"smooth_coeff": smooth_coeff, "l2_penalty": l2_penalty.item()}
            else:
                info = {}

            # Optional: KL penalty for stability
            if self.variant in ("kl_penalty", "adaptive_kl") and False:
                # handled above
                pass

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            self.global_step += 1

        metrics.update({"approx_kl": approx_kl, "entropy": ent, "ppo_loss": ppo_clip_loss.item()})
        metrics.update(info)
        return metrics, gen_texts

# ---------------------------
# Training loop + Experiment orchestration
# ---------------------------
def run_experiment(variant: str,
                   output_dir: str,
                   num_epochs=20,
                   batch_size=32,
                   device=None,
                   toy_samples=512):
    os.makedirs(output_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run_experiment] variant={variant} device={device} out={output_dir}")

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model = create_small_gpt2(tokenizer)
    model.resize_token_embeddings(len(tokenizer))

    # dataset
    prompts, targets = make_toy_dataset(tokenizer, n_samples=toy_samples)
    trainer = PPOTrainer(model,
                         tokenizer,
                         lr=2e-4,
                         clip_eps=0.2,
                         variant=variant,
                         scheduler_params={"max_steps": num_epochs * (toy_samples // batch_size), "beta": 0.2, "kl_penalty_coeff": 0.0, "smooth_coeff": 40.0},
                         kl_target=0.01,
                         kl_beta_init=0.2,
                         entropy_coef=0.01)

    metrics_rows = []
    samples_out_file = os.path.join(output_dir, "samples.txt")
    with open(samples_out_file, "w", encoding="utf-8") as sf:
        sf.write(f"Variant: {variant}\n\n")

    n_batches = max(1, len(prompts) // batch_size)
    for epoch in range(num_epochs):
        epoch_metrics = {"epoch": epoch}
        epoch_rewards = []
        epoch_kls = []
        epoch_ent = []
        epoch_losses = []
        pbar = tqdm(range(n_batches), desc=f"Epoch {epoch}/{num_epochs} ({variant})")
        for b in pbar:
            # sample a batch
            idxs = random.sample(range(len(prompts)), k=batch_size)
            batch_prompts = [prompts[i] for i in idxs]
            batch_targets = [targets[i] for i in idxs]
            metrics, gen_texts = trainer.ppo_update(batch_prompts, batch_targets, epochs=1)
            epoch_rewards.append(metrics.get("avg_reward", 0.0))
            if "approx_kl" in metrics:
                epoch_kls.append(metrics.get("approx_kl", 0.0))
            epoch_ent.append(metrics.get("entropy", 0.0))
            epoch_losses.append(metrics.get("ppo_loss", 0.0))
            pbar.set_postfix({"r": sum(epoch_rewards)/len(epoch_rewards), "kl": (sum(epoch_kls)/len(epoch_kls) if epoch_kls else 0.0)})
            # write samples occasionally
            if (b % max(1, n_batches // 5)) == 0:
                with open(samples_out_file, "a", encoding="utf-8") as sf:
                    sf.write(f"Epoch {epoch}, batch {b}\n")
                    for pr, gt in zip(batch_prompts[:3], gen_texts[:3]):
                        sf.write(f"PROMPT: {pr}\nGENERATED: {gt}\n\n")

        epoch_metrics["avg_reward"] = float(sum(epoch_rewards) / len(epoch_rewards))
        epoch_metrics["avg_kl"] = float(sum(epoch_kls) / len(epoch_kls)) if epoch_kls else 0.0
        epoch_metrics["avg_entropy"] = float(sum(epoch_ent) / len(epoch_ent)) if epoch_ent else 0.0
        epoch_metrics["avg_loss"] = float(sum(epoch_losses) / len(epoch_losses)) if epoch_losses else 0.0
        print(f"[epoch {epoch}] reward={epoch_metrics['avg_reward']:.4f} kl={epoch_metrics['avg_kl']:.6f} ent={epoch_metrics['avg_entropy']:.4f}")
        metrics_rows.append(epoch_metrics)

        # save interim model occasionally
        torch.save(trainer.model.state_dict(), os.path.join(output_dir, f"model_epoch_{epoch}.pt"))

    # save metrics CSV
    df = pd.DataFrame(metrics_rows)
    csv_path = os.path.join(output_dir, "metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"[run_experiment] finished. metrics -> {csv_path}. samples -> {samples_out_file}")
    return csv_path, samples_out_file

# ---------------------------
# Reporting
# ---------------------------
def combine_and_plot(results_root: str, variants: List[str], save_path="comparison.png"):
    """
    Look for results/<variant>/metrics.csv and combine them into a comparison plot/table.
    """
    rows = []
    for v in variants:
        csv_path = os.path.join(results_root, v, "metrics.csv")
        if not os.path.exists(csv_path):
            print(f"[warn] missing {csv_path}, skipping")
            continue
        df = pd.read_csv(csv_path)
        # summary stats
        mean_reward = df["avg_reward"].mean()
        final_reward = df["avg_reward"].iloc[-1]
        mean_kl = df["avg_kl"].mean() if "avg_kl" in df.columns else 0.0
        mean_ent = df["avg_entropy"].mean() if "avg_entropy" in df.columns else 0.0
        mean_loss = df["avg_loss"].mean() if "avg_loss" in df.columns else 0.0
        rows.append({"variant": v, "mean_reward": mean_reward, "final_reward": final_reward, "mean_kl": mean_kl, "mean_entropy": mean_ent, "mean_loss": mean_loss})
    comp_df = pd.DataFrame(rows).sort_values("final_reward", ascending=False)
    os.makedirs(results_root, exist_ok=True)
    comp_csv = os.path.join(results_root, "comparison_summary.csv")
    comp_df.to_csv(comp_csv, index=False)
    print(f"[report] summary -> {comp_csv}")

    # plot comparison: final_reward and mean_kl
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    comp_df.plot.bar(x="variant", y="final_reward", ax=ax[0], legend=False, title="Final (epoch) reward by variant")
    comp_df.plot.bar(x="variant", y="mean_kl", ax=ax[1], legend=False, title="Mean KL by variant")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[report] plot saved -> {save_path}")
    return comp_csv, save_path

# ---------------------------
# CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="original", help="Which PPO variant to run")
    parser.add_argument("--output_dir", type=str, default="results/original", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--toy_samples", type=int, default=512)
    parser.add_argument("--report_only", action="store_true")
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--report_path", type=str, default="results/comparison.png")
    args = parser.parse_args()

    variants = ["original", "sched2", "kl_penalty", "adaptive_kl", "entropy_bonus", "dynamic_clip", "smooth_clip"]

    if args.report_only:
        combine_and_plot(args.results_root, variants, save_path=args.report_path)
        return

    if args.variant not in variants:
        raise ValueError(f"Variant must be one of {variants}")

    outdir = args.output_dir
    run_experiment(variant=args.variant,
                   output_dir=outdir,
                   num_epochs=args.num_epochs,
                   batch_size=args.batch_size,
                   toy_samples=args.toy_samples)

if __name__ == "__main__":
    main()
