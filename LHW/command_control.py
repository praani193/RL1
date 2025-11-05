import os
import torch
import pickle
import re
import numpy as np

# === Load Actor, Critic, and Environment ===
def load_actor_critic(logdir):
    actor_path = os.path.join(logdir, "actor_19999.pt")
    critic_path = os.path.join(logdir, "critic_19999.pt")
    env_pkl = os.path.join(logdir, "experiment.pkl")

    with open(env_pkl, "rb") as f:
        env = pickle.load(f)

    actor = torch.load(actor_path, map_location=torch.device("cpu"))
    critic = torch.load(critic_path, map_location=torch.device("cpu"))

    actor.eval()
    critic.eval()

    print("✅ Loaded actor, critic, and environment from:", logdir)
    return env, actor, critic


# === Command Interpreter ===
def interpret_command(cmd):
    cmd = cmd.lower().strip()
    if "walk" in cmd:
        match = re.search(r"(\d+(\.\d+)?)\s*m", cmd)
        distance = float(match.group(1)) if match else 1.0
        return "walk", distance
    elif "turn left" in cmd:
        return "turn_left", None
    elif "turn right" in cmd:
        return "turn_right", None
    elif "quit" in cmd or "exit" in cmd:
        return "quit", None
    else:
        return "unknown", None


# === Action Performer ===
def perform_action(env, actor, command, param):
    obs = env.reset()
    done = False

    if command == "walk":
        steps = int(param * 100)  # scale meters to steps
        print(f"🚶 Walking forward for {param} meters ({steps} steps)...")
        for _ in range(steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = actor(obs_tensor).numpy()[0]
            obs, reward, done, info = env.step(action)
            try:
                env.render()
            except:
                pass
            if done:
                break

    elif command == "turn_left":
        print("↩️ Turning left...")
        # Implement specific rotation action here if env supports it

    elif command == "turn_right":
        print("↪️ Turning right...")
        # Implement specific rotation action here

    else:
        print("❓ Unknown command.")


# === Interactive Loop ===
if __name__ == "__main__":
    logdir = "."  # current folder where actor_19999.pt etc. are located
    env, actor, critic = load_actor_critic(logdir)

    print("\n🤖 Command-based robot controller ready!")
    print("Type something like: 'walk for 2 meters', 'turn left', or 'quit'.\n")

    while True:
        cmd = input(">>> ").strip()
        command, param = interpret_command(cmd)

        if command == "quit":
            print("👋 Exiting...")
            break
        else:
            perform_action(env, actor, command, param)
