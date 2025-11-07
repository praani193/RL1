import os
import torch
import pickle
import re
import numpy as np

# === Load Actor, Critic, and Environment ===
def import_env(env_name_str):
    if env_name_str == 'jvrc_walk':
        from LHW.envs.jvrc import JvrcWalkEnv as Env
    elif env_name_str == 'jvrc_step':
        from LHW.envs.jvrc import JvrcStepEnv as Env
    else:
        raise ValueError(f"Unknown environment: {env_name_str}")
    return Env

def load_actor_critic(logdir):
    actor_path = os.path.join(logdir, "actor_2499.pt")
    critic_path = os.path.join(logdir, "critic_2499.pt")
    env_pkl = os.path.join(logdir, "experiment.pkl")

    # Load experiment config
    with open(env_pkl, "rb") as f:
        config = pickle.load(f)

    # Recreate environment
    Env = import_env(config.env_name if hasattr(config, 'env_name') else 'jvrc_walk')
    env = Env()

    # Load actor and critic
    actor = torch.load(actor_path, map_location=torch.device('cpu'))
    critic = torch.load(critic_path, map_location=torch.device('cpu'))
    actor.eval()
    critic.eval()

    print(" Loaded actor, critic, and environment:", Env.__name__)
    return env, actor, critic

# === Command Interpreter (multi-action) ===
def interpret_commands(cmd):
    cmd = cmd.lower().strip()
    # Split input by "and", "then", or commas
    parts = re.split(r'\band\b|\bthen\b|,', cmd)
    actions = []

    for part in parts:
        part = part.strip()
        if "walk" in part:
            match = re.search(r"(\d+(\.\d+)?)\s*m", part)
            distance = float(match.group(1)) if match else 1.0
            actions.append(("walk", distance))
        elif "turn left" in part:
            actions.append(("turn_left", None))
        elif "turn right" in part:
            actions.append(("turn_right", None))
        elif "quit" in part or "exit" in part:
            actions.append(("quit", None))
        else:
            actions.append(("unknown", None))

    return actions

# === Action Performer (with turning bias) ===
def perform_action(env, actor, command, param, obs):
    done = False

    if command == "walk":
        steps = int(param * 50)  # fewer steps for speed
        print(f" Walking forward for {param} meters ({steps} steps)...")
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

    elif command in ["turn_left", "turn_right"]:
        direction = "left" if command == "turn_left" else "right"
        print(f"🔄 Turning {direction}...")

        yaw_bias = -0.78 if direction == "left" else 0.78  # adjust as needed

        # smooth turning for 30 steps
        for _ in range(30):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = actor(obs_tensor).numpy()[0]

            # Apply yaw bias to correct action index for robot rotation
            action[0] += yaw_bias  # adjust if needed for actual yaw joint

            obs, reward, done, info = env.step(action)
            try:
                env.render()
            except:
                pass
            if done:
                break

    else:
        print(" Unknown command.")

    return obs  # return the updated obs

# === Interactive Loop ===
if __name__ == "__main__":

    logdir = "."  # current folder where actor_*.pt etc. are located
    env, actor, critic = load_actor_critic(logdir)
    obs = env.reset()
    print("\n Command-based robot controller ready!")
    print("Type commands like: 'walk 2 meters', 'turn left', or 'quit'.")
    print("You can also combine: 'turn right and walk 5 meters then turn left'\n")

    while True:
        cmd = input(">>> ").strip()
        commands = interpret_commands(cmd)

        for command, param in commands:
            if command == "quit":
                print(" Exiting...")
                exit(0)
            else:
                # Update obs after each action
                obs = perform_action(env, actor, command, param, obs)
