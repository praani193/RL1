from pathlib import Path
import sys
from functools import partial
import pickle
import shutil

import ray
import torch

from LHW.rl.algos.ppo import PPO
from LHW.rl.envs.wrappers import SymmetricEnv
from LHW.rl.utils.eval import EvaluateEnv

# ---------------- Environment Import ----------------
def import_env(env_name_str):
    if env_name_str == 'jvrc_walk':
        from envs.jvrc import JvrcWalkEnv as Env
    elif env_name_str == 'jvrc_step':
        from envs.jvrc import JvrcStepEnv as Env
    elif env_name_str == 'h1':
        from envs.h1 import H1Env as Env
    else:
        raise Exception("Check env name!")
    return Env

# ---------------- Training ----------------
def run_experiment_train(env_name="h1", logdir=None):
    print("\n--- Training Configuration ---")
    if logdir is None:
        logdir = Path("C:/Users/Lenovo/Desktop/RL_logs")
    logdir.mkdir(parents=True, exist_ok=True)

    # Create Args class with defaults
    class Args:
        pass

    args = Args()
    args.env = env_name
    args.logdir = logdir
    args.input_norm_steps = 100000
    args.n_itr = 20000
    args.lr = 1e-4
    args.eps = 1e-5
    args.lam = 0.95
    args.gamma = 0.99
    args.std_dev = 0.223
    args.learn_std = False
    args.entropy_coeff = 0.0
    args.clip = 0.2
    args.minibatch_size = 64
    args.epochs = 3
    args.use_gae = True
    args.num_procs = 12
    args.max_grad_norm = 0.05
    args.max_traj_len = 400
    args.no_mirror = False
    args.mirror_coeff = 0.4
    args.eval_freq = 100
    args.continued = None
    args.recurrent = False
    args.imitate = None
    args.imitate_coeff = 0.3
    args.yaml = None

    # Import the correct environment
    Env = import_env(args.env)
    env_fn = partial(Env, path_to_yaml=args.yaml)
    _env = env_fn()

    if not args.no_mirror:
        try:
            print("Wrapping in SymmetricEnv.")
            env_fn = partial(SymmetricEnv, env_fn,
                             mirrored_obs=_env.robot.mirrored_obs,
                             mirrored_act=_env.robot.mirrored_acts,
                             clock_inds=_env.robot.clock_inds)
        except AttributeError as e:
            print("Warning! Cannot use SymmetricEnv.", e)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=args.num_procs)

    # Save hyperparameters
    pkl_path = Path(args.logdir, "experiment.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(args, f)

    # Copy YAML config if exists
    if args.yaml:
        config_out_path = Path(args.logdir, "config.yaml")
        shutil.copyfile(args.yaml, config_out_path)

    # Start PPO training
    algo = PPO(env_fn, args)
    algo.train(env_fn, args.n_itr)


# ---------------- Evaluation ----------------
def run_experiment_eval(path_to_model=None):
    # Defaults
    if path_to_model is None:
        path_to_model = Path("C:/Users/Lenovo/Desktop/RL_logs")

    path_to_actor = Path(path_to_model, "actor.pt")
    path_to_critic = Path(path_to_model, "critic.pt")
    path_to_pkl = Path(path_to_model, "experiment.pkl")

    # Load experiment args
    run_args = pickle.load(open(path_to_pkl, "rb"))

    # Load trained policy and critic
    policy = torch.load(path_to_actor, weights_only=False)
    critic = torch.load(path_to_critic, weights_only=False)
    policy.eval()
    critic.eval()

    # Import the correct environment
    Env = import_env(run_args.env)
    yaml_path = Path(run_args.yaml) if run_args.yaml else None
    env = partial(Env, yaml_path)()

    # Run evaluation
    e = EvaluateEnv(env, policy, args=run_args)
    e.run()


# ---------------- Main ----------------
if __name__ == "__main__":
    print("Welcome to LHW RL framework! Do you want to 'train' or 'eval'?")
    mode = input().strip().lower()

    if mode == 'train':
        print("Enter environment name (default: h1):")
        env_name = input().strip() or "h1"
        print("Enter log directory (default: C:/Users/Lenovo/Desktop/RL_logs):")
        logdir_input = input().strip()
        logdir = Path(logdir_input) if logdir_input else None
        run_experiment_train(env_name, logdir)

    elif mode == 'eval':
        print("Enter path to model directory (default: C:/Users/Lenovo/Desktop/RL_logs):")
        path_input = input().strip()
        path_to_model = Path(path_input) if path_input else None
        run_experiment_eval(path_to_model)

    else:
        print("Invalid option. Please enter 'train' or 'eval'.")
