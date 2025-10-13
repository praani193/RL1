from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make("Humanoid-v4")

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_humanoid_tensorboard/")
model.learn(total_timesteps=5_000_000)   # needs millions of steps

model.save("ppo_humanoid")