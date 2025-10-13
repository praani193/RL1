import gymnasium as gym

# Humanoid: full 3D biped
env = gym.make("Humanoid-v4", render_mode="human")

obs, info = env.reset()
done = False

for _ in range(1000):
    action = env.action_space.sample()  # random actions
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        obs, info = env.reset()

env.close()