
import gymnasium as gym
from stable_baselines3 import PPO

model = PPO.load("models/lunapilot_ppo")
env = gym.make("LunarLander-v3", render_mode="human")

episodes = 5
rewards = []

for episode in range(episodes):
    obs, info = env.reset()
    total_reward = 0
    terminated = False
    truncated = False
    steps = 0

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

    rewards.append(total_reward)

    if terminated:
        status = "Environment reached a terminal state"
    else:
        status = "Time limit reached"

    print(f"\nEpisode: {episode + 1}")
    print(f"Reward: {total_reward:.2f}")
    print(f"Steps: {steps}")
    print(f"Status: {status}")
    print(f"Final observation: {obs}")

print(f"\nAverage reward: {sum(rewards) / len(rewards):.2f}")

env.close()
