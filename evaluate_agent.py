
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

model = PPO.load("models/lunapilot_ppo_v2")
env = gym.make("LunarLander-v3")

episodes = 20
rewards = []
successful_landings = 0
crashes = 0
time_limits = 0

for episode in range(episodes):
    obs, info = env.reset()
    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    # LunarLander-v3 observation:
    # indices 6 and 7 represent landing-leg contact
    legs_contact = obs[6] == 1.0 and obs[7] == 1.0

    # A non-negative terminal reward plus both legs
    # touching the ground is our conservative success criterion.
    if terminated and legs_contact and total_reward >= 0:
        successful_landings += 1
        result = "Successful landing"
    elif terminated:
        crashes += 1
        result = "Terminal episode; inspect outcome"
    else:
        time_limits += 1
        result = "Time limit reached"

    rewards.append(total_reward)

    print(
        f"Episode {episode + 1}: "
        f"Reward={total_reward:.2f}, Result={result}"
    )

print("\n--- Evaluation Summary ---")
print(f"Average reward: {np.mean(rewards):.2f}")
print(f"Successful landings: {successful_landings}/{episodes}")
print(f"Other terminal episodes: {crashes}")
print(f"Time-limit endings: {time_limits}")
print(f"Landing rate: {successful_landings / episodes * 100:.1f}%")

env.close()
