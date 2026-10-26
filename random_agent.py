
import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")

observation, info = env.reset()

print("Initial observation:", observation)
print("Observation shape:", observation.shape)

for step in range(1000):
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    print(
        f"Step: {step + 1}, "
        f"Action: {action}, "
        f"Reward: {reward:.2f}"
    )

    if terminated or truncated:
        print("Episode finished. Resetting environment...")
        observation, info = env.reset()

env.close()
