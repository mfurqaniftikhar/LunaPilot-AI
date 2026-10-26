
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Create the Lunar Lander training environment
env = make_vec_env("LunarLander-v3", n_envs=4)

# Create the PPO reinforcement learning agent
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=1024,
    batch_size=64,
    gamma=0.99,
    tensorboard_log="./logs/"
)

# Train the agent
model.learn(total_timesteps=100_000)

# Save the trained model
model.save("models/lunapilot_ppo")

env.close()

print("Training completed! Model saved successfully.")
