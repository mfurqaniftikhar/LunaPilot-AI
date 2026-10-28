
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Create the training environment
env = make_vec_env("LunarLander-v3", n_envs=4)

# Load the existing trained model
model = PPO.load("models/lunapilot_ppo", env=env)

# Continue training from the existing policy
model.learn(
    total_timesteps=500_000,
    reset_num_timesteps=False,
    tb_log_name="PPO_continued"
)

# Save the improved model
model.save("models/lunapilot_ppo_v2")

env.close()

print("Continued training completed!")
print("Model saved: models/lunapilot_ppo_v2.zip")