import os
import wandb
import argparse
from stable_baselines3 import PPO
import gymnasium as gym
from wandb.integration.sb3 import WandbCallback
from clearml import Task
from OT2_env_wrapper import OT2Env

# Initialize ClearML Task
task = Task.init(
    project_name='Mentor Group A/Group1/ArnoutOpfergelt',  # Replace with your project path
    task_name='Experiment2'
)

task.set_base_docker('deanis/2023y2b-rl:latest')
task.execute_remotely(queue_name="default")

# Set WandB API key
os.environ['WANDB_API_KEY'] = '83fb9634de2859207dcfe7a3d26409cf65ace208'

# Initialize WandB project
run = wandb.init(project="sb3_ot2env_demo", sync_tensorboard=True)

# Create custom environment
env = OT2Env()

# Define PPO Hyperparameters
hyperparams = {
    'learning_rate': 0.001,
    'batch_size': 64,
    'n_steps': 2048,
    'epochs': 10,
}

# Initialize PPO Model
model = PPO(
    'MlpPolicy', 
    env, 
    verbose=1, 
    tensorboard_log=f"runs/{run.id}",
    **hyperparams
)

wandb_callback = WandbCallback(
    model_save_freq=1000,
    model_save_path=f"models/{run.id}",
    verbose=2,
)

# Training Loop
timesteps = 100000
os.makedirs(f"models/{run.id}", exist_ok=True)

for i in range(10):
    model.learn(
        total_timesteps=timesteps, 
        callback=wandb_callback, 
        progress_bar=True, 
        reset_num_timesteps=False, 
        tb_log_name=f"runs/{run.id}"
    )
    model.save(f"models/{run.id}/{timesteps*(i+1)}")

# Finish WandB run
wandb.finish()
