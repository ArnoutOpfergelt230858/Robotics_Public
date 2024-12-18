import os
import wandb
import argparse
from stable_baselines3 import PPO
import gymnasium as gym
from wandb.integration.sb3 import WandbCallback

from clearml import Task

# Replace Pendulum-v1/YourName with your own project name (Folder/YourName, e.g. 2022-Y2B-RoboSuite/Michael)
task = Task.init(project_name='Pendulum-v1/ArnoutOpfergelt', # NB: Replace YourName with your own name
                    task_name='Experiment2')

#copy these lines exactly as they are
#setting the base docker image
task.set_base_docker('deanis/2023y2b-rl:latest')
#setting the task to run remotely on the default queue
task.execute_remotely(queue_name="default")


os.environ['WANDB_API_KEY'] = '83fb9634de2859207dcfe7a3d26409cf65ace208'

env = gym.make('Pendulum-v1', g=9.81)

# Initialize wandb project
run = wandb.init(project="sb3_pendulum_demo", sync_tensorboard=True)
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=f"runs/{run.id}")

wandb_callback = WandbCallback(
    model_save_freq=1000,
    model_save_path=f"models/{run.id}",
    verbose=2,
)

# Training Loop
timesteps = 100000
os.makedirs(f"models/{run.id}", exist_ok=True)

for i in range(10):
    model.learn(total_timesteps=timesteps, 
                callback=wandb_callback, 
                progress_bar=True, 
                reset_num_timesteps=False, 
                tb_log_name=f"runs/{run.id}")
    model.save(f"models/{run.id}/{timesteps*(i+1)}")

# Finish Wandb run
wandb.finish()
