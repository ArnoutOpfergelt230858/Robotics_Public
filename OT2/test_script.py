import numpy as np
from ot2_env_wrapper import OT2Env
import os
import wandb
import argparse
from stable_baselines3 import PPO
import gymnasium as gym
from wandb.integration.sb3 import WandbCallback
from clearml import Task

# Replace Pendulum-v1/YourName with your own project name (Folder/YourName, e.g. 2022-Y2B-RoboSuite/Michael)
task = Task.init(project_name='Mentor Group A/Group 1/ArnoutOpfergelt', # NB: Replace YourName with your own name
                    task_name='Experiment2')

#copy these lines exactly as they are
#setting the base docker image
task.set_base_docker('deanis/2023y2b-rl:latest')
#setting the task to run remotely on the default queue
task.execute_remotely(queue_name="default")


os.environ['WANDB_API_KEY'] = '83fb9634de2859207dcfe7a3d26409cf65ace208'

# Load your custom environment
env = OT2Env()

# Number of episodes
num_episodes = 50

for episode in range(num_episodes):
    obs = env.reset()
    done = False
    step = 0

    while not done:
        # Take a random action from the environment's action space
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Episode: {episode + 1}, Step: {step + 1}, Action: {action}, Reward: {reward}")

        step += 1
        if done:
            print(f"Episode finished after {step} steps. Info: {info}")
            break
