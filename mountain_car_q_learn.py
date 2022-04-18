import gym
from deep_q_learning import DeepQLearning
import torch

env = gym.make('MountainCar-v0')

q_learning = DeepQLearning(env)
q_learning.train(1)

q_learning.eval(1)

# Close the env
env.close()