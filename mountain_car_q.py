import gym
from q_learning import QLearning

env = gym.make('MountainCar-v0')

q_learning = QLearning(env)
#q_learning.train(50000)
q_learning.execute()

# Close the env
env.close()