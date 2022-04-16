import time 
import numpy as np
import random
from q_table import QTable
from IPython.display import clear_output
import matplotlib.pyplot as plt 
import time 
import numpy as np

class QLearning:
    gamma = 0.99
    epsilon = 0.01
    state_buckets = [100, 40]
    logging = False
    env = None

    def __init__(self, env):
        self.env = env

    # alpha is in [0.1, 1.0] range. We start with 1.0 (learning everyrhing) and fade out to 0.1
    def get_alpha(self, step, num_training_steps):
        max_alpha = 1.0
        min_alpha = 0.1
        fraction_to_add = (num_training_steps-step)/num_training_steps
        fraction_value = fraction_to_add*(max_alpha-min_alpha)
        return min_alpha + fraction_value

    def convert_state(self, state):
        converted_state = np.multiply(np.divide(np.subtract(state, self.env.observation_space.low), np.subtract(self.env.observation_space.high, self.env.observation_space.low)), self.state_buckets)
        return tuple(converted_state.astype(int))

    def train(self, num_training_steps):
        q_table = QTable(self.env.action_space.n)
        for i in range(0, num_training_steps):  
            state = self.env.reset()
            state = self.convert_state(state)
            alpha = self.get_alpha(i, num_training_steps)
            epochs, reward = 0, 0
            done = False
            
            while not done:
                if random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample() # Explore action space
                else:
                    action = q_table.get_best_action(state) # Exploit learned values

                next_state, reward, done, info = self.env.step(action)
                if next_state[0] >= 0.5:
                    reward = 0.0
                    done = True
                else:
                    done = False
                next_state = self.convert_state(next_state)

                # Let's add a reward for being closer to the goal state.
                #reward+=0.1*next_state[0]/state_buckets[0]

                old_value = q_table.get_q(state, action)
                next_max = q_table.get_best_value(next_state)

                # Should terminal state have better initial reward?
                new_value = (1 - alpha) * old_value + alpha * (reward + self.gamma * next_max)
                q_table.set_q(state, action, new_value)

                state = next_state
                epochs += 1
                
            if i % 10 == 0:
                clear_output(wait=True)
                print(f"Episode: {i}")
                print(f"Epochs: {epochs}")

        print("Training finished.\n")
        q_table.write()

    def execute(self, episodes):
        q_table = QTable(self.env.action_space.n)
        q_table.read()

        for _ in range(episodes):
            state = self.env.reset()
            epochs = 0
            
            done = False
            
            while not done:
                action = q_table.get_best_action(self.convert_state(state))
                state, reward, _, info = self.env.step(action)
                if state[0] >= 0.5:
                    done = True
                self.env.render()
                time.sleep(0.01)
                epochs += 1
            print("Finished after epochs: ", epochs)