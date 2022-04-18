import time 
import numpy as np
import random
from q_table import QTable
from IPython.display import clear_output
import matplotlib.pyplot as plt 
import time 
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # input is state and action. Output is the q-value.
        # or input is state (position and velocity) and output is the q-value for each action
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        # print("Forward")
        # print(list(self.parameters())[1].dtype)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DeepQLearning():
    gamma = 0.99
    epsilon = 0.01
    logging = False
    env = None

    def __init__(self, env):
        self.env = env
        self.model = Model()

        self.n_actions = self.env.action_space.n

    # def copy(self):
    #     return DeepQLearning(self.env)

    def get_input(self, state, action):
        input = np.concatenate((state, [action]))
        #print(torch.from_numpy(input).type())
        return torch.from_numpy(input).float()

    def get_q_tensor(self, state, action):
        return self.model.forward(self.get_input(state, action))
    
    def get_q(self, state, action):
        return self.get_q_tensor(state, action).detach().numpy()[0]

    def get_best_action(self, state):
        best_action = None
        highest_q = float('-inf')
        for action in range(self.n_actions):
            q = self.get_q(state, action)
            if q > highest_q:
                highest_q = q
                best_action = action
        if self.logging:
            print("highest_q is: ", highest_q)
            print("best action is: ", best_action)
        return best_action

    def get_best_q(self, state):
        highest_q = float('-inf')
        highest_q_tensor = None
        for action in range(self.n_actions):
            q_tensor = self.get_q_tensor(state, action)
            q = q_tensor.detach().numpy()[0]
            if q > highest_q:
                highest_q = q
                highest_q_tensor = q_tensor
        if self.logging:
            print("highest_q is: ", highest_q)
        return highest_q_tensor

    # def get_best_action(self, state):
    #     q_actions = self.forward(torch.tensor(state)).numpy()
    #     return np.argmax(q_actions)

    # def get_q_tensor(self, state, action):
    #     q_actions = self.forward(torch.tensor(state)).numpy()
    #     return q_actions[action]

    # def get_best_q(self, state):
    #     q_actions = self.forward(torch.tensor(state)).numpy()
    #     return np.max(q_actions)

    # alpha is in [0.1, 1.0] range. We start with 1.0 (learning everyrhing) and fade out to 0.1
    def get_alpha(self, step, num_training_steps):
        max_alpha = 1.0
        min_alpha = 0.1
        fraction_to_add = (num_training_steps-step)/num_training_steps
        fraction_value = fraction_to_add*(max_alpha-min_alpha)
        return min_alpha + fraction_value

    def train(self, num_training_steps):
        self.model.zero_grad()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        optimizer.zero_grad()

        for i in range(0, num_training_steps):
            state = self.env.reset()
            #alpha = self.get_alpha(i, num_training_steps)
            done = False
            epochs = 0
            while not done:
                if random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample() # Explore action space
                else:
                    action = self.get_best_action(state) # Exploit learned values

                next_state, reward, done, info = self.env.step(action)
                if next_state[0] >= 0.5:
                    done = True              

                # Need to update our Q-table.
                # compute loss and do a backward pass.

                # print("next_state is: ", next_state)
                # print("self.get_q_tensor(state, action).size() is: ", self.get_q_tensor(state, action).size())
                # print("self.get_best_q(next_state).size() is: ", self.get_best_q(next_state).size())

                optimizer.zero_grad()
                loss = torch.square(torch.tensor(reward) + self.get_best_q(next_state) - self.get_q_tensor(state, action))
                loss.backward()
                optimizer.step()

                state = next_state
                epochs += 1

            if i % 1 == 0:
                clear_output(wait=True)
                print(f"Episode: {i}")
                print(f"Epochs: {epochs}")
        
        torch.save(self.model, 'model.pt')

    def eval(self, episodes):
        self.model.load_state_dict(torch.load('model.pt'))

        for _ in range(episodes):
            state = self.env.reset()
            epochs = 0
            
            done = False
            
            while not done:
                action = self.get_best_action(state)
                state, reward, _, info = self.env.step(action)
                if state[0] >= 0.5:
                    done = True
                self.env.render()
                time.sleep(0.01)
                epochs += 1
            print("Finished after epochs: ", epochs)
                