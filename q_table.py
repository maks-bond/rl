from collections import defaultdict

from q_table_serialization import read_q_table, write_q_table

class QTable:
    logging = False
    n_actions = 0

    def __init__(self, n_actions):
        self.n_actions = n_actions

        def q_action_def_value():
            return 0

        def q_def_value():
            return defaultdict(q_action_def_value)

        # Define Q-table
        self.q_table = defaultdict(q_def_value)
    
    def get_q(self, state, action):
        if state not in self.q_table:
            return 0
        if action not in self.q_table[state]:
            return 0
        return self.q_table[state][action]

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

    def get_best_value(self, state):
        highest_q = float('-inf')
        for action in range(self.n_actions):
            q = self.get_q(state, action)
            if q > highest_q:
                highest_q = q
        return highest_q

    def set_q(self, state, action, q):
        self.q_table[state][action] = q

    def read(self):
        self.q_table = read_q_table()
    
    def write(self):
        write_q_table(self.q_table)
