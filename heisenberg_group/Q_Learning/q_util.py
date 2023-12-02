import random
import numpy as np
from collections import deque
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import math

def matrix_to_tuple(matrix):
    return (matrix[0][1], matrix[0][2], matrix[1][2])

# B is the inverse of A
A = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
B = np.array([[1, -1, 0], [0, 1, 0], [0, 0, 1]])

# C is the inverse of D
C = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]])
D = np.array([[1, 0, 0], [0, 1, -1], [0, 0, 1]])

actions = [A,B,C,D]

def df_to_Q_table(df):
    my_new_Q_table = defaultdict(lambda: 0)

    for index in range(len(df)):
        row = df.iloc[index]
        a = int(row['a'])
        b = int(row['b'])
        c = int(row['c'])
        my_new_Q_table[(a, c, b)] = row['value']

    return my_new_Q_table

def Q_table_to_df_and_save(my_Q_table):
    # this will create a dataframe from the Q_table and then save it
    data = {
        'a': [],
        'c': [],
        'b' : [],
        'value': []
    }
    Q_table_df = pd.DataFrame(data)
    for i in dict(my_Q_table):
        df2 = pd.DataFrame([[
            i[0],
            i[1],
            i[2],
            my_Q_table[i]
            ]],
            columns=['a', 'c', 'b', 'value'])
        Q_table_df = pd.concat([df2, Q_table_df])

    Q_table_df.to_csv("../Visualizations/Q_table_df.csv", index=False)

def do_we_loop(seen, matrix, agent):
    for i in range(20):
        seen.add(matrix_to_tuple(matrix))
        matrix = agent.matrix_to_next_matrix(matrix)
        if matrix_to_tuple(matrix) in seen: 
            return True
    return False


# together, A, B, C, and D generate the heisenberg group
class QLAgent:
    def __init__(self, qtable=defaultdict(lambda: 0), max_reward=10000, step_penalty=-10) -> None:
        self.qtable = qtable
        self.max_reward = max_reward
        self.step_penalty = step_penalty

    def epsilon_greedy_search(self, epsilon, state):
        if (random.random() < epsilon):
            # 0 is 'apply matrix A', 1 is 'apply matrix B'
            # 2 is 'apply matrix C', 3 is 'apply matrix D'
            return random.choice([0, 1, 2, 3])
        else:
            # get the best move for the current state
            return self.best_move_for_a_state(state=state)
        
    # I would like to return the best move for a given state
    def best_move_for_a_state(self, state):
        # vals = Q_table[(state[0][1], state[0][2], state[1][2])]

        apply_A = state @ A
        apply_B = state @ B
        apply_C = state @ C
        apply_D = state @ D

        vals = [0, 0, 0, 0]
        vals[0] = self.qtable[matrix_to_tuple(apply_A)]
        vals[1] = self.qtable[matrix_to_tuple(apply_B)]
        vals[2] = self.qtable[matrix_to_tuple(apply_C)]
        vals[3] = self.qtable[matrix_to_tuple(apply_D)]

        # if we haven't visited this state before, return a random choice of 0, 1, 2, or 3
        if vals==[0, 0, 0, 0]:
            return random.choice([0, 1, 2, 3])
        
        # if we have visited this state before, return the current best choice
        return np.argmax(vals)

    # over a given state, return the maximum value of the table for that state
    def max_a_prime(self, state):
        apply_A = state @ A
        apply_B = state @ B
        apply_C = state @ C
        apply_D = state @ D

        vals = [0, 0, 0, 0]
        vals[0] = self.qtable[matrix_to_tuple(apply_A)]
        vals[1] = self.qtable[matrix_to_tuple(apply_B)]
        vals[2] = self.qtable[matrix_to_tuple(apply_C)]
        vals[3] = self.qtable[matrix_to_tuple(apply_D)]
        
        return max(vals)

    def getReward(self, matrix):
        if (matrix==np.identity(3)).all():
            return self.max_reward
        else:
            return self.step_penalty

    def matrix_to_num_steps(self, cur_matrix, step_limit=50):
        index = 1
        for i in range(step_limit):
            if (cur_matrix==np.identity(3)).all():
                return i
            outputs = [0, 0, 0, 0]
            outputs[0] = self.qtable[matrix_to_tuple(cur_matrix@ A)]
            outputs[1] = self.qtable[matrix_to_tuple(cur_matrix@ B)]
            outputs[2] = self.qtable[matrix_to_tuple(cur_matrix@ C)]
            outputs[3] = self.qtable[matrix_to_tuple(cur_matrix@ D)]
            index = np.argmax(outputs)
            if index==0:
                cur_matrix = cur_matrix @ A
            elif index==1:
                cur_matrix = cur_matrix @ B
            elif index==2:
                cur_matrix = cur_matrix @ C
            elif index==3:
                cur_matrix = cur_matrix @ D
        return -1

    def test_Q_learning(self, row):
        cur_matrix = np.array([[1, int(row['val1']), int(row['val2'])], [0, 1, int(row['val3'])], [0, 0, 1]])
        return self.matrix_to_num_steps(cur_matrix)

    def get_next_step(self, oldObs, action):
        # action is always either 0, 1, 2, or 3
        next_state = oldObs @ actions[action]
        curReward = self.getReward(next_state)
        done = curReward==self.max_reward
        return (next_state, curReward, done)
    
    def update_table(self, obs, reward, discount, lr):
        if (obs == np.identity(3)).all():
            self.qtable[matrix_to_tuple(obs)] = self.max_reward
        else: 
            self.qtable[matrix_to_tuple(obs)] = (1-lr) * self.qtable[matrix_to_tuple(obs)] + (lr) * (reward + discount * (self.max_a_prime(obs)))

    def first_matrix_to_apply(self, row):
        outputs = [0, 0, 0, 0]
        cur_matrix = np.array([
            [1, int(row['val1']), int(row['val2'])], 
            [0, 1, int(row['val3'])], 
            [0, 0, 1]
            ])
        outputs[0] = self.qtable[matrix_to_tuple(cur_matrix@ A)]
        outputs[1] = self.qtable[matrix_to_tuple(cur_matrix@ B)]
        outputs[2] = self.qtable[matrix_to_tuple(cur_matrix@ C)]
        outputs[3] = self.qtable[matrix_to_tuple(cur_matrix@ D)]
        return np.argmax(outputs)
    
    def get_Q_value(self, row):
        try:
            return self.qtable[(int(row['val1']), 
        int(row['val2']), 
        int(row['val3']))]
        except KeyError as ke: 
            return 0
        
    def step(self, state):
        outputs = [0, 0, 0, 0]

        outputs[0] = self.qtable[matrix_to_tuple(state@ A)]
        outputs[1] = self.qtable[matrix_to_tuple(state@ B)]
        outputs[2] = self.qtable[matrix_to_tuple(state@ C)]
        outputs[3] = self.qtable[matrix_to_tuple(state@ D)]

        index = np.argmax(outputs)
        return state @ actions[index]