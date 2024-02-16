import numpy as np
import random

def get_generators(mult):
    A = np.eye(2)
    B = A.copy()
    A[0][1] = mult
    B[1][0] = mult
    return np.array([A, B, np.linalg.inv(A), np.linalg.inv(B)])
    
k_sl2z_gen = get_generators(1)

k_sl2z_2s_gen = get_generators(2)

k_sl2z_3s_gen = get_generators(3)

def df_row_to_mat(row):
    return np.array([
        [int(row['val1']), int(row['val2'])], 
        [int(row['val3']), int(row['val4'])]
        ])

def matrix_to_tuple(matrix):
    return tuple(matrix.flatten())

def is_done(m) -> bool:
    return np.allclose(m, np.eye(m.shape[0]))

def tuple_to_matrix(tu):
    return np.array([[tuple[0], tuple[1]], [tuple[2], tuple[3]]])

def mod_2_is_identity(test_tuple):
    assert len(test_tuple)==4
    return (test_tuple[0] % 2 == 1 and 
            test_tuple[1] % 2 == 0 and 
            test_tuple[2] % 2 == 0 and 
            test_tuple[3] % 2 == 1)

def apply_action(m, action) -> np.array:
    return m @ action


class TabularQEnv:
    def __init__(self, actions, Q_table, rwd_fn, max_rwd) -> None:
        self.actions = actions
        self.Q_table = Q_table
        self.rwd_fn = rwd_fn
        self.max_rwd = max_rwd

    def get_next_possible_Qs(self, state):
        vals = [0,0,0,0]
        for i in range(len(self.actions)):
            vals[i] = self.Q_table[matrix_to_tuple(state @ self.actions[i])]
        return vals

    def __epsilon_greedy_search(self, Epsilon, state):
        if (random.random() < Epsilon):
            # 0 is 'apply matrix A', 1 is 'apply matrix B'
            # 2 is 'apply matrix C', 3 is 'apply matrix D'
            return random.choice(range(len(self.actions)))
        else:
            # get the best move for the current state
            return self.best_move(state=state)
        
    # I would like to return the best move for a given state
    def best_move(self, state):

        vals = self.get_next_possible_Qs(state)

        # if we haven't visited this state before, return a random choice of 0, 1, 2, or 3
        if vals==[0, 0, 0, 0]:
            return random.choice(range(len(self.actions)))
        
        # if we have visited this state before, return the current best choice
        return np.argmax(vals)

    # over a given state, return the maximum value of the table for that state
    def __max_a_prime(self, *args, **kwargs):
        return max(self.get_next_possible_Qs(*args, **kwargs))

    def __get_next_step(self, oldObs, action) -> tuple[np.array, int, bool]:
        next_state = oldObs @ self.actions[action]
        curReward = self.rwd_fn(next_state)
        done = curReward==self.max_rwd
        return (next_state, curReward, done)
    
    def step(self, lr, gamma, eps, state) -> tuple[np.array, int, bool]:
        # perform an epsilon greedy action 
        # Q(s, a) = (1-lr)Q(s, a) + (lr)(r + DISCOUNT_FACTOR(max a'(Q(s', a'))))
        action = self.__epsilon_greedy_search(Epsilon=eps, state=state)

        state,reward,done = self.__get_next_step(state, action)

        # if done:
        #     assert(1==2)
        
        self.Q_table[matrix_to_tuple(state)] = (1-lr) * self.Q_table[matrix_to_tuple(state)]  \
                                                + (lr) * (reward + gamma * (self.__max_a_prime(state)))
        return state, reward, done
    
    def play(self, state, max_steps=50) -> int:
        for i in range(max_steps):
            if is_done(state):
                return i
            state = apply_action(state, self.actions[self.best_move(state)])
        return -1