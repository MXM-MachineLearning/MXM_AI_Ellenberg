import math
import numpy as np
import torch

# actions

def a_identity(state):
    return state

def a_mod(state):
    if state[0] == 0:
        return state[0], state[1]
    return state[0], state[1] % state[0]

def a_swp(state, device="cpu"):
    return state @ torch.tensor([[0,1],[1,0]], dtype=torch.float).to(device)
    return torch.tensor(state[1], state[0], dtype=torch.float).to(device)


def a_plsy(state, device="cpu"):
    return state @ torch.tensor([[1, 1], [0, 1]],dtype=torch.float).to(device)


def a_suby(state, device="cpu"):
    return state @ torch.tensor([[1, -1], [0, 1]],dtype=torch.float).to(device)


def a_plsx(state, device="cpu"):
    return state @ torch.tensor([[1, 0], [1, 1]],dtype=torch.float).to(device)


def a_subx(state, device="cpu"):
    return state @ torch.tensor([[1, 0], [-1, 1]],dtype=torch.float).to(device)


def UCT_fn(child, C):
    if child.visits == 0:
        return math.inf
    return child.subtree_value + 2 * C * math.sqrt(2 * math.log2(child.parent.visits) / child.visits)

k_2actions = (a_suby, a_swp)
k_4actions = (a_plsy, a_suby, a_plsx, a_subx)


def oh_encode(vec, width):
    p_target = torch.zeros(vec.shape[0],width)
    p_target.scatter_(1, vec,1)
    return p_target

def terminal(state, k_eps=1e-3):
    for i in state.flatten():
        if abs(i) <= k_eps:
            return True
    return False

# correct action
def determine_action(state):
    angle = (math.atan2(state[1],state[0]) + math.pi) % math.pi 
    if angle < math.pi / 4:
        return 3
    if angle < math.pi / 2:
        return 1
    if angle < math.pi * 3/4:
        return 2
    return 0
