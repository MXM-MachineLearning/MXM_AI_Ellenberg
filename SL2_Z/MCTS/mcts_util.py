import math
import numpy as np
import torch

from util import *

def get_action(action_mat, device="cpu"):
    return lambda a: a @ action_mat.float().to(device)

sl2z_gens = [torch.from_numpy(i) for i in k_sl2z_gen]

def sl2z_action_1(state, device="cpu"):
    return (state.view(2,2) @ sl2z_gens[0].float().to(device)).flatten()

def sl2z_action_2(state, device="cpu"):
    return (state.view(2,2) @ sl2z_gens[1].float().to(device)).flatten()

def sl2z_action_3(state, device="cpu"):
    return (state.view(2,2) @ sl2z_gens[2].float().to(device)).flatten()

def sl2z_action_4(state, device="cpu"):
    return (state.view(2,2) @ sl2z_gens[3].float().to(device)).flatten()


# def UCT_fn(child, C):
#     if child.visits == 0:
#         return math.inf
#     return child.subtree_value + 2 * C * math.sqrt(2 * math.log2(child.parent.visits) / child.visits)

k_mcts_actions = []

k_mcts_actions = [sl2z_action_1, sl2z_action_2, sl2z_action_3, sl2z_action_4] 
# k_mcts_actions_2 = [get_action(i) for i in k_sl2z_2s_gen]
# k_mcts_actions_3 = [get_action(i) for i in k_sl2z_3s_gen]

def oh_encode(vec, width):
    p_target = torch.zeros(vec.shape[0],width)
    p_target.scatter_(1, vec,1)
    return p_target

def terminal(state, k_eps=1e-3):
    for i in state.flatten():
        if abs(i) <= k_eps:
            return True
    return False


# copy pasted from deep_mcts.ipynb to satisfy ProcessPoolExecutor reqs
import threading
import torch.nn as nn
import random
import concurrent
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class Node:
    def __init__(self, parent, state, n_children, value, depth=0):
        self.state = state
        self.parent = parent
        self.visits = 0
        self.depth = depth
        self.children = [None] * n_children
        self.is_terminal = terminal(self.state)
        self.value = value
        self.subtree_value = torch.zeros(1).to(device)

        # for more on virtual loss/shared tree search, see "Parallel MCTS" by Chaslot et al. 
        # https://dke.maastrichtuniversity.nl/m.winands/documents/multithreadedMCTS2.pdf
        self.active_threads = 0
        self.lock = threading.Lock()


    def __str__(self):
        return ("State: " + str(self.state) + "; Value: " + str(self.value)
                + "; Subtree Value: " + str(self.subtree_value) + "; Visits:", str(self.visits))

    def is_leaf(self):
        for i in self.state:
            if i is not None:
                return False
        return True
    

def get_train_data(fname):
    x = np.loadtxt(fname, delimiter=",")
    return torch.tensor([x[:,2], x[:,2:]], dtype=torch.float)

def get_nonterm_rwd(mcts):
    return -torch.tensor(mcts.max_depth, dtype=torch.float)

def get_terminal_rwd(terminal_depth, start):
    return -terminal_depth + torch.linalg.norm(start)
    
def UCT_fn(child, C):
    uct = child.subtree_value + 2 * C * math.sqrt(2 * math.log2(child.parent.visits+1) / (child.visits+1))
    return uct - child.active_threads
    
class MCTS:
    def __init__(self, actions, C, weight, value_fn):
        self.actions = actions
        self.k_C = C
        self.k_weight = weight
        self.value_fn = value_fn
        self.max_depth = 0
        self.terminal = None    # None if no terminal state found; terminal Node if found
        self.root = None
        self.propagation_lock = threading.Lock()

    def pick_child(self, node):
        # UCT
        t = []
        for i in node.children:
            if i is None:
                continue
            t.append(UCT_fn(i, self.k_C))

        if len(t) == 0:
            return random.randint(0, len(node.children)-1)
        
        t = torch.tensor(t)

        rvs = torch.squeeze(torch.argwhere(t == torch.max(t)), axis=1)
        return int(random.choice(rvs))

    def default_search(self, node):
        """
        If node is fully explored (neither child is None), return True
        Otherwise, initialize value of a random unexplored next state

        :param node: node to search from
        :return: if fully explored, True. Else, value of the random unexplored next state
        """
        possible = []
        for i in range(len(node.children)):
            if node.children[i] is None:
                possible.append(i)
        if len(possible) == 0:
            return True

        i = random.choice(possible)
        # if unexplored or non-terminal, get value
        state = self.actions[i](node.state.flatten()).float().to(device).clamp(-1e38, 1e38) # hacky clamp to prevent overflow
        state = state.reshape(node.state.shape)
        child_val = self.value_fn(state) - node.depth - 1  # give penalty -1 for each additional step taken
        # child_val = self.value_fn(state)
        child_val = child_val.flatten()[0]

        # with node.lock:
        node.children[i] = Node(node, state, len(self.actions), value=child_val, depth=node.depth+1)

        # if new Node is terminal, take it as the tree's terminal if it takes less time to reach than current terminal
        if node.children[i].is_terminal:
        #     # if terminal, add reward of ||start_vec||_2^2
        #     node.children[i].value += torch.linalg.vector_norm(torch.square(self.root.state)).item()
            if self.terminal is None or node.children[i].depth < self.terminal.depth:
                self.terminal = node.children[i]

        if node.children[i].depth > self.max_depth:
            self.max_depth = node.children[i].depth
        return node.children[i]

    def tree_policy(self, node):
        prev = None
        while node.is_terminal is False:
            # add some virtual loss to the node for each thread that's exploring (released after back propagating)
            # with node.lock:
                # node.active_threads += 1

            explored = self.default_search(node)
            if explored is not True:
                return explored
            node = node.children[self.pick_child(node)]
            prev = node
            # node = random.choice(node.children)
        return prev
    
    def max_prop(self, node):
        # with node.lock:
        node.subtree_value = torch.nan
        for i in node.children:
            if i is None:
                continue
            if node.subtree_value is torch.nan:
                node.subtree_value = i.subtree_value
            else:
                node.subtree_value = max(node.subtree_value, i.subtree_value)
        node.visits += 1
        # node.active_threads -= 1
        if node.subtree_value is torch.nan:
            node.subtree_value = node.value
        if node.parent is None:
            return
        self.max_prop(node.parent)

    def mean_prop(self, node):
        """
        Backprop up from a leaf, where subtree_value is the average of a node's rewards and its subtree's rewards

        :param node: of subtree
        """
        with node.lock:
            node.subtree_value = torch.zeros(1).to(device)
            node.subtree_value += node.value + node.depth
            valid_children = 0
            if not node.is_leaf():
                for i in node.children:
                    if i is None:
                        continue
                    node.subtree_value += self.k_weight * (i.subtree_value + i.depth) 
                    valid_children += 1
            node.subtree_value /= valid_children + 1
            node.subtree_value -= node.depth
            node.visits += 1

            # remove virtual loss from node after thread done exploring its subtree
            node.active_threads -= 1

            if node.parent is None:
                return
        self.mean_prop(node.parent)

    def explore_once(self, number):
        node = self.tree_policy(self.root)
        with self.propagation_lock:
            # self.mean_prop(node)
            self.max_prop(node)
        return number

    def run(self, root, comp_limit=10, max_threads=5, nogil=False):
        """
        Shoutout "A Survey of MCTS Methods"
        :param root: the current state
        :param comp_limit: max number of possible future scenarios to compute (carries over)
        :return: index corresponding to best action
        """
        self.root = root
        if self.root.is_terminal:
            return True

        # if nogil:
        #     # spawn new thread for each computation
        #     with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        #         executor.map(self.explore_once, range(comp_limit))
        # # else:
        for i in range(comp_limit):
            self.explore_once(i)

        rv = self.pick_child(self.root)

        if False:
            print("root state:", root.state)
            print("child states: ",end="")
            for child in root.children:
                print(child.state, end=",")
            print()
        return rv
    
    def generate(self, init_state, actions):
        self.root = Node(None, init_state, n_children=len(self.actions), value=self.value_fn(init_state), depth=0)
        curr = self.root
        r_nodes = []
        for i in actions:
            newstate = self.actions[i](curr.state)
            n = Node(parent=curr,
                     state=newstate,
                     n_children=len(self.actions),
                     value=self.value_fn(newstate),
                     depth=curr.depth + 1)
            curr.children[i] = n
            curr = n            
            r_nodes.append(n)
        return r_nodes

        

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.v_loss_fn = torch.nn.MSELoss()
        self.p_loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, v_out, v_target):
        """
        Loss function designed to reward successful game completion while taking the least amount of steps possible
        Adapted from:
            - "Mastering the game of Go without human knowledge" (Silver et al)
            - "Discovering faster matrix multiplication algorithms with reinforcement learning" (Fawzi et al)

        :param v_out: the value outputed for the state by NN
        :param p_out: the policy outputed for the state by NN
        :param v_target: target value output
        :return: total loss
        """
        loss = self.v_loss_fn(v_out, v_target)
        # loss += self.p_loss_fn(p_out, p_target).sum()
        return loss


class ValueNN(nn.Module):
    def __init__(self, state_size):
        super(ValueNN, self).__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
    def forward(self, x):
#        x = self.flatten(x)
#        x = self.stack(x).flatten()
#        value = x[0:1].reshape((1,1))
#        return value
        return self.stack(x)


def one_batch(payload):
    start, actions, value_fn, comp_limit, k_C, k_thread_count_limit, nogil = payload
    mcts = MCTS(actions, C=k_C, weight=1, value_fn=value_fn)

    value = mcts.value_fn(start).flatten().to(device)

    start_node = Node(None, start, len(actions), value, 0)

    mcts.run(start_node, comp_limit=comp_limit, max_threads=k_thread_count_limit, nogil=nogil)


    # get attributes of game just played
    v_out = start_node.subtree_value.to(device)
    v_target = get_nonterm_rwd(mcts).to(device)
    if mcts.terminal is not None:
        v_target = get_terminal_rwd(mcts.terminal.depth, start).to(device)


    visits = []
    for i in start_node.children:
        if i is None:
            visits.append(0)
        else:
            visits.append(i.visits)
    visits = torch.tensor(visits, dtype=torch.float).to(device)
    p_sampled = visits / torch.sum(visits)
    return torch.cat((start.flatten().unsqueeze(0) ,v_target.flatten().unsqueeze(0), p_sampled.flatten().unsqueeze(0)),dim=1)
