import math
import numpy as np
import torch


def a_identity(state):
    return state


def a_inf(state):
    return 1e5, 1e5


def a_subtract(state):
    return state[0], state[1] - state[0]


def a_mod(state):
    if state[0] == 0:
        return state[0], state[1]
    return state[0], state[1] % state[0]

def a_swp(state):
    return state[1], state[0]


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

class MCTS:
    def __init__(self, actions, C, weight):
        self.actions = actions
        self.k_C = C
        self.k_weight = weight  # diminish future reward estimations (per move in the future)
        self.k_move_penalty = -1    # penalty for one move without finding a terminal state
        self.root = None

    def pick_child(self, node):
        # UCT
        t = []
        for i in node.children:
            if i is None:
                continue
            t.append(UCT_fn(i, self.k_C))
        return int(random.choice(np.squeeze(np.argwhere(t == np.max(t)), axis=1)))

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
        state = self.actions[i](node.state)
        node.children[i] = Node(node, state, len(self.actions), Node.completion_value(self.root), depth=node.depth+1)
        return node.children[i]

    def tree_policy(self, node, computations):
        while node.is_terminal is False:
            explored = self.default_search(node)
            if explored is not True:
                return explored, computations + 1
            node = node.children[self.pick_child(node)]
            # node = random.choice(node.children)
        return node, computations + 1

    def sum_prop(self, node):
        """
        Backprop up from a leaf using sum of rewards. Parent subtree value takes sum of child subtree values.

        :param node: of subtree
        """
        node.subtree_value = node.value
        if not node.is_leaf():
            for i in node.children:
                if i is None:
                    continue
                node.subtree_value += self.k_weight * i.subtree_value
        node.visits += 1
        if node.parent is None:
            return
        self.sum_prop(node.parent)

    def run(self, root, comp_limit=10):
        """
        Shoutout "A Survey of MCTS Methods"
        :param node: the current state
        :param comp_limit: max number of possible future scenarios to compute (carries over)
        :return: index corresponding to best action
        """
        self.root = root
        if root.is_terminal:
            return True
        comps = 0
        while comps < comp_limit:
            node, comps = self.tree_policy(self.root, comps)
            self.sum_prop(node)

        rv = self.pick_child(root)
        return rv
