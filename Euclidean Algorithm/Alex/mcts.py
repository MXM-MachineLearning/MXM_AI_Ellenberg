import math
import numpy as np
import random

ACTIONS = np.array([[1, 1], [0, -1]], dtype=np.float16), np.array([[0, 1], [1, 0]], dtype=np.float16)

def gen_start_config():
    return np.round(np.random.rand(1, 2) * 100)   # TODO make a better stochastic thing

def terminate(s):
    k_eps = 1e-4    # for float imprecision
    # terminate if one of the coordinates is 0
    return s[0] <= k_eps or s[1] <= k_eps


def UCT_fn(child, C):
    if child.visits == 0:
        return math.inf
    return child.subtree_value + 2 * C * math.sqrt(2 * math.log2(1 + child.parent.visits) / child.visits)

k_C = 1 / math.sqrt(2)

class Node:
    def __init__(self, parent, state):
        self.state = state
        self.parent = parent
        self.visits = 0
        self.children = [None, None]

        self.value = self.norm_value(self.state)
        self.subtree_value = 0
        self.is_terminal = self.terminal(self.state)

    @staticmethod
    def terminal(state, k_eps = 1e-4):
        if abs(state[0]) <= k_eps or abs(state[1]) <= k_eps:
            return True
        return False

    @staticmethod
    def norm_value(state):
        """
        :return: value normalized to be in [0,1] to satisfy Hoeffding ineq
        """
        if terminate(state):
            return math.inf
        return -np.min(np.array([state[0] / np.linalg.norm(state), state[1] / np.linalg.norm(state)]))


class MCTS:
    @staticmethod
    def is_leaf(node):
        return node.state[0] is None and node.state[1] is None


    @staticmethod
    def pick_child(node):
        # UCT
        t = []
        for i in node.children:
            if i is None:
                continue
            t.append(UCT_fn(i, k_C))
        return int(random.choice(np.argwhere(t == np.max(t))))

    @staticmethod
    def default_policy(node):
        """
        Keep randomly picking children until reach terminal node or leaf
        """
        while node is not None and node.is_terminal is False and not MCTS.is_leaf(node):
            temp = random.choice(node.children)
            if temp is None:
                break
            node = temp
        return node.value

    @staticmethod
    def default_search(node):
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
        state = node.state @ ACTIONS[i]
        node.children[i] = Node(node, state)
        return node.children[i].value

    @staticmethod
    def tree_policy(node, computations):
        while node.is_terminal is False:
            explored = MCTS.default_search(node)
            if explored is not True:
                return explored, computations + 1
            node = node.children[MCTS.pick_child(node)]
        return node, computations

    @staticmethod
    def backprop(root):
        """
        Recursively update node's value based off average raw values in subtree with root at node

        :param root: root of subtree
        :return: size of subtree, sum of values in subtree
        """
        n_child = root.visits
        v_child = root.value * root.visits
        for i in root.children:
            if i is None:
                continue
            i.visits += 1
            n_child += i.visits
            v_child += i.value * i.visits
            temp = MCTS.backprop(i)
            n_child += temp[0]
            v_child += temp[1]
        root.subtree_value = v_child / n_child
        return n_child, v_child


    @staticmethod
    def MCTS(node, comp_limit=10):
        """
        Shoutout "A Survey of MCTS Methods"
        :param node: the current state
        :param comp_limit: max number of possible future scenarios to compute (carries over)
        :return: index corresponding to best action
        """
        comps = 0
        while comps < comp_limit:
            c, comps = MCTS.tree_policy(node, comps)
            reward = MCTS.default_policy(node)
            MCTS.backprop(node)

        rv = MCTS.pick_child(node)
        # rv.parent = None
        return rv


start = Node(None, np.array((1, 1), dtype=np.float16))
decision = MCTS.MCTS(start, comp_limit=2)


# simple test
# print(decision)
# print(start.children[0].subtree_value)
# print(start.children[1].subtree_value)


def get_data(fname):
    x = np.array(np.loadtxt(fname, delimiter=","), dtype=np.float16)
    return x[:,0:2], x[:, -1]


# test_X, test_Y = get_data("test_data/test_simple.csv")
test_X, test_Y = get_data("test_data/test_simple.csv")
test_Y.reshape(-1, 1)

# test
k_cases = 10
correct = 0

for i in range(k_cases):
    if MCTS.MCTS(Node(None, test_X[i])) == test_Y[i]:
        correct += 1
    else:
        print(test_X[i])

acc = correct / k_cases
print("Accuracy:", acc)