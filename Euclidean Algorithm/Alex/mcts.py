import math
import numpy as np
import random

from util import *


class Node:
    def __init__(self, parent, state, n_children, use_inv=True):
        self.state = np.array(state, dtype=np.int32)
        self.parent = parent
        self.visits = 0
        self.children = [None] * n_children

        self.is_terminal = self.terminal()
        if use_inv:
            self.value = self.inv_axis_dist_value()
        else:
            self.value = self.axis_dist_value()
        self.subtree_value = 0

    def __str__(self):
        return ("State: " + str(self.state) + "; Value: " + str(self.value)
                + "; Subtree Value: " + str(self.subtree_value))

    def terminal(self, k_eps=1e-4):
        for i in self.state:
            if abs(i) <= k_eps:
                return True
        return False

    def axis_dist_value(self):
        if self.is_terminal:
            return math.inf
        x = np.min(np.abs(self.state))
        return - x

    def inv_axis_dist_value(self):
        if self.is_terminal:
            return math.inf
        return 1 / np.min(np.abs(np.array(self.state))) ** 2

    def is_leaf(self):
        for i in self.state:
            if i is not None:
                return False
        return True


class MCTS:
    def __init__(self, actions, k_C):
        self.actions = actions
        self.k_C = k_C

    def pick_child(self, node):
        # UCT
        t = []
        for i in node.children:
            if i is None:
                continue
            t.append(UCT_fn(i, self.k_C))
        return int(random.choice(np.squeeze(np.argwhere(t == np.max(t)), axis=1)))

    def default_policy(self, node):
        """
        Keep randomly picking children until reach terminal node or leaf
        """
        while node is not None and node.is_terminal is False and not node.is_leaf():
            temp = random.choice(node.children)
            if temp is None:
                break
            node = temp
        return node.value

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
        node.children[i] = Node(node, state, len(self.actions))
        return node.children[i]

    def tree_policy(self, node, computations):
        while node.is_terminal is False:
            explored = self.default_search(node)
            if explored is not True:
                return explored, computations + 1
            node = node.children[self.pick_child(node)]
        return node, computations

    def prop(self, node, k_weight=0.99):
        """
        Backprop up using sum of discounted rewards

        :param node of subtree
        :param weighted: True for take weighted average for value
        :return: size of subtree, sum of values in subtree
        """
        node.subtree_value = node.value
        if not node.is_leaf():
            for i in node.children:
                if i is None:
                    continue
                node.subtree_value += k_weight * i.subtree_value
        node.visits += 1
        if node.parent is None:
            return
        self.prop(node.parent)

    def run(self, root, comp_limit=10):
        """
        Shoutout "A Survey of MCTS Methods"
        :param node: the current state
        :param comp_limit: max number of possible future scenarios to compute (carries over)
        :return: index corresponding to best action
        """
        comps = 0
        while comps < comp_limit:
            node, comps = self.tree_policy(root, comps)
            self.prop(node)

        rv = self.pick_child(root)
        return rv

# simple test
# start = Node(None, np.array((1, 1), dtype=np.float16))
# decision = MCTS.MCTS(start, comp_limit=2)
# print(decision)
# print(start.children[0].subtree_value)
# print(start.children[1].subtree_value)


def get_data(fname):
    x = np.array(np.loadtxt(fname, delimiter=","), dtype=np.float16)
    return x[:,:-1], x[:,-1]


def test(x, y, C, comp_limit=10, actions=(a_mod, a_swap), zero_index=False):
    correct = 0
    mcts = MCTS(actions, C)
    if zero_index:
        y = y - np.ones(len(y))
    for i in range(len(x)):
        if mcts.run(Node(None, x[i], len(actions)), comp_limit=comp_limit) == y[i]:
            correct += 1
    return correct / len(x)


def test_simple(C, cases=100, lookahead=50):
    test_X, test_Y = get_data("test_data/test_simple.csv")
    test_Y.reshape(-1, 1)

    simple_as = [a_subtract, a_swap]

    acc = test(test_X[:cases], test_Y[:cases], C, comp_limit=lookahead, actions=simple_as)
    print("Simple Test Accuracy:", acc)


def test_quad(C, cases=100, lookahead=100):
    test_X, test_Y = get_data("../Donald/four_step_euclidean/four_directions_test.csv") # thanks, donald
    test_Y.reshape(-1, 1)

    quad_as = [a_plsy, a_suby, a_plsx, a_subx]

    acc = test(test_X[:cases], test_Y[:cases], C, comp_limit=lookahead, actions=quad_as, zero_index=True)
    print("Quad Test Accuracy:", acc)


k_C = 1 / math.sqrt(2)  # satisfies Hoeffding Ineq (Kocsis and Szepesvari)
k_cases = 50

test_simple(k_C, k_cases, lookahead=20)
# ~90% accuracy while using mod transform
# ~90% accuracy using subtract transform

# test_quad(k_C, k_cases)
# 10-20% accuracy with simple dist_to_axis
