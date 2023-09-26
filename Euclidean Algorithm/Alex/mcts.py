import math
import numpy as np
import random

from util import *


class Node:
    def __init__(self, parent, state, n_children):
        self.state = np.array(state, dtype=np.int32)
        self.parent = parent
        self.visits = 0
        self.children = [None] * n_children

        self.value = self.norm_value(self.state)
        self.subtree_value = 0
        self.is_terminal = self.terminal(self.state)

    def __str__(self):
        return ("State: " + str(self.state) + "; Value: " + str(self.value)
                + "; Subtree Value: " + str(self.subtree_value))

    @staticmethod
    def terminal(state, k_eps=1e-4):
        for i in state:
            if abs(i) > k_eps:
                return False
        return True

    @staticmethod
    def norm_value(state):
        if Node.terminal(state):
            return math.inf
        x = np.min(np.abs(np.array(state)))
        return - x


class MCTS:
    def __init__(self, actions, k_C):
        self.actions = actions
        self.k_C = k_C

    def is_leaf(self, node):
        for i in node.state:
            if i is not None:
                return False
        return True

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
        while node is not None and node.is_terminal is False and not self.is_leaf(node):
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
        return node.children[i].value

    def tree_policy(self, node, computations):
        while node.is_terminal is False:
            explored = self.default_search(node)
            if explored is not True:
                return explored, computations + 1
            node = node.children[self.pick_child(node)]
        return node, computations

    def prop(self, root, weighted=True):
        """
        Recursively update node's value based off average raw values in subtree with root at node

        :param root: root of subtree
        :param weighted: True for take weighted average for value
        :return: size of subtree, sum of values in subtree
        """
        n_child = root.visits
        v_child = root.value * root.visits

        all_none = True
        for i in root.children:
            if i is None:
                continue
            all_none = False
            i.visits += 1
            n_child += i.visits
            dv = i.value
            if weighted:
                dv *= i.visits
            v_child += dv
            temp = self.prop(i)
            n_child += temp[0]
            v_child += temp[1]
        if all_none:
            root.subtree_value = 0
        else:
            root.subtree_value = v_child / (n_child + 1e-4)     # so root doesn't blow up
        return n_child, v_child

    def run(self, node, comp_limit=10):
        """
        Shoutout "A Survey of MCTS Methods"
        :param node: the current state
        :param comp_limit: max number of possible future scenarios to compute (carries over)
        :return: index corresponding to best action
        """
        comps = 0
        while comps < comp_limit:
            c, comps = self.tree_policy(node, comps)
            reward = self.default_policy(node)
            # node.visits += 1
            self.prop(node)

        rv = self.pick_child(node)
        # rv.parent = None
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

    simple_as = [a_mod, a_swap]

    acc = test(test_X[:cases], test_Y[:cases], C, comp_limit=lookahead, actions=simple_as)
    print("Simple Test Accuracy:", acc)


def test_quad(C, cases=100, lookahead=100):
    test_X, test_Y = get_data("../Donald/four_step_euclidean/four_directions_test.csv") # thanks, donald
    test_Y.reshape(-1, 1)

    quad_as = [a_plsy, a_suby, a_plsx, a_subx]

    acc = test(test_X[:cases], test_Y[:cases], C, comp_limit=lookahead, actions=quad_as, zero_index=True)
    print("Quad Test Accuracy:", acc)


k_C = 1 / math.sqrt(2)
k_cases = 50

test_simple(k_C, k_cases, lookahead=5)
# test_quad(k_C, k_cases)
