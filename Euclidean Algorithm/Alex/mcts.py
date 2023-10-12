import random

import matplotlib.pyplot as plt

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
                + "; Subtree Value: " + str(self.subtree_value) + "; Visits:", str(self.visits))

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
        return 1 / np.min(np.abs(np.array(self.state)))

    def is_leaf(self):
        for i in self.state:
            if i is not None:
                return False
        return True


class MCTS:
    def __init__(self, actions, C, weight):
        self.actions = actions
        self.k_C = C
        self.k_weight = weight  # penalty on future reward estimations (per move in the future)

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
        node.children[i] = Node(node, state, len(self.actions))
        return node.children[i]

    def tree_policy(self, node, computations):
        while node.is_terminal is False:
            explored = self.default_search(node)
            if explored is not True:
                return explored, computations + 1
            # node = node.children[self.pick_child(node)]
            node = random.choice(node.children)
        return node, computations + 1

    def prop(self, node):
        """
        Backprop up using sum of rewards. Parent subtree value takes sum of child subtree values.

        :param node of subtree
        :return: size of subtree, sum of values in subtree
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
        self.prop(node.parent)

    def run(self, root, comp_limit=10):
        """
        Shoutout "A Survey of MCTS Methods"
        :param node: the current state
        :param comp_limit: max number of possible future scenarios to compute (carries over)
        :return: index corresponding to best action
        """
        if root.is_terminal:
            return True
        comps = 0
        while comps < comp_limit:
            node, comps = self.tree_policy(root, comps)
            self.prop(node)

        rv = self.pick_child(root)
        return rv


def get_data(fname):
    x = np.array(np.loadtxt(fname, delimiter=","), dtype=np.float16)
    return x[:,:-1], x[:,-1]


def plot_db(mcts, actions, comp_limit):
    k_dbound_size = 100
    X = np.linspace(2, k_dbound_size, k_dbound_size-1)
    Y = np.linspace(2, k_dbound_size, k_dbound_size-1)
    action_plot = [[] for i in actions]
    for i in X:
        for j in Y:
            result = mcts.run(Node(None, (i,j), len(actions)), comp_limit=comp_limit)
            action_plot[result].append((i,j))
    for i in range(len(action_plot)):
        action = np.array(action_plot[i])
        plt.scatter(action[:,0], action[:,1], color=("C"+str(i)), label=action)
    plt.show()


def test(x, y, C, weight=1., comp_limit=10, actions=(a_subtract, a_swap), zero_index=False, want_plot=True):
    correct = 0
    mcts = MCTS(actions, C, weight)
    guess_dist = [0] * len(actions)
    if zero_index:
        y = y - np.ones(len(y))
    for i in range(len(x)):
        rv = mcts.run(Node(None, x[i], len(actions)), comp_limit=comp_limit)
        if rv == y[i] or rv is True:
            correct += 1
        guess_dist[rv] += 1
        # if (i+1) % 100 == 0:
            # print("epoch", i+1, ":", correct / (i+1))

    if want_plot:
        # graphing decision boundary
        plot_db(mcts, actions, comp_limit)
    return correct / len(x), guess_dist


def run_test(data_name, actions, C, cases=100, lookahead=100, weight=1., zero_index=False):
    test_X, test_Y = get_data(data_name)
    test_Y.reshape(-1, 1)

    acc, guesses = test(test_X[:cases], test_Y[:cases],
                        C, weight, comp_limit=lookahead, actions=actions, zero_index=zero_index)
    print("Test Accuracy:", acc)
    print("Guess Distribution:", guesses)


dual_file = "test_data/test_simple.csv"
quad_file = "../Donald/four_step_euclidean/four_directions_test.csv"     # thanks, donald

k_C = 1 / math.sqrt(2)  # satisfies Hoeffding Ineq (Kocsis and Szepesvari)
k_cases = 2000

run_test(dual_file, [a_subtract, a_swap], k_C, k_cases, lookahead=1000)
# ~90% accuracy

run_test(quad_file, [a_plsy, a_suby, a_plsx, a_subx], k_C, k_cases, lookahead=1000, zero_index=True)
# 8% accuracy on Donald test csv
