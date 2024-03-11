
from DragonBallEnv import DragonBallEnv
from typing import List, Tuple


Down = 0
Right = 1
Up = 2
Left = 3


class Node:
    def __init__(self, state, father):
        self.state = state
        self.father = father
        self.cost = 1
        self.g = 0
        self.h = 0
        self.f = 0
        self.terminated = False

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def get_father(self):
        return self.father

    def update_dragon_ball(self, env):
        location, d1, d2 = self.get_state()
        location_d1, d1_d1, d2_d1 = env.d1
        location_d2, d1_d2, d2_d2 = env.d2
        d1_new = d1
        d2_new = d2
        if location == location_d1 and not d1:
            d1_new = True
        if location == location_d2 and not d2:
            d2_new = True
        if self.father is not None:
            location_f, d1_f, d2_f = self.get_father().get_state()
            d1_new = d1_f or d1_new
            d2_new = d2_f or d2_new
        new_state = (location, d1_new, d2_new)
        self.set_state(new_state)

    def set_cost(self, cost):
        self.cost = cost

    def get_cost(self):
        return self.cost

    def set_g(self, g):
        self.g = g

    def get_g(self):
        return self.g

    def set_h(self, h):
        self.h = h

    def set_f_a(self, h_weight):
        self.f = h_weight*self.h + (1 - h_weight)*self.g

    def set_f_epsilon(self):
        self.f = (0.5 * self.h) + (0.5 * self.g)

    def get_f(self):
        return self.f

    def get_terminated(self):
        return self.terminated

    def set_terminated(self, terminated):
        self.terminated = terminated


def is_in_open(open_lst, item) -> bool:
    for e in open_lst:
        if e.get_state() == item.get_state():
            return True
    return False


def is_in_close(close_lst, item) -> bool:
    return item in close_lst


def solution(node, env, expanded):
    actions_lst = []
    total_cost = 0
    n_col = env.ncol
    while node.father is not None:
        father_node = node.father
        delta = father_node.get_state()[0] - node.get_state()[0]
        if delta == n_col:
            actions_lst.append(Up)
        elif delta == 1:
            actions_lst.append(Left)
        elif delta == -1:
            actions_lst.append(Right)
        elif delta == -n_col:
            actions_lst.append(Down)
        total_cost = total_cost + father_node.get_cost()
        node = node.father
    actions_lst.reverse()
    return actions_lst, total_cost, expanded


class BFSAgent_sal():
    def __init__(self) -> None:
        self.env = None
        self.actions = []
        self.expanded = 0

    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        OPEN = []
        CLOSE = []
        all_states = []
        self.expanded = 0
        node = Node(self.env.get_state(), None)
        OPEN.append(node)
        node.update_dragon_ball(self.env)
        if self.env.is_final_state(node.get_state()):
            return (OPEN, CLOSE, all_states)
            # return solution(node, self.env, self.expanded)
        while not len(OPEN) == 0:
            all_states.append(node.get_state())
            node = OPEN.pop(0)
            print(node.get_state())
            if not is_in_close(CLOSE, node.get_state()):
                self.expanded += 1
            CLOSE.append(node.get_state())
            for action, (state, cost, terminated) in self.env.succ(node.get_state()).items():
                if state == None or node.get_state() == state:
                    continue
                node.update_dragon_ball(self.env)
                new_state = (state[0], node.get_state()[1], node.get_state()[2])
                child = Node(new_state, node)
                child.update_dragon_ball(self.env)
                child.set_cost(cost)
                if not is_in_close(CLOSE, child.get_state()) and not is_in_open(OPEN, child):
                    OPEN.append(child)
                if self.env.is_final_state(child.get_state()):
                    return (OPEN, CLOSE, all_states)

                    # return solution(child, self.env, self.expanded)


class WeightedAStarAgent_sal():
    def __init__(self) -> None:
        self.env = None
        self.h_weight = None
        self.expanded = 0
        self.OPEN = []
        self.CLOSE = []

    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        self.h_weight = h_weight
        self.OPEN = []
        self.CLOSE = []
        self.expanded = 0
        node = Node(self.env.get_state(), None)
        node.set_h(heuristic_calculation(self.env, node))
        node.set_f_a(h_weight)
        node.update_dragon_ball(self.env)
        self.OPEN.append(node)
        if env.is_final_state(node.get_state()):
            return solution(node, self.env, self.expanded)
        while not len(self.OPEN) == 0:
            node = self.OPEN.pop(get_minimal_f_node(self.OPEN))
            if not is_in_close(self.CLOSE, node.get_state()):
                self.expanded += 1
            self.CLOSE.append(node)
            if self.env.is_final_state(node.get_state()):
                return solution(node, self.env, self.expanded)
            for action, (state, cost, terminated) in env.succ(node.get_state()).items():
                if state == None or node.get_state() == state:
                    continue
                node.update_dragon_ball(self.env)
                new_state = (state[0], node.get_state()[1], node.get_state()[2])
                child = Node(new_state, node)
                child.update_dragon_ball(self.env)
                self.set_node_state(child, cost, terminated)
                if not update_open(child, self.OPEN) and not update_close(child, self.OPEN, self.CLOSE):
                    self.OPEN.append(child)

    def set_node_state(self, node: Node, cost, terminated):
        if node.get_father() is None:
            node.set_g(0)
        else:
            node.set_g(cost + node.get_father().get_g())
        node.set_h(heuristic_calculation(self.env, node))
        node.set_f_a(self.h_weight)
        node.set_terminated(terminated)
        node.set_cost(cost)


def heuristic_calculation(env, node: Node) -> int:
    dragon_ball_1_state = env.d1
    dragon_ball_2_state = env.d2
    goal_state = env.get_goal_states()
    if node.get_state()[1] and node.get_state()[2]:
        min_val = calculate_manhattan_distance(env, node.get_state(), goal_state[0])
        for item in goal_state:
            min_val = min(min_val, calculate_manhattan_distance(env, item, node.get_state()))
        return min_val
    if node.get_state()[1]:
        return calculate_manhattan_distance(env, node.get_state(), dragon_ball_2_state)
    if node.get_state()[2]:
        return calculate_manhattan_distance(env, node.get_state(), dragon_ball_1_state)
    min_val = calculate_manhattan_distance(env, node.get_state(), dragon_ball_1_state)
    min_val = min(min_val, calculate_manhattan_distance(env, node.get_state(), dragon_ball_2_state))
    return min_val


def calculate_manhattan_distance(env, p, g):
    p_row, p_col = env.to_row_col(p)
    g_row, g_col = env.to_row_col(g)
    return abs(p_row - g_row) + abs(p_col - g_col)


def get_minimal_f_node(OPEN):
    min_val = OPEN[0].get_f()
    index = 0
    curr_state = OPEN[0].get_state()
    for i, e in enumerate(OPEN):
        if e.get_f() < min_val:
            min_val = e.get_f()
            index = i
            curr_state = e.get_state()
        elif e.get_f() == min_val and e.get_state()[0] < curr_state[0]:
            index = i
            curr_state = e.get_state()
    return index


def update_open(node: Node, OPEN):
    for i, e in enumerate(OPEN):
        if e.get_state() == node.get_state():
            if node.get_f() < e.get_f():
                OPEN.pop(i)
                OPEN.append(node)
            return True
    return False


def update_close(node: Node, OPEN, CLOSE):
    for i, e in enumerate(CLOSE):
        if e.get_state() == node.get_state():
            if node.get_f() < e.get_f():
                CLOSE.pop(i)
                OPEN.append(node)
            return True
    return False


class AStarEpsilonAgent_sal():
    def __init__(self) -> None:
        self.env = None
        self.expanded = 0
        self.epsilon = 0
        self.OPEN = []
        self.CLOSE = []

    def search(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        self.OPEN = []
        self.CLOSE = []
        self.expanded = 0
        self.epsilon = epsilon
        node = Node(self.env.get_state(), None)
        node.set_h(heuristic_calculation(self.env, node))
        node.set_f_epsilon()
        node.update_dragon_ball(self.env)
        self.OPEN.append(node)
        if env.is_final_state(node.get_state()):
            return solution(node, self.env, self.expanded)
        while not len(self.OPEN) == 0:
            node = self.OPEN.pop(self.get_node_index_to_expanded())
            if self.env.is_final_state(node.get_state()):
                return solution(node, self.env, self.expanded)
            self.expanded += 1
            self.CLOSE.append(node)
            if self.env.is_final_state(node.get_state()):
                return solution(node, self.env, self.expanded)
            for action, (state, cost, terminated) in env.succ(node.get_state()).items():
                if state == None or node.get_state() == state:
                    continue
                node.update_dragon_ball(self.env)
                new_state = (state[0], node.get_state()[1], node.get_state()[2])
                child = Node(new_state, node)
                child.update_dragon_ball(self.env)
                self.set_node_state(child, cost, terminated)
                if not update_open(child, self.OPEN) and not update_close(child, self.OPEN, self.CLOSE):
                    self.OPEN.append(child)

    def set_node_state(self, node: Node, cost, terminated):
        node.set_g(cost + node.get_father().get_g())
        node.set_h(heuristic_calculation(self.env, node))
        node.set_f_epsilon()
        node.set_terminated(terminated)
        node.set_cost(cost)

    def get_node_index_to_expanded(self):
        min_f_index = get_minimal_f_node(self.OPEN)
        min_value = self.OPEN[min_f_index].get_f() * (1+self.epsilon)
        min_g = self.OPEN[min_f_index].get_g()
        node_index_to_return = min_f_index
        curr_location = self.OPEN[node_index_to_return].get_state()[0]
        for i, e in enumerate(self.OPEN):
            if e.get_f() <= min_value:
                if e.get_g() < min_g:
                    min_g = e.get_g()
                    curr_location = e.get_state()[0]
                    node_index_to_return = i
                if e.get_g() == min_g and e.get_state()[0] < curr_location:
                    curr_location = e.get_state()[0]
                    node_index_to_return = i
        return node_index_to_return
