
from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
import copy

class Node:
    def __init__(self, env, father):
        self.env = env
        self.father = father
        self.g = 0
        self.h = 0
        self.f = 0
        self.terminated = False

    def __eq__(self, other):
        return (self.env.get_state() == other.env.get_state())
    
    def set_f_a(self, h_weight):
        self.f = h_weight*self.h + (1 - h_weight)*self.g

    def set_f_epsilon(self):
        self.f = (0.5 * self.h) + (0.5 * self.g)


class Agent:
    def __init__(self) -> None:
        self.root_env = None
        self.OPEN = []
        self.CLOSE = []
        self.expanded = 0
        self.root = None

    def solution(self, node: Node, env: DragonBallEnv) -> List[int]:
        path = []
        total_cost = 0
        while node.father:
            loc = node.env.to_row_col(node.env.get_state())
            loc_p = node.env.to_row_col(node.father.env.get_state())
            if loc[0] > loc_p[0]:
                path.append(0)
            elif loc[1] > loc_p[1]:
                path.append(1)
            elif loc[0] < loc_p[0]:
                path.append(2)
            elif loc[1] < loc_p[1]:
                path.append(3)
            node = node.father
        fresh_env = copy.deepcopy(self.root_env)
        for action in path[::-1]:
            _, c, _ = fresh_env.step(action)
            total_cost += c
        return path[::-1], total_cost



    def b4Search(self, env: DragonBallEnv):
        self.root_env = copy.deepcopy(env)
        self.root_env.reset()
        self.OPEN = []
        self.CLOSE = []
        self.expanded = 0
        self.root = Node(self.root_env, None)

    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        raise NotImplementedError("Subclasses must implement the search method.")



class WeightedAStarAgent(Agent):
    def __init__(self) -> None:
        self.h_weight = 0
    
    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        self.b4Search(env)
        self.h_weight = h_weight
        self.root.h = heuristic_calculation(self.root_env, self.root)
        self.root.set_f_a(h_weight)
        self.OPEN.append(self.root)
        while self.OPEN:
            node = self.OPEN.pop(get_minimal_f_node(self.OPEN))
            if node not in self.CLOSE:
                self.expanded += 1
            self.CLOSE.append(node)
            if self.root_env.is_final_state(node.env.get_state()):
                    return (self.solution(node, env) + (self.expanded,))
            for action, (state, cost, terminated) in env.succ(node.env.get_state()).items():
                if state == None or node.env.get_state() == state:
                    continue
                new_env = copy.deepcopy(node.env)
                new_env.step(action)
                child = Node(new_env, node)
                self.set_node_state(child, cost, terminated)
                if not update_open(child, self.OPEN) and not update_close(child, self.OPEN, self.CLOSE):
                    self.OPEN.append(child)
                

    def set_node_state(self, node: Node, cost, terminated):
        if node.father is None:
            node.g = 0
        else:
            node.g = cost + node.father.g
        node.h = heuristic_calculation(node.env, node)
        node.set_f_a(self.h_weight)
        node.terminated = terminated
        


class BFSAgent(Agent):

    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        self.b4Search(env)
        self.OPEN.append(self.root)
        while self.OPEN:
            node = self.OPEN.pop(0)
            if node.env.get_state() not in self.CLOSE:
                self.expanded += 1
            self.CLOSE.append(node.env.get_state())
            for action, (state, cost, terminated) in node.env.succ(node.env.get_state()).items():
                if state == None or node.env.get_state() == state:
                    continue
                new_env = copy.deepcopy(node.env)
                new_env.step(action)
                child = Node(new_env, node)
                if child.env.get_state() not in self.CLOSE and child not in self.OPEN:
                    self.OPEN.append(child)
                if node.env.is_final_state(child.env.get_state()):
                    return (self.solution(child, env) + (self.expanded,))


class AStarEpsilonAgent(Agent):
    def __init__(self) -> None:
        self.epsilon = 0

    def search(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        self.b4Search(env)
        self.epsilon = epsilon
        self.root.h = heuristic_calculation(self.root.env, self.root)
        self.root.set_f_epsilon()
        self.OPEN.append(self.root)
        if env.is_final_state(self.root.env.get_state()):
            return (self.solution(self.root, env) + (self.expanded,))
        while self.OPEN:
            node = self.OPEN.pop(self.get_node_index_to_expanded())
            if self.root_env.is_final_state(node.env.get_state()):
                return (self.solution(node, env) + (self.expanded,))
            self.expanded += 1
            self.CLOSE.append(node)
            if self.root_env.is_final_state(node.env.get_state()):
                return (self.solution(node, env) + (self.expanded,))
            for action, (state, cost, terminated) in env.succ(node.env.get_state()).items():
                if state == None or node.env.get_state() == state:
                    continue
                    
                new_env = copy.deepcopy(node.env)
                new_env.step(action)
                child = Node(new_env, node)
                self.set_node_state(child, cost, terminated)
                if not update_open(child, self.OPEN) and not update_close(child, self.OPEN, self.CLOSE):
                    self.OPEN.append(child)

    def set_node_state(self, node: Node, cost, terminated):
        node.g = cost + node.father.g
        node.h = heuristic_calculation(node.env, node)
        node.set_f_epsilon()
        node.terminated = terminated

    def get_node_index_to_expanded(self):
        min_f_index = get_minimal_f_node(self.OPEN)
        min_value = self.OPEN[min_f_index].f * (1+self.epsilon)
        min_g = self.OPEN[min_f_index].g
        node_index_to_return = min_f_index
        curr_location = self.OPEN[node_index_to_return].env.get_state()[0]
        for i, e in enumerate(self.OPEN):
            if e.f <= min_value:
                if e.g < min_g:
                    min_g = e.g
                    curr_location = e.env.get_state()[0]
                    node_index_to_return = i
                if e.g == min_g and e.env.get_state()[0] < curr_location:
                    curr_location = e.env.get_state()[0]
                    node_index_to_return = i
        return node_index_to_return

def heuristic_calculation(env, node: Node) -> int:
    dragon_ball_1_state = env.d1
    dragon_ball_2_state = env.d2
    goal_state = env.get_goal_states()
    if node.env.get_state()[1] and node.env.get_state()[2]:
        min_val = calculate_manhattan_distance(env, node.env.get_state(), goal_state[0])
        for item in goal_state:
            min_val = min(min_val, calculate_manhattan_distance(env, item, node.env.get_state()))
        return min_val
    if node.env.get_state()[1]:
        return calculate_manhattan_distance(env, node.env.get_state(), dragon_ball_2_state)
    if node.env.get_state()[2]:
        return calculate_manhattan_distance(env, node.env.get_state(), dragon_ball_1_state)
    min_val = calculate_manhattan_distance(env, node.env.get_state(), dragon_ball_1_state)
    min_val = min(min_val, calculate_manhattan_distance(env, node.env.get_state(), dragon_ball_2_state))
    return min_val


def calculate_manhattan_distance(env, p, g):
    p_row, p_col = env.to_row_col(p)
    g_row, g_col = env.to_row_col(g)
    return abs(p_row - g_row) + abs(p_col - g_col)


def get_minimal_f_node(OPEN):
    min_val = OPEN[0].f
    index = 0
    curr_state = OPEN[0].env.get_state()
    for i, e in enumerate(OPEN):
        if e.f < min_val:
            min_val = e.f
            index = i
            curr_state = e.env.get_state()
        elif e.f == min_val and e.env.get_state()[0] < curr_state[0]:
            index = i
            curr_state = e.env.get_state()
    return index

def update_open(node: Node, OPEN):
    for i, e in enumerate(OPEN):
        if e.env.get_state() == node.env.get_state():
            if node.f < e.f:
                OPEN.pop(i)
                OPEN.append(node)
            return True
    return False

def update_close(node: Node, OPEN, CLOSE):
    for i, e in enumerate(CLOSE):
        if e.env.get_state() == node.env.get_state():
            if node.f < e.f:
                CLOSE.pop(i)
                OPEN.append(node)
            return True
    return False

