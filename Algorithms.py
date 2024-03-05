import numpy as np

from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
import heapdict
class Node:
    def __init__(self, env, parent):
        self.env = env
        self.parent = parent
        self.cost = 0
        self.h = 0
        self.g = 0

    # def __eq__(self, other):
    #     return (self.state[0] == other[0])
    
    #method to compare nodes
    def fValueComp(self, other):
        return self.cost + self.h < other.cost + other.h

    #method to compare nodes state
    def sameState (self, other):
        return self.state == other.state
    
    def DballUpdate(self, env: DragonBallEnv):
        d1 = (env.get_state()[0] == env.d1[0])
        d2 = (env.get_state()[0] == env.d2[0])

        if self.parent is not None:
            d1 = self.parent.state[1] or d1
            d2 = self.parent.state[2] or d2
        
        self.state = (self.state[0], d1, d2)

def solution(node: Node, env: DragonBallEnv) -> List[int]:
    path = []
    while node.parent:
        loc = node.env.to_row_col(node.env.get_state())
        loc_p = node.env.to_row_col(node.parent.env.get_state())
        if loc[0] > loc_p[0]:
            path.append(0)
        elif loc[1] > loc_p[1]:
            path.append(1)
        elif loc[0] < loc_p[0]:
            path.append(2)
        elif loc[1] < loc_p[1]:
            path.append(3)
        node = node.parent
    return path[::-1]

class BFSAgent():
    def __init__(self) -> None:
        root = None
        self.expanded = 0

    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        root_env = env
        root_env.reset()
        root = Node(root_env, None)
        self.expanded = 0
        OPEN = []
        CLOSED = []
        OPEN.append(root)
        while OPEN:
            n = OPEN.pop(0)
            print(n.env.get_state())
            if n.env.is_final_state(n.env.get_state()):
                return (solution(n, env), n.cost, self.expanded)
            CLOSED.append(n.env.get_state())
            self.expanded += 1
            for action in n.env.succ(n.env.get_state()):
                print(action)
                new_env = DragonBallEnv(env.desc)
                new_env.set_state(n.env.get_state())
                try:
                    new_state, new_cost, ter= new_env.step(action)
                except:
                    continue

                if new_state == None or new_state == n.env.get_state():
                    print("Invalid state")
                    continue
                child = Node(new_env, n)
                child.cost = n.cost + new_cost
                if(child.env.get_state() not in CLOSED and child not in OPEN):
                    OPEN.append(child)
        return ([], 0, self.expanded)


class WeightedAStarAgent():
    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        raise NotImplementedError



class AStarEpsilonAgent():
    def __init__(self) -> None:
        raise NotImplementedError
        
    def ssearch(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        raise NotImplementedError