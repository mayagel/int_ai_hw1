import numpy as np

from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
import heapdict
class Node:
    def __init__(self, state, parent):
        self.state = state
        self.parent = parent
        self.cost = 1
        self.h = 0
        self.g = 0

    def __eq__(self, other):
        return (self.state[0] == other[0])
    
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
        
        # print(type(self.state))
        self.state = (self.state[0], d1, d2)

def solution(node: Node, env: DragonBallEnv) -> List[int]:
    path = []
    while node.parent:
        loc = env.to_row_col(node.state)
        loc_p = env.to_row_col(node.parent.state)
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
        self.env = None
        self.expanded = 0

    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        self.expanded = 0
        OPEN = []
        CLOSED = []
        root = Node(env.get_state(), None)
        OPEN.append(root)
        # print(type(self.env))
        # print(type(root))

        root.DballUpdate(self.env)

        while OPEN:
            n = OPEN.pop(0)
            print(n.state)

            # print(type(self.env))
            # print(type(n))
            n.DballUpdate(self.env)
            if self.env.is_final_state(n.state):
                return (solution(n, env), n.cost, self.expanded)
            if n.state not in CLOSED:
                CLOSED.append(n.state)
                self.expanded += 1
                # print(self.env.succ(n.state))
                for action in self.env.succ(n.state):
                    new_state = self.env.succ(n.state)[action][0]
                    # if new_state equals to some state inside OPEN or new_state in CLOSED or new_state == None: continue
                    in_close = False
                    if new_state == None:
                        continue
                    for state in CLOSED:
                        if state[0] == new_state[0]:
                            in_close = True
                            break
                    if new_state in OPEN or in_close:
                        continue
                    child = Node(new_state, n)
                    child.cost = n.cost + self.env.succ(n.state)[action][1]
                    # print(type(self.env))
                    # print(type(child))
                    child.DballUpdate(self.env)
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