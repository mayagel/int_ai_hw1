
# %%
import time
from IPython.display import clear_output

from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
from Algorithms import *


# %%
DOWN = 0
RIGHT = 1
UP = 2
LEFT = 3


# %%
MAPS = {
    "4x4": ["SFFF",
            "FDFF",
            "FFFD",
            "FFFG"],
    "8x8": [
        "SFFFFFFF",
        "FFFFFTAL",
        "TFFHFFTF",
        "FFFFFHTF",
        "FAFHFFFF",
        "FHHFFFHF",
        "DFTFHDTL",
        "FLFHFFFG",
    ],
}


# %%
env = DragonBallEnv(MAPS["8x8"])
state = env.reset()
print('Initial state:', state)
print('Goal states:', env.goals) 

# %% [markdown]
# First, take a look at the state space $\mathcal{S}$ (all possible states) and action space $\mathcal{A}$ (all possible actions). 

# %%
print(f"Action Space {env.action_space}")
print(f"State Space {env.observation_space}")


print(env.render())


# %%
env.set_state((18, False, False))
print(env.render())
print(f"the agent is at state: {env.get_state()}")


# %%
current_state = env.get_state()
print(f"Current state: {current_state}\n")
for action, successor in env.succ(current_state).items():
  print(f"*** Action: {action} ***")
  print(f"Next state: {successor[0]}")
  print(f"Cost: {successor[1]}")
  print(f"Terminated: {successor[2]}\n")


# %%
state, cost, terminated = env.succ(current_state)[1]

print(f"Next state: {state}")
print(f"Cost: {cost}")
print(f"Terminated: {terminated}")
print(f"Final state: {env.is_final_state(state)}")

# %% [markdown]
# Let's see what happens when we apply succ(state) on a hole:
# 
# 

# %%
print(f"Current state: 19\n")
for action, (state, cost, terminated) in env.succ((19, False, False)).items():
  print(f"*** Action: {action} ***")
  print(f"Next state: {state}")
  print(f"Cost: {cost}")
  print(f"Terminated: {terminated}\n")


# %%
new_state, cost, terminated = env.step(DOWN)
print(env.render())
print("New state:", new_state)
print("cost:", cost)
print("Terminated:", terminated)

 


# %%
print(f"current state befor reset: {env.get_state()}")
env.reset()
print(f"current state after reset: {env.get_state()}")

# %%
class RandomAgent():
  def __init__(self):
    self.env = None

  def animation(self, epochs: int ,state: int, action: List[int], total_cost: int) -> None:
      clear_output(wait=True)
      print(self.env.render())
      print(f"Timestep: {epochs}")
      print(f"State: {state}")
      print(f"Action: {action}")
      print(f"Total Cost: {total_cost}")
      time.sleep(1)

  def random_search(self, DragonBallEnv: env) -> Tuple[List[int],int]:
    self.env = env
    self.env.reset()
    epochs = 0
    cost = 0
    total_cost = 0

    actions = []

    state = self.env.get_initial_state()
    while not self.env.is_final_state(state):
      action = self.env.action_space.sample()
      new_state, cost, terminated = self.env.step(action)
        
      while terminated is True and self.env.is_final_state(state) is False:
        self.env.set_state(state)
        action = self.env.action_space.sample()
        new_state, cost, terminated = self.env.step(action)
        
      actions.append(action)
      total_cost += cost
      state = new_state
      epochs += 1
      
      self.animation(epochs,state,action,total_cost)

    return (actions, total_cost)

# Let's check out this agent's performance!
# 
# The output of this agent is the sequence of actions that led to the solution and the route's cost. 
# 
# Our random agent is not very successful, so we'll print his actions as they happen. 
# 
# 1.   **Stop his run in the middle if you are tired of looking at him.**
# 2.   After watching the agent please put the code in the box below in the a comment for your comfort.
# 
#  
# 
# 

# %%
# agent = RandomAgent()
# agent.random_search(env)

# %% [markdown]
# **Did you remember to put the code above in a comment?!**

# %% [markdown]
# As you can see, a random policy is, unsurprisingly, not a good policy. However, what else can we do?
# 
# This is where you come in!
# 
# In this assignment you will be required to implement the following algorithms taught in class in order to solve the problem.
# 
# Algorithms: 
# 1. BFS-G
# 2. W-A*
# 3. epsilon-A*
# 
# Important to note!
# 
# Each agent should return a tuple: (actions, cost, expended) 
# *  actions - the list of integers containing the sequence of actions that produce your agent's solution (and not the entire search process).
# * cost -  an integer which holds the total cost of the solution.
# * expanded - an integer which holds the number of nodes that have been expanded during the search (A node is considered expanded if we check for it's successors).
# 
# The solution to our search problem is the a to the final state, not the final state itself. By saving the actions, we are able to restore the path your agent found.
# 
# 
# Any other output, unless otherwise specified, will cause the running of the tests to fail and will result in a grade of 0 !
# 
# 

# %% [markdown]
# 
# Some Tips:
# 1. Follow the pseudo-code shown in the lectures and tutorials.
# 2. You should write all your code within the classes. This way, we prevent overlapping functions with the same name while running the notebook.
# 3. You may implement your code as you like but consider inherenting from the a general "Agent" class and implement some utilty methods such as the "solution" method which recieves a node and returns a path (sequence of actions) leading to that node.
# 4. Consider implementing a "node" class.
# 5. Using small boards will help you debug.
# 

# %% [markdown]
# The function below (`print_solution()`) can be used for debugging purposes. It prints the sequence of actions it receives. The function will not be used to test your code, so you are welcome to change it.

# %%
def print_solution(actions,env: DragonBallEnv) -> None:
    env.reset()
    total_cost = 0
    print(env.render())
    print(f"Timestep: {1}")
    print(f"State: {env.get_state()}")
    print(f"Action: {None}")
    print(f"Cost: {0}")
    time.sleep(1)

    for i, action in enumerate(actions):
      state, cost, terminated = env.step(action)
      total_cost += cost
      clear_output(wait=True)

      print(env.render())
      print(f"Timestep: {i + 2}")
      print(f"State: {state}")
      print(f"Action: {action}")
      print(f"Cost: {cost}")
      print(f"Total cost: {total_cost}")
      
      time.sleep(1)

      if terminated is True:
        break


# %%
BFS_agent = BFSAgent()
actions, total_cost, expanded = BFS_agent.search(env)
print(f"Total_cost: {total_cost}")
print(f"Expanded: {expanded}")
print(f"Actions: {actions}")

assert total_cost == 119.0, "Error in total cost returned"

# %%
# print_solution(actions, env)

# %% [markdown]
# # Heapdict
# For the next algorithms, you will be required to maintain an "open" queue based on a certain value (g/h/v). To manage these queues efficiently and conveniently, please use [Heapdict](https://www.geeksforgeeks.org/priority-queue-using-queue-and-heapdict-module-in-python/). Heapdict implements the MutableMapping ABC, meaning it works pretty much like a regular Python [dictionary](https://www.geeksforgeeks.org/python-dictionary/). It’s designed to be used as a priority queue. Along with functions provided by ordinary dict(), it also has popitem() and peekitem() functions which return the pair with the lowest priority.
# 
# Note:
# 
# When two nodes have the same minimum value, select the node with the lower state index first. Instead of defining priority as an integer, you can define it as a tuple (value, state, ...).

# %% [markdown]
# ## 2. Weighted A*
# TO DO: implement Wighted A* like shown in class.
# 
# Note:
# 
# *   A parameter called `h_weight` is passed to `Greedy_Best_First_search()`, which indicates how much weight is given to the heuristics (ranging from 0 to 1).
# *   The heurisitcs needed to be implemented. Instructions in dry pdf.
# 
# 
# 
# 
# 

# %%
WA_agent = WeightedAStarAgent()
actions, total_cost, expanded = WA_agent.search(env, h_weight=0.5)
print(f"Total_cost: {total_cost}")
print(f"Expanded: {expanded}")
print(f"Actions: {actions}")

# %%
# print_solution(actions, env)

# %% [markdown]
# ## 3. A*-epsilon::
# TO DO: implement A*-epsilon: like shown in class.
# 
# use the same heuristic as in previous sections.
# 
# Note:
# *   A parameter called `epsilon` is passed to `A_star_epsilon_search()`.
# *   We will not test the amount of expanded nodes for this algorithm.
# 

# %%
AStarEpsilon_agent = AStarEpsilonAgent()
actions, total_cost, expanded = AStarEpsilon_agent.search(env, epsilon=100)
print(f"Total_cost: {total_cost}")
print(f"Expanded: {expanded}")
print(f"Actions: {actions}")

# %%
print_solution(actions, env)

# %% [markdown]
# ## 7. Benchmarking:
# In this section we want to compare the different search algorithms. The take-home message is that there is no "one algorithm fits all".

# %%
import csv

test_boards = {
"map12x12": 
['SFAFTFFTHHHF',
'AFLTFFFFTALF',
'LHHLLHHLFTHD',
'HALTHAHHADHF',
'FFFTFHFFAHFL',
'LLTHFFFAHFAT',
'HAAFFALHTATF',
'LLLFHFFHTLFH',
'FATAFHTTFFAF',
'HHFLHALLFTLF',
'FFAFFTTAFAAL',
'TAAFFFHAFHFG'],
"map15x15": 
['SFTTFFHHHHLFATF',
'ALHTLHFTLLFTHHF',
'FTTFHHHAHHFAHTF',
'LFHTFTALTAAFLLH',
'FTFFAFLFFLFHTFF',
'LTAFTHFLHTHHLLA',
'TFFFAHHFFAHHHFF',
'TTFFLFHAHFFTLFD',
'TFHLHTFFHAAHFHF',
'HHAATLHFFLFFHLH',
'FLFHHAALLHLHHAT',
'TLHFFLTHFTTFTTF',
'AFLTDAFTLHFHFFF',
'FFTFHFLTAFLHTLA',
'HTFATLTFHLFHFAG'],
"map20x20" : 
['SFFLHFHTALHLFATAHTHT',
'HFTTLLAHFTAFAAHHTLFH',
'HHTFFFHAFFFFAFFTHHHT',
'TTAFHTFHTHHLAHHAALLF',
'HLALHFFTHAHHAFFLFHTF',
'AFTAFTFLFTTTFTLLTHDF',
'LFHFFAAHFLHAHHFHFALA',
'AFTFFLTFLFTAFFLTFAHH',
'HTTLFTHLTFAFFLAFHFTF',
'LLALFHFAHFAALHFTFHTF',
'LFFFAAFLFFFFHFLFFAFH',
'THHTTFAFLATFATFTHLLL',
'HHHAFFFATLLALFAHTHLL',
'HLFFFFHFFLAAFTFFDAFH',
'HTLFTHFFLTHLHHLHFTFH',
'AFTTLHLFFLHTFFAHLAFT',
'HAATLHFFFHHHHAFFFHLH',
'FHFLLLFHLFFLFTFFHAFL',
'LHTFLTLTFATFAFAFHAAF',
'FTFFFFFLFTHFTFLTLHFG']}

test_envs = {}
for board_name, board in test_boards.items():
    test_envs[board_name] = DragonBallEnv(board)


BFS_agent = BFSAgent()
WAStar_agent = WeightedAStarAgent()

weights = [0.5, 0.7, 0.9]

agents_search_function = [
    BFS_agent.search,
]

header = ['map',  "BFS-G cost",  "BFS-G expanded",\
           'WA* (0.5) cost', 'WA* (0.5) expanded', 'WA* (0.7) cost', 'WA* (0.7) expanded', 'WA* (0.9) cost', 'WA* (0.9) expanded']

with open("results.csv", 'w') as f:
  writer = csv.writer(f)
  writer.writerow(header)
  for env_name, env in test_envs.items():
    data = [env_name]
    for agent in agents_search_function:
      _, total_cost, expanded = agent(env)
      data += [total_cost, expanded]
    for w in weights:
        _, total_cost, expanded = WAStar_agent.search(env, w)
        data += [total_cost, expanded]

    writer.writerow(data)


