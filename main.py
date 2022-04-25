import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import time

# Initial states cosntants
P0 = 0.2
TMAX = 1000
TRANS = 900
NREP = 100

MU = 0.5
BETA = 0.5

def update_node_state(g, state, old_state, vertex_id, mu=MU, beta=BETA):
    if old_state[vertex_id]:
        state[vertex_id] = np.random.binomial(1, mu)

    else:
        infected_vertices = sum([old_state[v] for v in g.neighbors(vertex_id)])
        state[vertex_id] = np.random.binomial(1, 1 - (1 - beta)**infected_vertices)

    return state[vertex_id]

len_g = 1000
g = ig.Graph.Barabasi(len_g, 5)

#ig.plot(g)

state = [np.random.binomial(1, P0) for _ in range(len_g)] # 1 means infected
old_state = state[:]

infected_ratio_list = []

# Simulation loop
for i in tqdm(range(TMAX)):
    old_state = state[:]

    infected_ratio_list.append(0)

    for j in range(len_g):
        infected_ratio_list[i] += update_node_state(g, state, old_state, j)

    infected_ratio_list[i] /= len_g

#ig.plot(g)
plt.plot(infected_ratio_list)
plt.show()






