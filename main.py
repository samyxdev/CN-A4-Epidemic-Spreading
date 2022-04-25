import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import time

# Initial states cosntants
P0 = 0.2
TMAX = 1000
TTRANS = 900
NREP = 100

MU = 0.5
BETA = 0.5

"""
Update the state of one node depending on its Infected/Susceptible state
and Return the new state of the node (for general infected ratio measurement purposes)
"""
def update_node_state(g, state, old_state, vertex_id, mu=MU, beta=BETA):
    if old_state[vertex_id]:
        state[vertex_id] = np.random.binomial(1, mu)

    else:
        infected_vertices = sum([old_state[v] for v in g.neighbors(vertex_id)])
        state[vertex_id] = np.random.binomial(1, 1 - (1 - beta)**infected_vertices)

    return state[vertex_id]

"""
Applies one simulation to the graph in argument and returns the infected ratio
list
"""
start_time = time.time()
def simulation(g, len_g, mu=MU, beta=BETA):
    state = [np.random.binomial(1, P0) for _ in range(len_g)] # 1 means infected
    old_state = state[:]

    infected_ratio_list = np.zeros(TMAX - TTRANS)

    # Simulation loop
    for i in tqdm(range(TMAX)):
        old_state = state[:]

        # To skip the saving of transionary states
        if i < TMAX - TTRANS:
            for j in range(len_g):
                infected_ratio_list[i] += update_node_state(g, state, old_state, j, mu, beta)

            infected_ratio_list[i] /= len_g

        else:
            for j in range(len_g):
                update_node_state(g, state, old_state, j)

    return infected_ratio_list


len_g = 1000
g = ig.Graph.Barabasi(len_g, 5)

avg_infected_ratio = np.zeros(TMAX - TTRANS)

for i in range(NREP):
    infected_ratio_list = simulation(g, len_g)

    avg_infected_ratio += infected_ratio_list

    # To plot 3 first
    if i < 3:
        plt.plot(infected_ratio_list)
        plt.show()

avg_infected_ratio /= NREP
plt.plot(avg_infected_ratio)
plt.show()






