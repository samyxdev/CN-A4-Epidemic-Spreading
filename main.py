import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import time

import os

# Initial states cosntants
P0 = 0.2
TMAX = 1000
TSKIP = 100
NREP = 20

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
list of length TMAX - TSKIP
"""
start_time = time.time()
def simulation(g, len_g, mu=MU, beta=BETA):
    state = [np.random.binomial(1, P0) for _ in range(len_g)] # 1 means infected
    old_state = state[:]

    infected_ratio_list = np.zeros(TMAX - TSKIP)

    # Simulation loop
    for i in range(TMAX):
        old_state = state[:]

        # To skip the saving of transionary states
        if i >= TSKIP:
            for j in range(len_g):
                infected_ratio_list[i - TSKIP] += update_node_state(g, state, old_state, j, mu, beta)

            infected_ratio_list[i - TSKIP] /= len_g

        else:
            for j in range(len_g):
                update_node_state(g, state, old_state, j)

    return infected_ratio_list


len_g = 500
g = ig.Graph.Barabasi(len_g, 5)

"""
Runs n simulations and returns the average over all simulations
of the average infected ratio on a network
"""
def get_avg_infected(g, len_g, mu, beta, n=NREP):
    avg_infected_ratio = np.zeros(TMAX - TSKIP)

    for i in tqdm(range(n)):
        infected_ratio_list = simulation(g, len_g, mu, beta)

        avg_infected_ratio += infected_ratio_list

    avg_infected_ratio /= NREP

    plt.plot(avg_infected_ratio, label=f"beta={beta}")

    return sum(avg_infected_ratio)/NREP


"""
Applies simulations with different beta values and plots them in one plot if plot=True.
Returns also the list of average infected ratio for each beta value
"""
def iterate_over_beta(mu, beta_list, model_name, plot=False):
    avg_over_beta_list = [] # Contains avg ratio for all beta
    for beta in tqdm(beta_list):
        avg_over_beta_list.append(get_avg_infected(g, len_g, MU, beta))

    if plot:
        plt.title(f"{model_name}, SIS(p0={P0}, mu={MU}, various beta)(t)")
        plt.xlabel("t")
        plt.ylabel("p")
        plt.legend()

        plt.savefig(os.path.join("fig", f"{model_name}_mu{mu}_{len(beta_list)}betavals.png"))

    #plt.show()

    return avg_over_beta_list

# Simple plot
beta_list = np.linspace(0.1, 1, 5)
plt.plot(iterate_over_beta(MU, beta_list, "BA", plot=True))

plt.xlabel("beta")
plt.ylabel("p")
plt.title(f"BA, SIS(p0={P0}, mu={MU}, various beta)(t)")
plt.show()

# Large beta list plot and different mu
for mu in [0.1, 0.5, 0.9]:
    beta_list = np.linspace(0.01, 1, int((1-0.01)/0.02))
    plt.plot(iterate_over_beta(mu, beta_list, "BA"))

    plt.xlabel("beta")
    plt.ylabel("p")
    plt.title(f"BA, SIS(p0={P0}, mu={mu}, various beta)(t)")
    plt.show()


