import math
import time

import pulp
import numpy as np
import networkx as nx

from cvrp import utils


def build_model(instance, n_vehicles=None):
    nodes = list(instance.get_nodes())
    n_nodes = len(nodes)
    coord = instance.node_coords
    dist = np.array([
        [
            np.linalg.norm([coord[a][0] - coord[b][0], coord[a][1] - coord[b][1]])
            for b in nodes
        ]
        for a in nodes
    ])

    model = pulp.LpProblem()

    x = np.array(pulp.LpVariable.matrix('x', (nodes, nodes), 0, 1, cat=pulp.LpBinary))

    utils.fix_value(np.diagonal(x), 0)

    objective = pulp.lpSum(dist*x)
    model.setObjective(objective)

    for i in range(n_nodes):
        if nodes[i] in instance.depots:
            if not n_vehicles:
                model.addConstraint(pulp.lpSum(x[i, :]) == pulp.lpSum(x[:, i]))
            else:
                model.addConstraint(pulp.lpSum(x[i, :]) == n_vehicles)
                model.addConstraint(pulp.lpSum(x[:, i]) == n_vehicles)
        else:
            model.addConstraint(pulp.lpSum(x[i, :]) == 1)
            model.addConstraint(pulp.lpSum(x[:, i]) == 1)

    return model, x


def decode_result(x):
    graph = nx.from_numpy_array(utils.get_values(x), create_using=nx.DiGraph)
    return list(nx.simple_cycles(graph))

def draw_result(x, pos):
    graph = nx.from_numpy_array(utils.get_values(x))
    nx.draw(graph, pos, node_size=1)

def get_cuts()


def solve(instance, n_vehicles, timelimit):
    model, x = build_model(instance, n_vehicles=n_vehicles)
    nodes = list(instance.get_nodes())
    node_to_idx = {n:i for i, n in enumerate(instance.get_nodes())}
    solver = utils.get_solver()

    start = time.time()
    limit = start + timelimit
    model.solve(solver)

    while True:
        graph = nx.from_numpy_array(utils.get_values(x))
        for depot in instance.depots:
            graph.remove_node(node_to_idx[depot])
        tours = list(nx.connected_components(graph))
        feasible = True
        for tour in tours:
            tour = list(tour)
            tour_size = len(tour)
            demands = sum(instance.demands[nodes[i]] for i in tour)
            n_edges = pulp.lpSum(x[tour, :][:, tour])
            v = math.ceil(demands/instance.capacity)
            if utils.get_values(n_edges) > tour_size - v:
                feasible = False
                model.addConstraint(n_edges <= tour_size - v)
        if feasible:
            break

        solver.timeLimit = int(limit - time.time())
        if solver.timeLimit < 1:
            print('time limit exceeded.')
            break
        if pulp.LpStatus[model.solve(solver)] != 'Optimal':
            print(model)
        assert pulp.LpStatus[model.solve(solver)] == 'Optimal'

    draw_result(x, [instance.node_coords[a] for a in instance.get_nodes()])
    return decode_result(x)