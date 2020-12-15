import math
import time

import pulp
import numpy as np
import networkx as nx

from cvrp import utils


def build_model(n_nodes, dists, n_vehicles):
    nodes = list(range(n_nodes))

    model = pulp.LpProblem()
    x = np.array(pulp.LpVariable.matrix('x', (nodes, nodes), cat=pulp.LpBinary))
    utils.fix_value(np.diagonal(x), 0)

    objective = pulp.lpSum(dists*x)
    model.setObjective(objective)

    for i in nodes[1:]:
        model.addConstraint(pulp.lpSum(x[i, :]) == 1)
        model.addConstraint(pulp.lpSum(x[:, i]) == 1)  
    model.addConstraint(pulp.lpSum(x[0, :]) == n_vehicles)
    model.addConstraint(pulp.lpSum(x[:, 0]) == n_vehicles)

    return model, x


def decode_result(x, labels):
    labels = dict(enumerate(labels))
    try:
        graph = nx.from_numpy_array(x, create_using=nx.DiGraph)
    except TypeError:
        print('type error')
        return []
    graph = nx.relabel_nodes(graph, labels)
    return list(nx.simple_cycles(graph))


def draw_result(x, pos):
    graph = nx.from_numpy_array(utils.get_values(x))
    nx.draw(graph, pos, node_size=1)

def solve(instance,  timelimit, log_dir):
    nodes = list(instance.get_nodes())
    n_nodes = len(nodes)
    demands = np.array([instance.demands[node] for node in nodes])
    coords = np.array([instance.node_coords[node] for node in nodes])
    dists = utils.dists(coords, coords)
    capacity = instance.capacity
    n_vehicles = math.ceil(sum(demands)/capacity)

    start = time.time()
    limit = start + timelimit
    model, x = build_model(n_nodes, dists, n_vehicles)
    model.solver = utils.get_solver(warm_start=True)
    assert model.solve() == 1

    while time.time() < limit:
        graph = nx.from_numpy_array(utils.get_values(x))
        graph.remove_node(0)
        tours = list(nx.connected_components(graph))
        feasible = True
        for tour in tours:
            tour = list(tour)
            demand = demands[tour].sum()
            n_edges = pulp.lpSum(x[tour, :][:, tour])
            v = math.ceil(demand/capacity)
            constraint = n_edges <= len(tour) - v
            if not constraint.valid():
                feasible = False
                model.addConstraint(constraint)
        if feasible:
            break

        model.solver.timeLimit = int(limit - time.time())
        if model.solver.timeLimit < 1:
            print('time limit exceeded.')
            break
        assert pulp.LpStatus[model.solve()] == 'Optimal'

    return decode_result(utils.get_values(x), nodes)