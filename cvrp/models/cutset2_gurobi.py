import math
import time

import pulp
import numpy as np
import networkx as nx

from cvrp import utils


def build_model(n_nodes, dists, demands, capacity, n_vehicles, relax=False):
    nodes = list(range(n_nodes))
    vehicles = list(range(n_vehicles))
    model = pulp.LpProblem()

    cat = pulp.LpContinuous if relax else pulp.LpBinary

    x = np.array(pulp.LpVariable.matrix('x', (vehicles, nodes, nodes), 0, 1, cat=cat))
    z = np.array(pulp.LpVariable.matrix('z', (vehicles, nodes), 0, 1, cat=pulp.LpBinary))
    utils.fix_value(np.diagonal(x, axis1=1, axis2=2), 0)
    utils.fix_value(z[:, 0], 1)

    model.setObjective(pulp.lpSum(dists*x))

    for i in nodes[1:]:
        model.addConstraint(pulp.lpSum(z[:, i]) == 1)

    for a in vehicles:
        for i in nodes:
            model.addConstraint(pulp.lpSum(x[a, i, :]) == z[a, i])
            model.addConstraint(pulp.lpSum(x[a, :, i]) == z[a, i])
            model.addConstraint(pulp.lpSum(x[a, :, i]) == x[a, i, :])

    for a in vehicles:
        model.addConstraint(pulp.lpDot(demands, z[a]) <= capacity)

    for a in vehicles[:-1]:
        model.addConstraint(pulp.lpSum(z[a]) >= pulp.lpSum(z[a + 1]))

    for a in vehicles:
        for i in nodes[1:]:
            for j in nodes:
                model.addConstraint(x[a, i, j] + x[a, j, i] <= 1)

    return model, (x, z)


def decode_result(x, labels):
    labels = dict(enumerate(labels))
    try:
        graph = nx.from_numpy_array(x.sum(axis=0), create_using=nx.DiGraph)
    except TypeError:
        print('type error')
        return []
    graph = nx.relabel_nodes(graph, labels)
    return list(nx.simple_cycles(graph))

def draw_result(x, pos):
    graph = nx.from_numpy_array(utils.get_values(x))
    nx.draw(graph, pos, node_size=1)

def solve(instance, timelimit, log_dir):
    nodes = list(instance.get_nodes())
    n_nodes = len(nodes)
    demands = np.array([instance.demands[node] for node in nodes])
    coords = np.array([instance.node_coords[node] for node in nodes])
    dists = utils.dists(coords, coords)
    capacity = instance.capacity
    capacity = 150 # demands: 410
    n_vehicles = math.ceil(sum(demands)/capacity)

    start = time.time()
    limit = start + timelimit

    model, (x, z) = build_model(n_nodes, dists, demands, capacity, n_vehicles, relax=True)
    model.solver = utils.get_solver(msg=False, warm_start=True)
    assert model.solve() == pulp.LpStatusOptimal

    while time.time() < limit:
        feasible = True
        for a in range(n_vehicles):
            graph = nx.from_numpy_array(utils.get_values(x[a]), create_using=nx.DiGraph)
            tours = list(nx.simple_cycles(graph))
            print(tours)
            if len(tours) > 1:
                feasible = False
            for tour in tours:
                if 0 in tour:
                    tour.remove(0)
                n_edges = pulp.lpSum(x[:, tour, :][:, :, tour])
                v = math.ceil(demands[tour].sum()/capacity)
                model.addConstraint(n_edges <= len(tour) - v)

        if feasible:
            break
        model.solver.timeLimit = int(limit - time.time())
        if model.solver.timeLimit < 1:
            print('time limit exceeded.')
            break
        assert model.solve() == pulp.LpStatusOptimal
    model.writeLP(f'{log_dir}/model.lp')

    print(np.array(np.where(utils.get_values(x) == 1)).T)
    for a in range(n_vehicles):
        graph = nx.from_numpy_array(utils.get_values(x[a]))
        for tour in nx.connected_components(graph):
            tour = list(tour)
            if len(tour) == 1:
                break
            print(f'vehicle: {a}, demands: {demands[tour].sum()}, tour: {tour}')
    return decode_result(utils.get_values(x), nodes)