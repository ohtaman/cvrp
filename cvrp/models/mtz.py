import math

import pulp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import mlflow

from cvrp import utils


def build_model(n_nodes, demands, capacity, dists, n_vehicles):
    model = pulp.LpProblem()
    nodes = list(range(n_nodes))

    x = np.array(pulp.LpVariable.matrix('x', (nodes, nodes), cat=pulp.LpBinary))
    u = np.array(pulp.LpVariable.matrix('u', nodes, 0, capacity, cat=pulp.LpContinuous))
    utils.fix_value(np.diagonal(x), 0)
    utils.fix_value(u[0], 0)

    model.setObjective(pulp.lpSum(dists*x))

    for i in nodes[1:]:
        model.addConstraint(pulp.lpSum(x[i, :]) == 1)
        model.addConstraint(pulp.lpSum(x[:, i]) == 1)

    model.addConstraint(pulp.lpSum(x[0, :]) == n_vehicles)
    model.addConstraint(pulp.lpSum(x[:, 0]) == n_vehicles)

    for i in nodes:
        for j in nodes[1:]:
            model.addConstraint(u[i] + demands[j] - capacity*(1 - x[i, j]) <= u[j])

    for i in nodes:
        model.addConstraint(u[i] >= demands[i])

    return model, (x, u)


def decode_result(x, labels):
    try:
        graph = nx.from_numpy_array(x, create_using=nx.DiGraph)
    except TypeError:
        return []
    graph = nx.relabel_nodes(graph, labels)
    return list(nx.simple_cycles(graph))


def solve(instance, timelimit, log_dir, n_vehicles=None):
    nodes = list(instance.get_nodes())
    demands = np.array([instance.demands[node] for node in nodes])
    coords = np.array([instance.node_coords[node] for node in nodes])
    dists = utils.dists(coords, coords)
    capacity = instance.capacity
    if n_vehicles is None:
        n_vehicles = math.ceil(sum(demands)/capacity)

    assert len(instance.depots) == 1
    assert instance.depots[0] == nodes[0]
    model, (x, q) = build_model(n_nodes=len(nodes), demands=demands, capacity=capacity, dists=dists, n_vehicles=n_vehicles)
    model.writeLP(log_dir.joinpath(f'model.lp'))
    mlflow.log_params(dict(
        n_constraints=len(model.constraints),
        n_variables=len(model.variables()))
    )

    solver = utils.get_solver()
    solver.timeLimit = timelimit
    model.solver = solver
    assert model.solve() == pulp.LpSolutionOptimal

    return decode_result(utils.get_values(x), dict(enumerate(nodes)))