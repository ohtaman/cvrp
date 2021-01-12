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
    f = np.array(pulp.LpVariable.matrix('f', (nodes, nodes), 0, capacity, cat=pulp.LpContinuous))
    utils.fix_value(np.diagonal(x), 0)
    utils.fix_value(np.diagonal(f), 0)

    model.setObjective(pulp.lpSum(dists*x))

    for i in nodes[1:]:
        model.addConstraint(pulp.lpSum(x[i, :]) == 1)
        model.addConstraint(pulp.lpSum(x[:, i]) == 1)

    model.addConstraint(pulp.lpSum(x[0, :]) == n_vehicles)
    model.addConstraint(pulp.lpSum(x[:, 0]) == n_vehicles)

    for i in nodes[1:]:
        model.addConstraint(pulp.lpSum(f[:, i]) - pulp.lpSum(f[i, :]) == demands[i])

    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            model.addConstraint(f[i, j] <= (capacity - demands[i])*x[i, j])
            # model.addConstraint(f[i, j] >= demands[j]*x[i, j])

    model._x = x
    model._f = f
    return model


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
    model = build_model(n_nodes=len(nodes), demands=demands, capacity=capacity, dists=dists, n_vehicles=n_vehicles)
    model.writeLP(log_dir.joinpath(f'model.lp'))
    mlflow.log_params(dict(
        n_constraints=len(model.constraints),
        n_variables=len(model.variables()))
    )

    solver = utils.get_solver()
    solver.timeLimit = timelimit
    model.solver = solver
    assert model.solve() == pulp.LpSolutionOptimal

    return decode_result(utils.get_values(model._x), dict(enumerate(nodes)))