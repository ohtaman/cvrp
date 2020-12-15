import math

import pulp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import mlflow

from cvrp import utils

def build_model(n_nodes, demands, capacity, dists, n_vehicles):
    nodes = list(range(n_nodes))

    model = pulp.LpProblem('mtz_strong')

    x = np.array(pulp.LpVariable.matrix('x', (nodes, nodes), cat=pulp.LpBinary))
    q = np.array(pulp.LpVariable.matrix('q', nodes, 0, capacity, cat=pulp.LpContinuous))
    utils.fix_value(np.diagonal(x), 0)
    utils.fix_value(q[0], 0)

    model.setObjective(pulp.lpSum(dists*x))

    for i in nodes[1:]:
        model.addConstraint(pulp.lpSum(x[i, :]) == 1)
        model.addConstraint(pulp.lpSum(x[:, i]) == 1)

    model.addConstraint(pulp.lpSum(x[0, :]) == n_vehicles)
    model.addConstraint(pulp.lpSum(x[:, 0]) == n_vehicles)

    for i in nodes:
        for j in nodes[1:]:
            model.addConstraint(
                q[i] + demands[j] - capacity*(1 - x[i, j]) + (capacity - demands[i] - demands[j])*x[j, i] <= q[j]
            )

    for i in nodes[1:]:
        model.addConstraint(q[i] >= demands[i] + pulp.lpDot(demands, x[:, i]))
        model.addConstraint(q[i] <= capacity - pulp.lpDot(demands, x[i, :]))
        model.addConstraint(q[i] <= capacity - (capacity - max(demands[j] for j in range(len(demands)) if i != j) - demands[i])*x[0, i] - pulp.lpDot(demands, x[i, :]))

    return model, (x, q)


def decode_result(x, labels):
    try:
        graph = nx.from_numpy_array(x, create_using=nx.DiGraph)
    except TypeError:
        return []
    graph = nx.relabel_nodes(graph, labels)
    return list(nx.simple_cycles(graph))


def solve(instance, timelimit, log_dir):
    nodes = list(instance.get_nodes())
    demands = np.array([instance.demands[node] for node in nodes])
    coords = np.array([instance.node_coords[node] for node in nodes])
    dists = utils.dists(coords, coords)
    capacity = instance.capacity
    n_vehicles = math.ceil(sum(demands)/capacity)

    assert len(instance.depots) == 1
    assert instance.depots[0] == nodes[0]
    model, (x, q) = build_model(n_nodes=len(nodes), demands=demands, capacity=capacity, dists=dists, n_vehicles=n_vehicles)
    print(pulp.lpDot(demands, x))
    model.writeLP(f'{log_dir}/model.lp')
    mlflow.log_params(dict(
        n_constraints=len(model.constraints),
        n_variables=len(model.variables()))
    )

    solver = utils.get_solver()
    solver.timeLimit = timelimit
    model.solver = solver
    model.solve()
    np.savetxt(f'{log_dir}/q.csv', utils.get_values(q))
    np.savetxt(f'{log_dir}/x.csv', utils.get_values(x))

    return decode_result(utils.get_values(x), dict(enumerate(nodes)))