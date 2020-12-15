from collections import defaultdict
from dataclasses import dataclass
import math
import time

import numpy as np
import networkx as nx
import pulp

from cvrp import utils


@dataclass
class Route:
    nodes: list
    cost: float

    @classmethod
    def from_nodes(cls, nodes: list, dists: np.array):
        cost = sum(
            dists[nodes[i], nodes[i + 1]]
            for i in range(-1, len(nodes) - 1)
        )
        return cls(nodes=nodes, cost=cost)

    def __hash__(self):
        return hash(sorted(self.nodes))


def build_master_model(n_nodes, n_vehicles):
    model = pulp.LpProblem()
    y = pulp.LpVariable.matrix('y', range(n_nodes), 0, cat=pulp.LpContinuous)
    model.setObjective(-(pulp.lpSum(y[1:]) + n_vehicles*y[0]))

    return model, y


def build_slave_model(n_nodes, demands, capacity, dists):
    model = pulp.LpProblem()
    x = np.array(pulp.LpVariable.matrix('x', (range(n_nodes), range(n_nodes)), cat=pulp.LpBinary))
    w = np.array(pulp.LpVariable.matrix('w', range(n_nodes), cat=pulp.LpBinary))
    q = np.array(pulp.LpVariable.matrix('u', range(n_nodes), 0, capacity, cat=pulp.LpContinuous))
    utils.fix_value(np.diagonal(x), 0)
    utils.fix_value(w[0], 0)
    utils.fix_value(q[0], 0)

    for i in range(1, n_nodes):
        model.addConstraint(pulp.lpSum(x[i, :]) == w[i])
        model.addConstraint(pulp.lpSum(x[:, i]) == w[i])

    model.addConstraint(pulp.lpSum(x[0, :]) == 1)
    model.addConstraint(pulp.lpSum(x[:, 0]) == 1)
    
    for i in range(n_nodes):
        for j in range(1, n_nodes):
            model.addConstraint(
                q[i] + demands[j] - capacity*(1 - x[i, j]) + (capacity - demands[i] - demands[j])*x[j, i] <= q[j]
            )

    for i in range(1, n_nodes):
        model.addConstraint(q[i] >= demands[i] + pulp.lpDot(demands, x[:, i]))
        model.addConstraint(q[i] <= capacity - pulp.lpDot(demands, x[i, :]))
        model.addConstraint(q[i] <= capacity - (capacity - max(demands[j] for j in range(len(demands)) if i != j) - demands[i])*x[0, i] - pulp.lpDot(demands, x[i, :]))

    model.addConstraint(pulp.lpDot(w, demands) <= capacity)

    return model, (x, q, w)


def build_primal_model(n_nodes, routes, n_vehicles):
    costs = np.array([route.cost for route in routes])
    a = defaultdict(list)
    for r, route in enumerate(routes):
        for i in route.nodes:
            a[i].append(r)

    model = pulp.LpProblem()
    z = np.array(pulp.LpVariable.matrix('z', range(len(routes)), cat=pulp.LpBinary))

    model.setObjective(pulp.lpDot(costs, z))

    for i in range(n_nodes):
        model.addConstraint(
            pulp.lpSum(z[r] for r in a[i]) >= 1
        )

    model.addConstraint(pulp.lpSum(z) == n_vehicles)

    return model, z


def update_slave_model(model, dists, n_vehicles, x, w, y_opt):
    model.setObjective(pulp.lpSum(dists*x) - pulp.lpDot(y_opt, w) - y_opt[0]*n_vehicles)


def update_master_model(model, y, route, capacity):
    model.addConstraint(pulp.lpSum(y[i] for i in route.nodes) <= route.cost)
    for i in route.nodes[1:]:
        y[i].bounds(0, None)
    y[0].bounds(0, 0) # y[0] は下限なし


def solve(instance, timelimit, log_dir):
    nodes = list(instance.get_nodes())
    n_nodes = len(nodes)
    demands = np.array([instance.demands[node] for node in nodes])
    coords = np.array([instance.node_coords[node] for node in nodes])
    dists = utils.dists(coords, coords)
    capacity = instance.capacity
    n_vehicles = math.ceil(sum(demands)/capacity)

    start = time.time()
    routes = []
    master_model, y = build_master_model(n_nodes, n_vehicles)
    master_model.solver = utils.get_solver(warm_start=True)
    for i in range(1, n_nodes):
        route = Route.from_nodes([0, i], dists)
        routes.append(route)
        update_master_model(master_model, y, route, capacity)

    slave_model, (x, q, w) = build_slave_model(n_nodes, demands, capacity, dists)
    slave_model.solver = utils.get_solver(warm_start=True)

    while start + timelimit > time.time():
        assert master_model.solve() == 1
        y_opt = utils.get_values(y)
        print(f'#y_opt > 0: {(y_opt > 0).sum()}, #y_opt > 1: {(y_opt > 1).sum()}')
        update_slave_model(slave_model, dists, n_vehicles, x, w, y_opt)
        assert slave_model.solve() == 1
        if slave_model.objective.value() > 0:
            break

        graph = nx.from_numpy_array(utils.get_values(x), create_using=nx.DiGraph)
        route = Route.from_nodes(next(nx.simple_cycles(graph)), dists)
        routes.append(route)
        print(f'objective: {slave_model.objective.value()}, route: {route.nodes}')
        update_master_model(master_model, y, route, capacity)

    primal_model, z = build_primal_model(n_nodes, routes, n_vehicles)
    primal_model.solver = utils.get_solver()
    primal_model.solve()

    return [
        [nodes[i] for i in routes[r].nodes]
        for r, v in enumerate(utils.get_values(z))
        if v == 1
    ]