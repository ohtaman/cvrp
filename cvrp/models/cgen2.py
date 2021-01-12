from collections import defaultdict
import math
import time

import numpy as np
import networkx as nx
import pulp

from cvrp import utils


class Route:
    def __init__(self, nodes, cost):
        self.nodes = nodes
        self.cost = cost

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
    model = pulp.LpProblem(sense=pulp.LpMaximize)
    y = pulp.LpVariable.matrix('y', range(n_nodes), 0, cat=pulp.LpContinuous)
    model.setObjective(pulp.lpSum(y[1:]) + n_vehicles*y[0])

    model._y = y
    return model


def update_master_model(model, route):
    model.addConstraint(pulp.lpSum(model._y[i] for i in route.nodes) <= route.cost)


def build_slave_model(n_nodes, demands, capacity, dists, n_vehicles):
    model = pulp.LpProblem()
    x = np.array(pulp.LpVariable.matrix('x', (range(n_nodes), range(n_nodes)), cat=pulp.LpBinary))
    w = np.array(pulp.LpVariable.matrix('w', range(n_nodes), cat=pulp.LpBinary))
    utils.fix_value(np.diagonal(x), 0)
    utils.fix_value(w[0], 1)

    for i in range(n_nodes):
        model.addConstraint(pulp.lpSum(x[i, :]) == w[i])
        model.addConstraint(pulp.lpSum(x[:, i]) == w[i])
    model.addConstraint(pulp.lpDot(w, demands) <= capacity)

    model._x = x
    model._w = w
    model._demands = demands
    model._capacity = capacity
    model._dists = dists
    model._n_vehicles = n_vehicles
    return model


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

    model._z = z
    return model


def add_cuts(model):
    graph = nx.from_numpy_array(utils.get_values(model._x))
    tours = list(tour for tour in nx.connected_components(graph) if len(tour) > 1)
    n_cuts = 0
    # print(tours)
    for tour in tours:
        if 0 in tour:
            continue
        tour = list(tour)
        n_edges = pulp.lpSum(model._x[tour, :][:, tour])
        model.addConstraint(n_edges <= len(tour) - 1)
        n_cuts += 1

    return n_cuts


def get_route(slave_model, y_opt, timelimit):
    start = time.time()
    cost = pulp.lpSum(slave_model._dists*slave_model._x)
    objective =  cost - pulp.lpDot(y_opt, slave_model._w)
    # objective =  cost - pulp.lpDot(y_opt[1:], slave_model._w[1:]) - y_opt[0]*slave_model._n_vehicles
    slave_model.setObjective(objective)
    slave_model.addConstraint(objective <= 0)
    while time.time() < start + timelimit:
        assert slave_model.solve() == pulp.LpSolutionOptimal
        n_cuts = add_cuts(slave_model)
        if n_cuts == 0:
            break
    else:
        print('Time limit reached.')
        return None
    if slave_model.objective.value() > 0:
        return None
    graph = nx.from_numpy_array(utils.get_values(slave_model._x), create_using=nx.DiGraph)
    return Route(nodes=next(nx.simple_cycles(graph)), cost=cost.value())


def solve(instance, timelimit, log_dir, n_vehicles=None):
    nodes = list(instance.get_nodes())
    n_nodes = len(nodes)
    demands = np.array([instance.demands[node] for node in nodes])
    coords = np.array([instance.node_coords[node] for node in nodes])
    dists = utils.dists(coords, coords)
    capacity = instance.capacity
    if n_vehicles is None:
        n_vehicles = math.ceil(sum(demands)/capacity)

    start = time.time()
    routes = []
    master_model = build_master_model(n_nodes, n_vehicles)
    master_model.solver = utils.get_solver(msg=False, warm_start=True)
    slave_model = build_slave_model(n_nodes, demands, capacity, dists, n_vehicles)
    slave_model.solver = utils.get_solver(msg=False, warm_start=True)
#    slave_model.solver = utils.get_solver(msg=False, warm_start=True, gapRel=2)
#    slave_model.solver = utils.get_solver(msg=False, warm_start=True, gapRel=100)
    for i in range(1, n_nodes):
        route = Route.from_nodes([0, i], dists)
        routes.append(route)
        update_master_model(master_model, route)

    while time.time() < start + timelimit:
        assert master_model.solve() == pulp.LpSolutionOptimal
        y_opt = utils.get_values(master_model._y)
        s = slave_model.solver
        slave_model = build_slave_model(n_nodes, demands, capacity, dists, n_vehicles)
        slave_model.solver = s
        route = get_route(slave_model, y_opt, start + timelimit - time.time())
        print(f'objective of slave: {slave_model.objective.value()}')
        if route is None:
            print('No route added.')
            break
        routes.append(route)
        print(f'add route: {route.nodes}, cost: {route.cost}')
        update_master_model(master_model, route)

    print(f'{len(routes)} routes generated.')
    primal_model = build_primal_model(n_nodes, routes, n_vehicles)
    primal_model.solver = utils.get_solver()
    primal_model.solve()

    return [
        [nodes[i] for i in routes[r].nodes]
        for r, v in enumerate(utils.get_values(primal_model._z))
        if v == 1
    ]