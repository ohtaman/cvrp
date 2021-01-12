from collections import defaultdict
import math
import time

import numpy as np
import networkx as nx
import gurobipy as gp

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


def build_slave_model(n_nodes, demands, capacity, dists, n_vehicles):
    nodes = list(range(n_nodes))

    model = gp.Model('cvrp.cutset_gurobi')
    model.setParam('LazyConstraints', 1)
    
    x = {}
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            x[i, j] = model.addVar(vtype=gp.GRB.BINARY, name=f'x_{i}_{j}')
    w = {}
    for i in nodes[1:]:
        w[i] = model.addVar(vtype=gp.GRB.BINARY, name=f'w_{i}')

    for i in nodes[1:]:
        model.addConstr(gp.quicksum(x[i, j] for j in nodes if i != j) == w[i])
        model.addConstr(gp.quicksum(x[j, i] for j in nodes if i != j) == w[i])
    model.addConstr(gp.quicksum(x[0, j] for j in nodes[1:]) == n_vehicles)
    model.addConstr(gp.quicksum(x[j, 0] for j in nodes[1:]) == n_vehicles)
    model.addConstr(gp.quicksum(demands[i]*w[i] for i in w) <= capacity)

    model._x = x
    model._w = w
    model._dists = dists
    model._capacity = capacity
    model._demands = demands
    return model


def slave_callback(model, where):
    if where != gp.GRB.Callback.MIPSOL:
        return

    edges = []
    for (i, j) in model._x:
        if model.cbGetSolution(model._x[i, j]) > 0.5:
            edges.append((i, j))
    graph = nx.from_edgelist(edges, create_using=nx.DiGraph)
    for tour in nx.simple_cycles(graph):
        dept_contained = 0 in tour
        if dept_contained:
            tour.remove(0)

        n_vehicles = math.ceil(model._demands[tour].sum()/model._capacity)
        n_edges = gp.quicksum(
            model._x[i, j]
            for i in tour
            for j in tour if i != j
        )
        if len(tour) >= 2 and (not dept_contained or n_vehicles > 1):
            model.cbLazy(n_edges <= len(tour) - n_vehicles)


def build_master_model(n_nodes, n_vehicles):
    nodes = list(range(n_nodes))
    model = gp.Model('cvrp.cgen_gurobi.master')
    
    y = {}
    for i in nodes:
        y[i] = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f'y_{i}')

    
    #model.setObjective(gp.quicksum(y[i] for i in nodes[1:]) + n_vehicles*y[0], gp.GRB.MAXIMIZE)
    model.setObjective(gp.quicksum(y[i] for i in nodes[1:]), gp.GRB.MAXIMIZE)

    model._y = y
    return model


def update_master_model(model, route):
    model.addConstr(gp.quicksum(model._y[i] for i in route.nodes) <= route.cost)
    model.update()


def build_primal_model(n_nodes, routes, n_vehicles):
    model = gp.Model('cvrp.cgen_gurobi.primal')

    costs = np.array([route.cost for route in routes])
    a = defaultdict(list)
    for r, route in enumerate(routes):
        for i in route.nodes:
            a[i].append(r)

    z = {}
    for r in range(len(routes)):
        z[r] = model.addVar(vtype=gp.GRB.BINARY, name=f'z_{i}')

    model.setObjective(gp.quicksum(costs[r]*z[r] for r in z))

    for i in range(n_nodes):
        model.addConstr(
            gp.quicksum(z[r] for r in a[i]) >= 1
        )

    model.addConstr(gp.quicksum(z.values()) == n_vehicles)

    model._z = z
    return model


def get_route(slave_model, y_opt, timelimit):
    start = time.time()
    cost = gp.quicksum(slave_model._dists[i, j]*slave_model._x[i, j] for i, j in slave_model._x)
    objective =  cost - gp.quicksum(y_opt[i]*slave_model._w[i] for i in slave_model._w)

    slave_model.setObjective(objective)
    slave_model.update()
    slave_model.optimize(callback=slave_callback)
    assert slave_model.Status in (gp.GRB.Status.OPTIMAL, gp.GRB.Status.TIME_LIMIT)
    
    edges = []
    for (i, j), x in slave_model._x.items():
        if x.x >= 0.5:
            edges.append((i, j))
    graph = nx.from_edgelist(edges, create_using=nx.DiGraph)
    return Route(nodes=next(nx.simple_cycles(graph)), cost=cost.getValue())


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
    slave_model = build_slave_model(n_nodes, demands, capacity, dists, n_vehicles)
    for i in range(1, n_nodes):
        route = Route.from_nodes([0, i], dists)
        routes.append(route)
        update_master_model(master_model, route)

    while time.time() < start + timelimit:
        master_model.optimize()
        assert master_model.Status == gp.GRB.Status.OPTIMAL
        y_opt = [y.x for y in master_model._y.values()]
        route = get_route(slave_model, y_opt, start + timelimit - time.time())
        print(y_opt)
        print('-----------------------'*10)
        print(f'objective of slave: {slave_model.objVal}')
        if route is None:
            print('No route added.')
            break
        routes.append(route)
        print(f'add route: {route.nodes}, cost: {route.cost}')
        update_master_model(master_model, route)

    print(f'{len(routes)} routes generated.')
    primal_model = build_primal_model(n_nodes, routes, n_vehicles)
    primal_model.optimize()
    assert primal_model.Status == gp.GRB.Status.OPTIMAL

    return [
        [nodes[i] for i in routes[r].nodes]
        for r, v in enumerate(z[r].x for z[r] in primal_model._z)
        if v == 1
    ]