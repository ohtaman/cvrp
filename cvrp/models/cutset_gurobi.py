import collections
import math
import time

import numpy as np
import networkx as nx
import gurobipy as gp

from cvrp import utils


def build_model(n_nodes, dists, demands, capacity, n_vehicles):
    nodes = list(range(n_nodes))

    model = gp.Model('cvrp.cutset_gurobi')
    
    x = {}
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            x[i, j] = model.addVar(vtype=gp.GRB.BINARY, name=f'x_{i}_{j}')

    model.setObjective(gp.quicksum(dists[k]*v for k, v in x.items()), gp.GRB.MINIMIZE)

    for i in nodes[1:]:
        model.addConstr(gp.quicksum(x[i, j] for j in nodes if i != j) == 1)
        model.addConstr(gp.quicksum(x[j, i] for j in nodes if i != j) == 1)
    model.addConstr(gp.quicksum(x[0, j] for j in nodes[1:]) == n_vehicles)
    model.addConstr(gp.quicksum(x[j, 0] for j in nodes[1:]) == n_vehicles)

    model._x = x
    model._capacity = capacity
    model._demands = demands
    return model


def callback(model, where):
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
            print(f'add cut for {tour}')
            model.cbLazy(n_edges <= len(tour) - n_vehicles)
    

def solve(instance,  timelimit, log_dir, n_vehicles=None):
    nodes = list(instance.get_nodes())
    n_nodes = len(nodes)
    demands = np.array([instance.demands[node] for node in nodes])
    coords = np.array([instance.node_coords[node] for node in nodes])
    dists = utils.dists(coords, coords)
    capacity = instance.capacity
    if n_vehicles is None:
        n_vehicles = math.ceil(sum(demands)/capacity)

    model = build_model(n_nodes, dists, demands, capacity, n_vehicles)
    model.setParam('TimeLimit', timelimit)
    model.setParam('LazyConstraints', 1)
    model.optimize(callback=callback)
    assert model.Status in (gp.GRB.Status.OPTIMAL, gp.GRB.Status.TIME_LIMIT)

    edges = []
    for (i, j), x in model._x.items():
        if x.x >= 0.5:
            edges.append((i, j))
    graph = nx.from_edgelist(edges, create_using=nx.DiGraph)
    graph = nx.relabel_nodes(graph, dict(enumerate(nodes)))
    return list(nx.simple_cycles(graph))