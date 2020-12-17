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
    for i in nodes[1:]:
        for j in nodes[i + 1:]:
            x[i, j] = model.addVar(vtype=gp.GRB.BINARY, name=f'x_{i}_{j}')
    for j in nodes[1:]:
        x[0, j] = model.addVar(vtype=gp.GRB.INTEGER, ub=2, name=f'x_{0}_{j}')

    model.setObjective(gp.quicksum(dists[k]*v for k, v in x.items()), gp.GRB.MINIMIZE)

    for i in nodes[1:]:
       model.addConstr(gp.quicksum(x[j, i] for j in nodes[:i]) + gp.quicksum(x[i, j] for j in nodes[i + 1:]) == 2)
    model.addConstr(gp.quicksum(x[0, j] for j in nodes[1:]) == 2*n_vehicles)

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
    graph = nx.from_edgelist(edges)
    graph.remove_node(0)
    for tour in nx.connected_components(graph):
        tour = list(tour)
        n_vehicles = math.ceil(model._demands[tour].sum()/model._capacity)
        n_edges = gp.quicksum(
            model._x[i, j]
            for i in tour
            for j in tour if i < j
        )
        if len(tour) >= 2:
            # print(f'add cut for {tour}')
            model.cbLazy(n_edges <= len(tour) - n_vehicles)
    

def solve(instance,  timelimit, log_dir):
    nodes = list(instance.get_nodes())
    n_nodes = len(nodes)
    demands = np.array([instance.demands[node] for node in nodes])
    coords = np.array([instance.node_coords[node] for node in nodes])
    dists = utils.dists(coords, coords)
    capacity = instance.capacity
    n_vehicles = math.ceil(sum(demands)/capacity)

    model = build_model(n_nodes, dists, demands, capacity, n_vehicles)
    model.setParam('TimeLimit', timelimit)
    # model.setParam('Heuristics', 1.0)
    # model.setParam('MIPFocus', 1)
    model.setParam('LazyConstraints', 1)
    model.optimize(callback=callback)
    assert model.Status in (gp.GRB.Status.OPTIMAL, gp.GRB.Status.TIME_LIMIT)

    edges = []
    for (i, j), x in model._x.items():
        if x.x >= 0.5:
            edges.append((i, j))
    graph = nx.from_edgelist(edges)
    graph.remove_node(0)
    graph = nx.relabel_nodes(graph, dict(enumerate(nodes)))
    return [[nodes[0]] + list(tour) for tour in nx.connected_components(graph)]