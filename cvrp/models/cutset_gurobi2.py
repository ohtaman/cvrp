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
    z = {}
    for a in range(n_vehicles):
        for i in nodes:
            for j in nodes:
                if i == j:
                    continue
                x[a, i, j] = model.addVar(vtype=gp.GRB.BINARY, name=f'x_{a}_{i}_{j}')
    for a in range(n_vehicles):
        for i in nodes:
            z[a, i] = model.addVar(vtype=gp.GRB.BINARY, name=f'z_{a}_{i}')

    model.setObjective(gp.quicksum(dists[k[1], k[2]]*v for k, v in x.items()), gp.GRB.MINIMIZE)

    for i in nodes:
        model.addConstr(gp.quicksum(z[a, i] for a in range(n_vehicles)) == 1)

    for a in range(n_vehicles):
        for i in nodes[1:]:
            model.addConstr(gp.quicksum(x[a, i, j] for j in nodes if i != j) == z[a, i])
            model.addConstr(gp.quicksum(x[a, j, i] for j in nodes if i != j) == z[a, i])
        model.addConstr(gp.quicksum(x[a, 0, j] for j in nodes[1:]) == 1)
        model.addConstr(gp.quicksum(x[a, j, 0] for j in nodes[1:]) == 1)

    for a in range(n_vehicles):
        model.addConstr(gp.quicksum(demands[i]*z[a, i] for i in nodes) <= capacity)

    model._x = x
    model._z = z
    model._capacity = capacity
    model._demands = demands
    model._n_vehicles = n_vehicles
    return model


def callback(model, where):
    if where != gp.GRB.Callback.MIPSOL:
        return
  
    edges = []
    for (a, i, j) in model._x:
        if model.cbGetSolution(model._x[a, i, j]) > 0.5:
            edges.append((i, j))

    graph = nx.from_edgelist(edges, create_using=nx.DiGraph)
    tours = nx.simple_cycles(graph)
    for i, tour in enumerate(tours):
        if 0 in tour:
            continue
        
        print(f'add cut for vehicle {a}, tour {i}, {tour}')
        for a in range(model._n_vehicles):
            n_edges = gp.quicksum(
                model._x[a, i, j]
                for i in tour
                for j in tour if i != j
            )
            model.cbLazy(n_edges <= len(tour) - 1)
    

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
    model.setParam('Heuristics', 1.0)
    model.setParam('MIPFocus', 1)
    model.setParam('LazyConstraints', 1)
    model.update()
    model.optimize(callback=callback)

    edges = []
    for (a, i, j), x in model._x.items():
        if x.x == 1:
            edges.append((i, j))
    graph = nx.from_edgelist(edges, create_using=nx.DiGraph)
    graph = nx.relabel_nodes(graph, dict(enumerate(nodes)))

    q = [0]*n_vehicles
    for (a, i), z in model._z.items():
        if z.x == 1:
            q[a] += demands[i]
    print(capacity, q)
    model.write('test.lp')

    return list(nx.simple_cycles(graph))