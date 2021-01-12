import math
import time

import pulp
import numpy as np
import networkx as nx

from cvrp import utils


def build_model(n_nodes, dists, n_vehicles):
    nodes = list(range(n_nodes))

    model = pulp.LpProblem()
    x = np.array(pulp.LpVariable.matrix('x', (nodes, nodes), cat=pulp.LpBinary))
    utils.fix_value(np.diagonal(x), 0)

    objective = pulp.lpSum(dists*x)
    model.setObjective(objective)

    for i in nodes[1:]:
        model.addConstraint(pulp.lpSum(x[i, :]) == 1)
        model.addConstraint(pulp.lpSum(x[:, i]) == 1)  
    model.addConstraint(pulp.lpSum(x[0, :]) == n_vehicles)
    model.addConstraint(pulp.lpSum(x[:, 0]) == n_vehicles)

    model._x = x
    return model


def add_cuts(model, demands, capacity):
    # NetworkX のグラフに変換
    graph = nx.from_numpy_array(
        np.vectorize(lambda x: x.value())(model._x)
    )
    # 各トラックのルートと部分巡回路を取得
    graph.remove_node(0)
    tours = list(nx.connected_components(graph))

    n_cuts = 0
    for tour in tours:
        tour = list(tour)
        if len(tour) <= 1:
            continue

        # エッジ数と、必要なトラックの台数を計算
        n_edges = pulp.lpSum(model._x[tour, :][:, tour])
        v = math.ceil(demands[tour].sum()/capacity)

        constraint = n_edges <= len(tour) - v
        if not constraint.valid():
            model.addConstraint(constraint)
            n_cuts += 1
    return n_cuts


def decode_result(x, labels):
    labels = dict(enumerate(labels))
    try:
        graph = nx.from_numpy_array(x, create_using=nx.DiGraph)
    except TypeError:
        print('type error')
        return []
    graph = nx.relabel_nodes(graph, labels)
    return list(nx.simple_cycles(graph))


def solve(instance,  timelimit, log_dir, n_vehicles=None):
    nodes = list(instance.get_nodes())
    n_nodes = len(nodes)
    demands = np.array([instance.demands[node] for node in nodes])
    coords = np.array([instance.node_coords[node] for node in nodes])
    dists = utils.dists(coords, coords)
    capacity = instance.capacity
    if n_vehicles is None:
        n_vehicles = math.ceil(sum(demands)/capacity)

    start = time.time()
    model = build_model(n_nodes, dists, n_vehicles)
    # model.solver = utils.get_solver(warm_start=True, gapRel=1)
    model.solver = utils.get_solver(warm_start=True)#, gapRel=0.2)
    left = timelimit - (time.time() - start)

    flg = False
    tours = []
    while left > 0:
        model.solver.timeLimit = left
        #if  model.solve() != pulp.LpSolutionOptimal:
        #    print(tours)
        #    break
        assert model.solve() == pulp.LpSolutionOptimal
        if add_cuts(model, demands, capacity) == 0:
        #n_cuts, t = add_cuts(model, demands, capacity)
        #tours.append(t)
        #if n_cuts == 0:
            if flg:
                break
            #model.solver = utils.get_solver(warm_start=True, gapRel=1)
            flg = True
        left = timelimit - (time.time() - start)
    else:
        print('Time limit reached.')

    return decode_result(utils.get_values(model._x), nodes)