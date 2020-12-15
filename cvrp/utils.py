import multiprocessing

import numpy as np
import pandas as pd
import pulp


@np.vectorize
def fix_value(x, value):
    x.bounds(value, value)
    return x


@np.vectorize
def get_values(x):
    return x.value()


@np.vectorize
def set_category(x, cat, initial_value=None):
    x.cat = cat
    if initial_value:
        x.setInitialValue(initial_value)
    return x


def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))



def get_dists(coords):
    return np.array([[
            np.linalg.norm([coords[a][0] - coords[b][0], coords[a][1] - coords[b][1]])
            for b in coords.keys()
        ]
        for a in coords.keys()
    ])


def dists(coord1, coord2):
    return np.array([[
            np.linalg.norm(np.array(a) - np.array(b))
            for b in coord2
        ]
        for a in coord1
    ])

def get_solver(msg=True, timelimit=None, warm_start=True, **kwargs):
    if pulp.GUROBI_CMD().available():
        return pulp.GUROBI_CMD(msg=msg, warmStart=warm_start, timeLimit=timelimit, **kwargs)
    elif pulp.MIPCL_CMD().available():
        return pulp.MIPCL_CMD(msg=msg, timeLimit=timelimit, **kwargs)
    elif pulp.COIN_CMD().available():
        return pulp.COIN_CMD(msg=msg, warmStart=warm_start, threads=multiprocessing.cpu_count(), timeLimit=timelimit, **kwargs)
    else:
        return pulp.PULP_CBC_CMD(msg=True, warmStart=warm_start, threads=multiprocessing.cpu_count(), timeLimit=timelimit, **kwargs)
