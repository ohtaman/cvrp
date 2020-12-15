import argparse
import contextlib
import importlib
import math
import os
import shutil
import sys
import time

import mlflow
import pandas as pd
import psutil
import tsplib95

from cvrp import utils


TIME_LIMIT_DEFAULT = 10*60


def parse_args(argv):
    parser = argparse.ArgumentParser('cvrp')
    parser.add_argument(
        'instance',
    )
    parser.add_argument(
        '-m',
        '--model',
        help='model (cutset/flow/mflow/mtz)',
        default='cutset'
    )
    parser.add_argument(
        '-t',
        '--time',
        help='time limit for MIP solver (sec)',
        type=float,
        default=TIME_LIMIT_DEFAULT
    )
    parser.add_argument(
        '-a',
        '--log_dir',
        help='path to the log_dir directory',
        default='logs'
    )
    return parser.parse_args(argv[1:])


def main(argv=sys.argv):
    args = parse_args(argv)
    log_dir = f'{args.log_dir}/{time.time()}'
    os.makedirs(log_dir, exist_ok=True)

    logfile = open(f'{log_dir}/stdout.log', 'w')
    org_std_out = os.dup(sys.stdout.fileno())
    # os.dup2(logfile.fileno(), sys.stdout.fileno())

    mlflow.set_experiment('cvrp')
    with mlflow.start_run():
        mlflow.log_params(dict(
            vcpus=psutil.cpu_count(),
            cpus=psutil.cpu_count(logical=False),
            cpu_freq=psutil.cpu_freq().current,
            mem_total=psutil.virtual_memory().total,
            mem_free=psutil.virtual_memory().free
        ))
        mlflow.log_params(dict(
            model=args.model,
            timel_imit=args.time,
            instance=args.instance
        ))

        start = time.time()
        model = importlib.import_module(f'cvrp.models.{args.model}')
        instance = tsplib95.load(args.instance)
        result = model.solve(instance, timelimit=args.time, log_dir=log_dir)

        cost = 0
        for tour in result:
            cost += sum(utils.dist(instance.node_coords[tour[i]], instance.node_coords[tour[i + 1]]) for i in range(-1, len(tour) - 1))

        mlflow.log_metrics(dict(
                elapsed_time=time.time() - start,
                cost=cost
        ))

        os.dup2(org_std_out, sys.stdout.fileno())
        os.close(org_std_out)
        logfile.close()

        mlflow.log_artifacts(log_dir)

if __name__ == '__main__':
    main()