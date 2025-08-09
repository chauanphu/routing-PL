"""Sensitivity analysis CLI for PACO.

Previously provided via analysis.py --experiment sensitivity
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from statistics import mean, stdev

from .common import load_parameters, run_paco, get_paco_exec


def sensitivity(args, config, paco_exec):
    print("Running sensitivity analysis...")
    parameters = config['parameters']
    default_params = {k: v['default'] for k, v in parameters.items()}
    for param_name, param_cfg in parameters.items():
        if not param_cfg.get('range'):
            continue
        for value in param_cfg['range']:
            test_params = default_params.copy()
            test_params[param_name] = value
            runtimes, qualities = [], []
            for run_idx in range(args.num_runs):
                print(f"Run {run_idx + 1}/{args.num_runs} for {param_name} = {value}")
                runtime, quality = run_paco(test_params, args.instance_file, paco_exec, args.size)
                if runtime is not None and quality is not None:
                    runtimes.append(runtime)
                    qualities.append(quality)
            if runtimes:
                result_row = {
                    'param_name': param_name,
                    'param_value': value,
                    'avg_runtime': mean(runtimes),
                    'std_runtime': stdev(runtimes) if len(runtimes) > 1 else 0.0,
                    'avg_solution_quality': mean(qualities),
                    'std_solution_quality': stdev(qualities) if len(qualities) > 1 else 0.0,
                    'instance': Path(args.instance_file).name,
                }
                write_mode = 'w' if not os.path.exists(args.output) or os.path.getsize(args.output) == 0 else 'a'
                with open(args.output, write_mode, newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=result_row.keys())
                    if write_mode == 'w':
                        writer.writeheader()
                    writer.writerow(result_row)
                print(f"Results for {param_name} = {value}: Avg Runtime {result_row['avg_runtime']:.2f} ± {result_row['std_runtime']:.2f}; Avg Quality {result_row['avg_solution_quality']:.2f} ± {result_row['std_solution_quality']:.2f}")
    print(f"Sensitivity analysis complete. Results saved to {args.output}")


def build_arg_parser():
    p = argparse.ArgumentParser(description="PACO Sensitivity Analysis")
    p.add_argument('--size', type=str, default='medium', help='Problem instance size')
    p.add_argument('--instance-file', type=str, required=True, help='Path to problem instance file')
    p.add_argument('--parameters', type=str, required=True, help='Path to PACO parameters YAML file')
    p.add_argument('--output', type=str, required=True, help='Path to output CSV file')
    p.add_argument('--num-runs', type=int, default=5, help='Number of repetitions for each configuration')
    return p


def main():  # pragma: no cover - CLI entry
    args = build_arg_parser().parse_args()
    config = load_parameters(args.parameters)
    paco_exec = get_paco_exec(__file__)
    if not paco_exec.exists():
        print(f"Error: PACO executable not found at {paco_exec}")
        raise SystemExit(1)
    sensitivity(args, config, paco_exec)


if __name__ == '__main__':  # pragma: no cover
    main()
