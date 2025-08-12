#!/usr/bin/env python3
"""
Bayesian Optimization Tuning Script for VRPPL Solvers
"""
import argparse
import yaml
import subprocess
import json
from pathlib import Path
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

# --- Helper functions ---
def load_tune_config(tune_file, size):
    with open(tune_file, 'r') as f:
        config = yaml.safe_load(f)
    if size not in config:
        raise ValueError(f"Size '{size}' not found in tune file {tune_file}")
    return config[size]

def build_search_space(param_grid):
    space = []
    param_names = []
    for k, v in param_grid.items():
        if v.get('tune', True) is False:
            continue
        param_names.append(k)
        if len(v['range']) == 0:
            continue
        if v['type'] == 'int':
            if isinstance(v['range'], list) and len(v['range']) > 2:
                space.append(Categorical(v['range'], name=k))
            else:
                space.append(Integer(v['range'][0], v['range'][-1], name=k))
        elif v['type'] == 'float':
            space.append(Real(v['range'][0], v['range'][-1], name=k))
        elif v['type'] == 'categorical':
            space.append(Categorical(v['range'], name=k))
    return space, param_names

def update_param_file(param_file_path, params, base_param_grid, instance_dir, size, output_csv):
    # Load the existing param file
    with open(param_file_path, 'r') as f:
        config = yaml.safe_load(f)
    # Update the params for the given size
    if size not in config:
        config[size] = {}
    if 'params' not in config[size]:
        config[size]['params'] = {}
    for k, v in base_param_grid.items():
        def to_scalar(val):
            # Convert numpy types to native python types
            if hasattr(val, 'item'):
                val = val.item()
            if isinstance(val, list):
                return val[0] if len(val) == 1 else val
            return val
        if v.get('tune', True) is False:
            config[size]['params'][k] = to_scalar(v['default'])
        elif k in params:
            config[size]['params'][k] = to_scalar(params[k])
        else:
            config[size]['params'][k] = to_scalar(v['default'])
    config[size]['data_dir'] = str(instance_dir)
    config[size]['num_runs'] = 1
    config[size]['output_csv'] = output_csv
    with open(param_file_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def run_solver(solver, param_file, instance_file, test_exec, size):
    cmd = [
        str(test_exec),
        '--solver', solver,
        '--params', param_file,
        '--size', size,
        '--instance-file', str(instance_file),
        '--verbose', '1',
        '--output', '/tmp/bao_temp_output.csv'
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        if result.returncode != 0:
            print(f"Solver failed: {result.stderr}")
            return float('inf'), float('inf')
        import re
        for line in result.stdout.splitlines():
            m = re.search(r"Obj = ([0-9.eE+-]+), Vehicles = (\d+), Time = ([0-9.eE+-]+)s", line)
            if m:
                obj = float(m.group(1))
                time = float(m.group(3))
                return obj, time
        print("Could not parse solver output:\n", result.stdout)
        return float('inf'), float('inf')
    except Exception as e:
        print(f"Error running solver: {e}")
        return float('inf'), float('inf')

def main():
    parser = argparse.ArgumentParser(description="Bayesian Optimization Tuning for VRPPL Solvers")
    parser.add_argument('--solver', type=str, required=True, help='Solver name (e.g., paco, sa, ga, aco-ts)')
    parser.add_argument('--tune-file', type=str, required=True, help='Path to tune YAML file')
    parser.add_argument('--param-file', type=str, required=True, help='Path to static param YAML file to update')
    parser.add_argument('--instance-dir', type=str, required=True, help='Directory containing instances')
    parser.add_argument('--size', type=str, default='small', help='Experiment size (small, medium, large)')
    parser.add_argument('--n-calls', type=int, default=30, help='Number of BO iterations')
    parser.add_argument('--output', type=str, default='bao_tuning_result.json', help='Output file for best params')
    parser.add_argument('--runtime-weight', type=float, default=0.1, help='Weight for runtime in the objective function')
    parser.add_argument('--test-exec', type=str, default='../../build/test', help='Path to compiled test executable')
    args = parser.parse_args()

    param_grid = load_tune_config(args.tune_file, args.size)
    space, param_names = build_search_space(param_grid)
    instance_dir = Path(args.instance_dir)
    test_exec = Path(args.test_exec)
    size = args.size
    output_csv = '/tmp/bao_temp_output.csv'
    param_file_path = args.param_file
    runtime_weight = args.runtime_weight

    # Use first instance in dir for tuning
    instance_files = sorted([f for f in instance_dir.iterdir() if f.is_file()])
    if not instance_files:
        print(f"No instance files found in {instance_dir}")
        exit(1)
    instance_file = instance_files[0]

    @use_named_args(space)
    def objective(**params):
        print(f"Testing params: {params}")
        update_param_file(param_file_path, params, param_grid, instance_dir, size, output_csv)
        obj, time = run_solver(args.solver, param_file_path, instance_file, test_exec, size)
        
        # Combine objective value and runtime into a single score
        # The weight determines how much to penalize runtime.
        # A higher weight means runtime is more important.
        score = obj + runtime_weight * time
        
        print(f"Objective value: {obj}, Time: {time}s, Combined score: {score}")
        return score

    res = gp_minimize(objective, space, n_calls=args.n_calls, random_state=42, verbose=True)
    best_params = dict(zip(param_names, res.x))
    print(f"Best params: {best_params}")
    print(f"Best objective: {res.fun}")
    with open(args.output, 'w') as f:
        json.dump({'best_params': best_params, 'best_objective': res.fun}, f, indent=2)

if __name__ == '__main__':
    main()
