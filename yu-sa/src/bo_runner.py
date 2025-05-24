import sys
import os
import yaml
import json
import tempfile
import subprocess
from skopt.space import Integer, Real
from skopt import Optimizer
from pathlib import Path

# --- Parse parameter space from YAML ---
def parse_param_space(yaml_path):
    with open(yaml_path, 'r') as f:
        y = yaml.safe_load(f)
    params = y['parameters']
    search_space = []
    param_names = []
    param_types = {}
    param_defaults = {}
    for k, v in params.items():
        if not v.get('tune', True):
            continue
        rng = v.get('range', [])
        if not rng or (len(rng) == 2 and rng[0] == rng[1]):
            continue
        if v['type'] == 'int':
            # Support both [a, b] and [a, b, ...] forms
            if len(rng) == 2:
                search_space.append(Integer(rng[0], rng[1], name=k))
            else:
                search_space.append(Integer(min(rng), max(rng), name=k))
            param_types[k] = 'int'
        elif v['type'] == 'float':
            if len(rng) == 2:
                search_space.append(Real(rng[0], rng[1], name=k))
            else:
                search_space.append(Real(min(rng), max(rng), name=k))
            param_types[k] = 'float'
        else:
            continue
        param_names.append(k)
        param_defaults[k] = v['default']
    # Add untuned params with default
    for k, v in params.items():
        if k not in param_defaults:
            param_defaults[k] = v['default']
    return search_space, param_names, param_types, param_defaults, params

# --- Run solver using test.cpp binary ---
def run_solver(config, param_defaults, param_types, params_yaml, instance_dir, solver_exec, size='medium', num_runs=1):
    # Merge config with defaults for untuned params
    full_config = param_defaults.copy()
    full_config.update(config)
    # Build YAML for test.cpp
    yaml_params = {k: (int(v) if param_types.get(k, params_yaml[k]['type']) == 'int' else float(v)) for k, v in full_config.items()}
    temp_yaml = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump({size: {'data_dir': str(instance_dir), 'params': yaml_params, 'num_runs': num_runs, 'output_csv': '/tmp/paco_temp_output.csv'}}, temp_yaml)
    temp_yaml.close()
    cmd = [
        str(solver_exec),
        '--solver', os.environ.get('SOLVER_NAME', 'paco'),
        '--params', temp_yaml.name,
        '--instances', str(instance_dir),
        '--num-runs', str(num_runs),
        '--output', '/tmp/paco_temp_output.csv',
        '--size', size
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=2000)
        if result.returncode != 0:
            return float('inf')
        import re
        for line in result.stdout.splitlines():
            m = re.search(r"Obj = ([0-9.eE+-]+), Vehicles = (\d+), Time = ([0-9.eE+-]+)s", line)
            if m:
                return float(m.group(1))
        return float('inf')
    except Exception:
        return float('inf')
    finally:
        os.unlink(temp_yaml.name)
        if os.path.exists('/tmp/paco_temp_output.csv'):
            os.unlink('/tmp/paco_temp_output.csv')

# --- Main Bayesian Optimization Loop ---
def main():
    if len(sys.argv) < 4:
        print("Usage: bo_runner.py <param_yaml> <instance_dir> <max_evals> [num_runs]", file=sys.stderr)
        sys.exit(1)
    param_yaml = sys.argv[1]
    instance_dir = sys.argv[2]
    max_evals = int(sys.argv[3])
    num_runs = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    size = 'medium'
    paco_exec = Path(__file__).parent.parent.parent / "build" / "test"
    search_space, param_names, param_types, param_defaults, params_yaml_dict = parse_param_space(param_yaml)
    opt = Optimizer(
        dimensions=search_space,
        base_estimator="GP",
        acq_func="EI",
        acq_optimizer="auto",
        random_state=42
    )
    history = []
    # Initial random sampling
    init_samples = min(max(5, max_evals // 5), max_evals)
    for i in range(init_samples):
        x = opt.ask()
        config = {}
        for idx, k in enumerate(param_names):
            if param_types[k] == 'int':
                config[k] = int(round(x[idx]))
            else:
                config[k] = float(x[idx])
        score = run_solver(config, param_defaults, param_types, params_yaml_dict, instance_dir, paco_exec, size, num_runs)
        history.append((x, score))
    # BO loop
    evals = init_samples
    while evals < max_evals:
        X = [h[0] for h in history]
        y = [h[1] for h in history]
        opt.tell(X, y)
        x = opt.ask()
        config = {}
        for idx, k in enumerate(param_names):
            if param_types[k] == 'int':
                config[k] = int(round(x[idx]))
            else:
                config[k] = float(x[idx])
        score = run_solver(config, param_defaults, param_types, params_yaml_dict, instance_dir, paco_exec, size, num_runs)
        history.append((x, score))
        evals += 1
    # Output best config
    best_idx = min(range(len(history)), key=lambda i: history[i][1])
    best_x = history[best_idx][0]
    best_config = {}
    for idx, k in enumerate(param_names):
        if param_types[k] == 'int':
            best_config[k] = int(round(best_x[idx]))
        else:
            best_config[k] = float(best_x[idx])
    for k, v in params_yaml_dict.items():
        if k not in best_config:
            best_config[k] = v['default']
    print(json.dumps(best_config))

if __name__ == '__main__':
    main()