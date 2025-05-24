#!/usr/bin/env python3
"""
PACO Analysis Script

Runs sensitivity or scalability analysis for the PACO algorithm.
"""

import argparse
import yaml
import csv
import subprocess
import tempfile
import os
from pathlib import Path
from statistics import mean, stdev

def load_parameters(param_file):
    with open(param_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_temp_param_file(params, instance_path, size="medium"):
    temp_config = {
        size: {
            "data_dir": str(Path(instance_path).parent),
            "params": params,
            "num_runs": 1,
            "output_csv": "/tmp/paco_temp_output.csv"
        }
    }
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(temp_config, temp_file, default_flow_style=False)
    temp_file.close()
    return temp_file.name

def run_paco(params, instance_path, paco_exec, size="medium"):
    temp_param_file = create_temp_param_file(params, instance_path, size)
    try:
        cmd = [
            str(paco_exec),
            "--solver", "paco",
            "--params", temp_param_file,
            "--instances", str(Path(instance_path).parent),
            "--num-runs", "1",
            "--output", "/tmp/paco_temp_output.csv",
            "--size", size,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20000)
        if result.returncode != 0:
            print(f"PACO failed: {result.stderr}")
            return None, None
        # Parse stdout for summary line:   Obj = <objective_value>, Vehicles = <num_vehicles>, Time = <runtime>s
        import re
        obj, runtime = None, None
        for line in result.stdout.splitlines():
            m = re.search(r"Obj = ([0-9.eE+-]+), Vehicles = (\d+), Time = ([0-9.eE+-]+)s", line)
            if m:
                obj = float(m.group(1))
                runtime = float(m.group(3))
                break
        if obj is not None and runtime is not None:
            return runtime, obj
        else:
            print("Could not parse PACO output:\n", result.stdout)
            return None, None
    except Exception as e:
        print(f"Error running PACO: {e}")
        return None, None
    finally:
        if os.path.exists(temp_param_file):
            os.unlink(temp_param_file)
        if os.path.exists("/tmp/paco_temp_output.csv"):
            os.unlink("/tmp/paco_temp_output.csv")

def sensitivity_analysis(args, config, paco_exec):
    print("Running sensitivity analysis...")
    parameters = config['parameters']
    default_params = {k: v['default'] for k, v in parameters.items()}
    results = []
    for param_name, param_cfg in parameters.items():
        if not param_cfg.get('range'):
            continue
        for value in param_cfg['range']:
            test_params = default_params.copy()
            test_params[param_name] = value
            runtimes, qualities = [], []
            for _ in range(args.num_runs):
                print(f"Run {_ + 1}/{args.num_runs} for {param_name} = {value}")
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
                    'instance': Path(args.instance_file).name
                }
                results.append(result_row)
                print(f"Results for {param_name} = {value}:")
                print(f"  Avg Runtime: {result_row['avg_runtime']:.2f} ± {result_row['std_runtime']:.2f}")
                print(f"  Avg Solution Quality: {result_row['avg_solution_quality']:.2f} ± {result_row['std_solution_quality']:.2f}")
                # Write/append to CSV after each configuration
                write_mode = 'w' if not os.path.exists(args.output) or os.path.getsize(args.output) == 0 else 'a'
                with open(args.output, write_mode, newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=result_row.keys())
                    if write_mode == 'w':
                        writer.writeheader()
                    writer.writerow(result_row)
    print(f"Sensitivity analysis complete. Results saved to {args.output}")

def scalability_analysis(args, config, paco_exec):
    parameters = config['parameters']
    default_params = {k: v['default'] for k, v in parameters.items()}
    thread_counts = config.get('threading', {}).get('thread_counts', [1, 2, 4, 8])
    results = []
    for threads in thread_counts:
        test_params = default_params.copy()
        test_params['p'] = threads
        runtimes, qualities = [], []
        for _ in range(args.num_runs):
            runtime, quality = run_paco(test_params, args.instance_file, paco_exec, args.size)
            if runtime is not None and quality is not None:
                runtimes.append(runtime)
                qualities.append(quality)
        if runtimes:
            results.append({
                'thread_count': threads,
                'avg_runtime': mean(runtimes),
                'std_runtime': stdev(runtimes) if len(runtimes) > 1 else 0.0,
                'avg_solution_quality': mean(qualities),
                'std_solution_quality': stdev(qualities) if len(qualities) > 1 else 0.0,
                'num_successful_runs': len(runtimes),
                'instance': Path(args.instance_file).name
            })
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Scalability analysis complete. Results saved to {args.output}")

def speedup_analysis(args, config, paco_exec):
    print("Running speedup/efficiency analysis...")
    parameters = config['parameters']
    default_params = {k: v['value'] if 'value' in v and not isinstance(v['value'], list) else v.get('default', None) for k, v in parameters.items()}
    # Get total_ants and thread counts from config
    total_ants = parameters.get('total_ants', {}).get('value', 6400)
    thread_counts = parameters.get('p', {}).get('value', [1, 2, 4, 8, 16, 32, 64])
    results = []
    baseline_runtime = None
    for threads in thread_counts:
        m = max(1, total_ants // threads)
        test_params = default_params.copy()
        test_params['p'] = threads
        test_params['m'] = m
        runtimes, qualities = [], []
        for _ in range(args.num_runs):
            print(f"Run {_ + 1}/{args.num_runs} for threads = {threads}, m = {m}")
            runtime, quality = run_paco(test_params, args.instance_file, paco_exec, args.size)
            if runtime is not None and quality is not None:
                runtimes.append(runtime)
                qualities.append(quality)
        if runtimes:
            avg_runtime = mean(runtimes)
            avg_quality = mean(qualities)
            std_runtime = stdev(runtimes) if len(runtimes) > 1 else 0.0
            std_quality = stdev(qualities) if len(qualities) > 1 else 0.0
            if threads == thread_counts[0]:
                baseline_runtime = avg_runtime
            speedup = baseline_runtime / avg_runtime if baseline_runtime else 1.0
            efficiency = speedup / threads
            result_row = {
                'thread_count': threads,
                'm': m,
                'avg_runtime': avg_runtime,
                'std_runtime': std_runtime,
                'avg_solution_quality': avg_quality,
                'std_solution_quality': std_quality,
                'speedup': speedup,
                'efficiency': efficiency,
                'instance': Path(args.instance_file).name
            }
            results.append(result_row)
            print(f"Threads: {threads}, m: {m}, Speedup: {speedup:.2f}, Efficiency: {efficiency:.2f}")
        if results:
            with open(args.output, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            print(f"Speedup/efficiency analysis complete. Results saved to {args.output}")

def main():
    parser = argparse.ArgumentParser(description="PACO Algorithm Analysis Script")
    parser.add_argument('--size', type=str, default='medium', help='Problem instance size')
    parser.add_argument('--instance-file', type=str, required=True, help='Path to problem instance file')
    parser.add_argument('--parameters', type=str, required=True, help='Path to PACO parameters YAML file')
    parser.add_argument('--output', type=str, required=True, help='Path to output CSV file')
    parser.add_argument('--experiment', type=str, choices=['sensitivity', 'scalability', 'speedup'], default='sensitivity')
    parser.add_argument('--num-runs', type=int, default=5, help='Number of repetitions for each configuration')
    args = parser.parse_args()

    config = load_parameters(args.parameters)
    paco_exec = Path(__file__).parent.parent.parent / "build" / "test"
    print(f"Using PACO executable at: {paco_exec}")
    if not paco_exec.exists():
        print(f"Error: PACO executable not found at {paco_exec}")
        exit(1)

    if args.experiment == 'sensitivity':
        sensitivity_analysis(args, config, paco_exec)
    elif args.experiment == 'scalability':
        scalability_analysis(args, config, paco_exec)
    elif args.experiment == 'speedup':
        speedup_analysis(args, config, paco_exec)

if __name__ == '__main__':
    main()