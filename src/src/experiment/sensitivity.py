"""Sensitivity analysis CLI for PACO.

Previously provided via analysis.py --experiment sensitivity
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from statistics import mean, stdev

from .common import load_parameters, run_paco, get_paco_exec


def sensitivity(args, config, paco_exec):
    print("Running sensitivity analysis...")
    parameters = config['parameters']
    default_params = {k: v['default'] for k, v in parameters.items()}

    # Resolve instance input (file or directory); prefer --instance-file if provided
    instance_input = args.instance_file or args.instance
    if not instance_input:
        print("Error: must provide --instance-file (file/dir) or --instance (dir)")
        return
    inst_path = Path(instance_input)
    if not inst_path.exists():
        print(f"Error: instance path not found: {inst_path}")
        return

    if inst_path.is_dir():
        candidates = [p for p in inst_path.iterdir() if p.is_file() and p.suffix == '.txt']
        if not candidates:
            print(f"Error: no .txt instance files in directory {inst_path}")
            return
        if args.seed is not None:
            random.seed(args.seed)
        sample_n = args.sample_size if args.sample_size and args.sample_size > 0 else len(candidates)
        sample_n = min(sample_n, len(candidates))
        if sample_n < len(candidates):
            instances = random.sample(candidates, sample_n)
        else:
            instances = sorted(candidates)
        print(f"Selected {len(instances)} instance(s) from {inst_path}")
    else:
        instances = [inst_path]

    # Baseline per instance
    baseline = {}
    print("Running baselines (default parameters)...")
    for inst in instances:
        runtimes, qualities = [], []
        for run_idx in range(args.num_runs):
            print(f"Baseline {inst.name} run {run_idx + 1}/{args.num_runs}")
            runtime, quality = run_paco(default_params, str(inst), paco_exec, args.size)
            if runtime is not None and quality is not None:
                if quality == 0:
                    raise ValueError(f"Invalid solution quality (0) for baseline {inst.name}")
                runtimes.append(runtime)
                qualities.append(quality)
        if runtimes:
            avg_rt = mean(runtimes)
            std_rt = stdev(runtimes) if len(runtimes) > 1 else 0.0
            avg_q = mean(qualities)
            std_q = stdev(qualities) if len(qualities) > 1 else 0.0
            baseline[inst] = (avg_rt, std_rt, avg_q, std_q)
            print(f"Baseline {inst.name}: Runtime {avg_rt:.2f} ± {std_rt:.2f}; Quality {avg_q:.2f} ± {std_q:.2f}")
        else:
            print(f"Warning: no successful baseline runs for {inst.name}; excluding from analysis")

    if not baseline:
        print("Error: no baseline data collected; aborting")
        return

    detail_header = ['param_name','param_value','instance','avg_runtime','std_runtime','avg_solution_quality','std_solution_quality','runtime_dvt','solution_quality_dvt']
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=detail_header)
        writer.writeheader()
        for inst,(brt, brt_std, bq, bq_std) in baseline.items():
            writer.writerow({
                'param_name':'BASELINE','param_value':'DEFAULT','instance':inst.name,
                'avg_runtime':brt,'std_runtime':brt_std,
                'avg_solution_quality':bq,'std_solution_quality':bq_std,
                'runtime_dvt':0.0,'solution_quality_dvt':0.0
            })

    summary_rows = []

    for param_name, param_cfg in parameters.items():
        if not param_cfg.get('range'):
            continue
        for value in param_cfg['range']:
            per_inst_rt_dvt = []
            per_inst_q_dvt = []
            for inst, (brt, brt_std, bq, bq_std) in baseline.items():
                # reuse baseline if value is default
                if value == default_params.get(param_name):
                    per_inst_rt_dvt.append(0.0)
                    per_inst_q_dvt.append(0.0)
                    with open(args.output, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=detail_header)
                        writer.writerow({
                            'param_name':param_name,'param_value':value,'instance':inst.name,
                            'avg_runtime':brt,'std_runtime':brt_std,
                            'avg_solution_quality':bq,'std_solution_quality':bq_std,
                            'runtime_dvt':0.0,'solution_quality_dvt':0.0
                        })
                    continue
                test_params = default_params.copy(); test_params[param_name] = value
                runtimes, qualities = [], []
                for run_idx in range(args.num_runs):
                    print(f"Run {param_name}={value} {inst.name} {run_idx + 1}/{args.num_runs}")
                    runtime, quality = run_paco(test_params, str(inst), paco_exec, args.size)
                    if runtime is not None and quality is not None:
                        if quality == 0:
                            raise ValueError(f"Invalid solution quality (0) for {param_name}={value} on {inst.name}")
                        runtimes.append(runtime); qualities.append(quality)
                if not runtimes:
                    print(f"Warning: no successful runs for {param_name}={value} on {inst.name}; skipping instance")
                    continue
                avg_rt = mean(runtimes); std_rt = stdev(runtimes) if len(runtimes) > 1 else 0.0
                avg_q = mean(qualities); std_q = stdev(qualities) if len(qualities) > 1 else 0.0
                rt_dvt = ((avg_rt - brt)/brt*100.0) if brt else 0.0
                q_dvt = ((avg_q - bq)/bq*100.0) if bq else 0.0
                per_inst_rt_dvt.append(rt_dvt); per_inst_q_dvt.append(q_dvt)
                with open(args.output, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=detail_header)
                    writer.writerow({
                        'param_name':param_name,'param_value':value,'instance':inst.name,
                        'avg_runtime':avg_rt,'std_runtime':std_rt,
                        'avg_solution_quality':avg_q,'std_solution_quality':std_q,
                        'runtime_dvt':rt_dvt,'solution_quality_dvt':q_dvt
                    })
            if per_inst_rt_dvt:
                avg_rt_dvt = mean(per_inst_rt_dvt)
                avg_q_dvt = mean(per_inst_q_dvt) if per_inst_q_dvt else 0.0
                summary_rows.append({
                    'param_name':param_name,
                    'param_value':value,
                    'avg_runtime_dvt':avg_rt_dvt,
                    'avg_solution_quality_dvt':avg_q_dvt,
                    'num_instances':len(per_inst_rt_dvt)
                })
                print(f"Summary {param_name}={value}: runtime dev {avg_rt_dvt:+.2f}% quality dev {avg_q_dvt:+.2f}% over {len(per_inst_rt_dvt)} instance(s)")

    summary_path = args.summary_output or f"{args.output}.summary.csv"
    if summary_rows:
        with open(summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['param_name','param_value','avg_runtime_dvt','avg_solution_quality_dvt','num_instances'])
            writer.writeheader(); writer.writerows(summary_rows)
        print(f"Wrote summary to {summary_path}")
    print(f"Sensitivity analysis complete. Detail: {args.output}")


def build_arg_parser():
    p = argparse.ArgumentParser(description="PACO Sensitivity Analysis (multi-instance aware)")
    p.add_argument('--size', type=str, default='medium', help='Problem instance size label')
    p.add_argument('--instance-file', type=str, help='Path to a single instance file or directory of instances')
    p.add_argument('--instance', type=str, help='(Deprecated) Path to instance directory (kept for backward compatibility)')
    p.add_argument('--sample-size', type=int, default=0, help='If directory provided, number of instances to sample (0=all)')
    p.add_argument('--seed', type=int, default=None, help='Random seed for sampling')
    p.add_argument('--parameters', type=str, required=True, help='Path to PACO parameters YAML file')
    p.add_argument('--output', type=str, required=True, help='Path to detailed output CSV file')
    p.add_argument('--summary-output', type=str, default=None, help='Path for summary CSV (default: output + .summary.csv)')
    p.add_argument('--num-runs', type=int, default=5, help='Number of repetitions per (instance, configuration)')
    return p


def main():  # pragma: no cover - CLI entry
    args = build_arg_parser().parse_args()
    config = load_parameters(args.parameters)
    # Print the loaded configurations
    print("Loaded PACO configurations:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    paco_exec = get_paco_exec(__file__)
    if not paco_exec.exists():
        print(f"Error: PACO executable not found at {paco_exec}")
        raise SystemExit(1)
    sensitivity(args, config, paco_exec)


if __name__ == '__main__':  # pragma: no cover
    main()
