import csv
import itertools
import os
from pathlib import Path
import time
from statistics import mean, stdev
import yaml
import math
from tqdm import tqdm
from meta.solver import Problem
from meta.ACO import PACO
import matplotlib.pyplot as plt

def run_paco_instance(problem, num_ants, batch_size, **other_params):
    """
    Runs a single PACO instance.
    Each run is executed sequentially (not in parallel) to keep the cores free for the parallel PACO internal operations.
    It creates a fresh PACO instance, runs optimize(), measures runtime,
    and returns a dictionary of metrics including the number of vehicles used.
    """

    paco_instance = PACO(problem,
                         num_ants=num_ants,
                         batch_size=batch_size,
                         **other_params)
    
    start_time = time.time()
    result, fitness, routes = paco_instance.optimize(verbose=False)  # Capture routes
    end_time = time.time()
    total_runtime = end_time - start_time

    try:
        overhead = paco_instance.overhead  # Assume PACO stores an overall overhead metric
    except AttributeError:
        overhead = None

    return {
        'num_ants': num_ants,
        'batch_size': batch_size,
        'best_fitness': fitness,  # Use fitness directly from return
        'runtime': total_runtime,
        'overhead': overhead,
        'num_vehicles': len(routes) if routes else 0  # Add number of vehicles
    }

def run_solver_instance(solver_class, problem, solver_params):
    """
    Runs a solver (SACO or similar) instance and returns metrics.
    """
    solver = solver_class(problem, **solver_params)
    start_time = time.time()
    result = solver.optimize(verbose=False)
    end_time = time.time()
    total_runtime = end_time - start_time
    return {
        'best_fitness': result[1],
        'runtime': total_runtime
    }

def speed_up(config_path='speedup.yaml'):
    """
    Runs PACO with varying core counts, measures runtime, computes speedup and efficiency.
    Results are saved to output_dir/paco_scaling_vs_cores.csv.
    Also exports line charts for speedup and efficiency.
    """

    # Load config
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    exp_config = config['experiment']
    core_counts = exp_config['core_counts']
    output_dir = exp_config['output_dir']
    samples = exp_config['samples']

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results_file = os.path.join(output_dir, 'scaling_vs_cores.csv')
    header = ['instance_scale', 'instance_name', 'core_count', 'num_ants', 'batch_size', 'runtime', 'speedup', 'efficiency']

    print(f"Starting PACO scaling vs core analysis. Results will be saved to: {results_file}")
    with open(results_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for scale_name, scale_config in samples.items():
            instance_files = scale_config['instances']
            data_dir = scale_config['data_dir']
            paco_params = scale_config['paco_params']
            for instance_name in tqdm(instance_files):
                instance_path = os.path.join(data_dir, instance_name)
                print(f"  Loading Instance: {instance_path}")
                try:
                    problem = Problem()
                    problem.load_data(instance_path)
                except Exception as e:
                    print(f"    Error loading instance {instance_path}: {e}. Skipping.")
                    continue
                runtimes = []
                # For plotting
                speedups = []
                efficiencies = []
                for core in core_counts:
                    num_ants = paco_params['num_ants']
                    batch_size = num_ants // core
                    paco_args = dict(paco_params)
                    # Set process pool size via environment variable
                    os.environ['OMP_NUM_THREADS'] = str(core)
                    os.environ['NUMEXPR_MAX_THREADS'] = str(core)
                    os.environ['MKL_NUM_THREADS'] = str(core)
                    print(f"    Running PACO with {core} cores, num_ants={num_ants}, batch_size={batch_size}...")
                    result = run_paco_instance(
                        problem=problem,
                        batch_size=batch_size,
                        **paco_args
                    )
                    runtimes.append(result['runtime'])
                    print(f"    Cores={core}, num_ants={num_ants}, Time={result['runtime']:.2f}s")
                # Compute speedup and efficiency
                base_runtime = runtimes[0]
                for idx, core in enumerate(core_counts):
                    speedup = base_runtime / runtimes[idx] if runtimes[idx] > 0 else float('nan')
                    efficiency = speedup / core if core > 0 else float('nan')
                    speedups.append(speedup)
                    efficiencies.append(efficiency)
                    writer.writerow([
                        scale_name, instance_name, core, core * batch_size, batch_size, runtimes[idx], speedup, efficiency
                    ])
                    f.flush()
                # Plot and save line charts for this instance
                plt.figure()
                plt.plot(core_counts, speedups, marker='o')
                plt.xlabel('Core Count')
                plt.ylabel('Speedup S(p)')
                plt.title(f'Speedup vs Core Count ({scale_name}, {instance_name})')
                plt.grid(True)
                plt.savefig(os.path.join(output_dir, f'speedup_{scale_name}_{instance_name}.png'))
                plt.close()

                plt.figure()
                plt.plot(core_counts, efficiencies, marker='o')
                plt.xlabel('Core Count')
                plt.ylabel('Efficiency E(p)')
                plt.title(f'Efficiency vs Core Count ({scale_name}, {instance_name})')
                plt.grid(True)
                plt.savefig(os.path.join(output_dir, f'efficiency_{scale_name}_{instance_name}.png'))
                plt.close()
    print("\nPACO scaling vs core analysis complete. Line charts exported.")

def paco_sensitivity(config_path='sensitiviy.yaml'):
    """
    Runs PACO sensitivity analysis for parameters specified as lists in the config file.
    Results are saved to a CSV file in the output directory.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)['experiment']

    instance_path = config['instance']
    output_dir = config['output_dir']
    num_runs = config.get('num_runs', 1)
    num_core = config.get('num_core', 1) # Default to 1 core if not specified

    # Separate grid parameters (lists) from fixed parameters (single values)
    grid_params = {k: v for k, v in config.items() if isinstance(v, list)}
    fixed_params = {k: v for k, v in config.items() if not isinstance(v, list) and k not in ['instance', 'output_dir', 'num_runs', 'num_core']}

    # --- Default values ---
    # Set default alpha/beta if they are not grid parameters
    if 'alpha' not in grid_params and 'alpha' not in fixed_params:
        fixed_params['alpha'] = 1.0
    if 'beta' not in grid_params and 'beta' not in fixed_params:
        fixed_params['beta'] = 1.0
    # Ensure essential parameters have defaults if not provided
    if 'num_ants' not in grid_params and 'num_ants' not in fixed_params:
        fixed_params['num_ants'] = 100 # Example default
        print("Warning: 'num_ants' not found in config, using default:", fixed_params['num_ants'])
    if 'num_iterations' not in grid_params and 'num_iterations' not in fixed_params:
        fixed_params['num_iterations'] = 50 # Example default
        print("Warning: 'num_iterations' not found in config, using default:", fixed_params['num_iterations'])
    # Add defaults for other PACO parameters if necessary (e.g., Q, evaporation_rate, elitist_num)
    if 'Q' not in grid_params and 'Q' not in fixed_params:
        fixed_params['Q'] = 1.0
    if 'evaporation_rate' not in grid_params and 'evaporation_rate' not in fixed_params:
        fixed_params['evaporation_rate'] = 0.2
    if 'elitist_num' not in grid_params and 'elitist_num' not in fixed_params:
        fixed_params['elitist_num'] = 10

    # --- Setup Output ---
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    grid_param_names = sorted(grid_params.keys())
    csv_filename = f"sensitivity_{'_'.join(grid_param_names)}.csv" if grid_param_names else "sensitivity_run.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    header = grid_param_names + ['run', 'best_fitness', 'runtime', 'overhead']

    print(f"Starting PACO sensitivity analysis for parameters: {', '.join(grid_param_names)}")
    print(f"Fixed parameters: {fixed_params}")
    print(f"Results will be saved to: {csv_path}")

    # --- Load Problem ---
    problem = Problem()
    problem.load_data(instance_path)

    # --- Run Experiments ---
    # Generate all combinations of grid parameter values
    param_value_lists = [grid_params[k] for k in grid_param_names]
    param_combinations = list(itertools.product(*param_value_lists))

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for combo in param_combinations:
            current_run_params = dict(zip(grid_param_names, combo))
            # Combine fixed and current grid parameters
            paco_args = {**fixed_params, **current_run_params}

            # Determine num_ants and batch_size for this run
            num_ants = paco_args.get('num_ants')
            if num_ants is None:
                print("Error: 'num_ants' must be defined in fixed or grid parameters.")
                continue # Skip this combination if num_ants is missing

            batch_size = num_ants // num_core
            if batch_size == 0:
                print(f"Warning: Calculated batch_size is 0 (num_ants={num_ants}, num_core={num_core}). Setting batch_size=1.")
                batch_size = 1
            paco_args['batch_size'] = batch_size # Add batch_size to args passed to run_paco_instance

            # Prepare log message
            log_params = ', '.join(f"{k}={v}" for k, v in current_run_params.items())

            for run in range(num_runs):
                try:
                    res = run_paco_instance(
                        problem=problem,
                        **paco_args # Pass combined parameters
                    )
                    print(f"Run ({log_params}, run={run+1}) | Best Fitness: {res['best_fitness']:.2f} | Runtime: {res['runtime']:.2f}s")
                    # Prepare row data: grid param values + run + results
                    row_data = list(combo) + [run + 1, res['best_fitness'], res['runtime'], res['overhead']]
                    writer.writerow(row_data)
                    f.flush() # Ensure data is written periodically
                except Exception as e:
                    print(f"Error during run ({log_params}, run={run+1}): {e}")
                    # Optionally write an error row or skip
                    row_data = list(combo) + [run + 1, 'ERROR', 'ERROR', 'ERROR']
                    writer.writerow(row_data)
                    f.flush()

    print(f"PACO sensitivity analysis complete. Results saved to {csv_path}")

def run_complete(config_path='experiment_sizes.yaml'):
    """
    Runs PACO experiments for different instance sizes based on a YAML config.
    Automatically discovers instance files in the specified data directories.
    Calculates aggregated stats (Best Fitness, Avg Fitness, Std Fitness, Avg Runtime, Std Runtime)
    and reports the number of vehicles used in the best run for each instance.
    Saves aggregated results incrementally to separate CSV files for each size.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    for size, settings in config.items():
        print(f"--- Running Experiment for Size: {size.upper()} ---")
        data_dir = settings['data_dir']
        paco_params = settings['paco_params']
        num_runs = settings.get('num_runs', 1)
        output_csv = settings['output_csv']

        # Ensure output directory exists
        output_dir = os.path.dirname(output_csv)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Discover instance files
        try:
            instance_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
            if not instance_files:
                print(f"  Warning: No instance files (.txt) found in {data_dir}. Skipping size '{size}'.")
                continue
            print(f"  Found {len(instance_files)} instances in {data_dir}")
        except FileNotFoundError:
            print(f"  Error: Data directory not found: {data_dir}. Skipping size '{size}'.")
            continue
        # Sort instance files for consistent ordering
        instance_files.sort()
        # Define the header for the aggregated CSV
        header = ['instance_name', 'Num Vehicles', 'Best Distance', 'AVG Distance', 'Std Distance', 'AVG Runtime (s)', 'Std Runtime (s)']  # Add Num Vehicles
        # Write header only if file doesn't exist
        write_header = not os.path.exists(output_csv)

        with open(output_csv, 'a', newline='') as f:  # Open in append mode
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)

            for instance_name in instance_files:
                instance_path = os.path.join(data_dir, instance_name)
                print(f"  Processing Instance: {instance_name} ({num_runs} runs)")
                try:
                    problem = Problem()
                    problem.load_data(instance_path)
                except Exception as e:
                    print(f"    Error loading instance {instance_path}: {e}. Skipping.")
                    continue

                instance_run_results = []  # Collect results for this instance's runs
                for run in range(num_runs):
                    try:
                        # Use run_paco_instance which handles PACO setup and execution
                        res = run_paco_instance(
                            problem=problem,
                            **paco_params  # Pass all PACO parameters from YAML
                        )
                        instance_run_results.append(res)
                    except Exception as e:
                        print(f"      Error during PACO run for {instance_name}, run {run + 1}: {e}")
                        instance_run_results.append({
                            'best_fitness': float('nan'),
                            'runtime': float('nan'),
                            'num_vehicles': float('nan')  # Add NaN for vehicles on error
                        })

                # Aggregate results for the current instance
                if instance_run_results:
                    valid_runs = [r for r in instance_run_results if not math.isnan(r['best_fitness'])]

                    if valid_runs:
                        fitnesses = [r['best_fitness'] for r in valid_runs]
                        runtimes = [r['runtime'] for r in valid_runs]
                        vehicles = [r['num_vehicles'] for r in valid_runs]

                        best_run_index = fitnesses.index(min(fitnesses))
                        best_dist = fitnesses[best_run_index]
                        num_vehicles_best_run = vehicles[best_run_index]

                        avg_dist = mean(fitnesses)
                        std_dist = stdev(fitnesses) if len(fitnesses) > 1 else 0.0

                        avg_runtime = mean(runtimes)
                        std_runtime = stdev(runtimes) if len(runtimes) > 1 else 0.0

                    else:  # Handle case where all runs failed
                        best_dist, avg_dist, std_dist = float('nan'), float('nan'), float('nan')
                        avg_runtime, std_runtime = float('nan'), float('nan')
                        num_vehicles_best_run = 'NaN'

                    # Write the aggregated row for this instance
                    writer.writerow([
                        instance_name,
                        num_vehicles_best_run,  # Add number of vehicles from the best run
                        f"{best_dist:.2f}" if not math.isnan(best_dist) else 'NaN',
                        f"{avg_dist:.2f}" if not math.isnan(avg_dist) else 'NaN',
                        f"{std_dist:.2f}" if not math.isnan(std_dist) else 'NaN',
                        f"{avg_runtime:.2f}" if not math.isnan(avg_runtime) else 'NaN',
                        f"{std_runtime:.2f}" if not math.isnan(std_runtime) else 'NaN',
                    ])
                    f.flush()  # Ensure data is written to disk immediately
                    print(f"    -> Aggregated results written for {instance_name}")
                else:
                    print(f"    -> No valid results to aggregate for {instance_name}")

        print(f"--- Experiment Complete for Size: {size.upper()} ---")
        print(f"  Results saved to: {output_csv}")

    print("--- All Experiments Complete ---")

def main_wizard():
    print("\n==== PACO Experiment Wizard ====")
    print("Select an experiment to run:")
    print("1. PACO Sensitivity Analysis")
    print("2. PACO Speedup vs Cores")
    print("3. PACO Complete Run (by size)")
    print("0. Exit")
    choice = input("Enter your choice [1-3, 0 to exit]: ").strip()
    if choice == '1':
        default_cfg = 'sensitiviy.yaml'
        cfg = input(f"Enter config file for sensitivity analysis [default: {default_cfg}]: ").strip() or default_cfg
        paco_sensitivity(cfg)
    elif choice == '2':
        default_cfg = 'speedup.yaml'
        cfg = input(f"Enter config file for speedup experiment [default: {default_cfg}]: ").strip() or default_cfg
        speed_up(cfg)
    elif choice == '3':
        default_cfg = 'sizes.yaml'
        cfg = input(f"Enter config file for complete run [default: {default_cfg}]: ").strip() or default_cfg
        run_complete(cfg)
    elif choice == '0':
        print("Exiting wizard.")
        return
    else:
        print("Invalid choice. Please try again.")
        main_wizard()

if __name__ == '__main__':
    main_wizard()
