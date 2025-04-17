import csv
import itertools
import os
from pathlib import Path
import time
from statistics import mean, stdev
import yaml

from meta.solver import Problem
from meta.ACO import PACO
import matplotlib.pyplot as plt

def run_paco_instance(problem, num_ants, batch_size, **other_params):
    """
    Runs a single PACO instance.
    Each run is executed sequentially (not in parallel) to keep the cores free for the parallel PACO internal operations.
    It creates a fresh PACO instance, runs optimize(), measures runtime,
    and returns a dictionary of metrics.
    """

    paco_instance = PACO(problem,
                         num_ants=num_ants,
                         batch_size=batch_size,
                         **other_params)
    
    start_time = time.time()
    result = paco_instance.optimize(verbose=True)
    end_time = time.time()
    total_runtime = end_time - start_time

    try:
        overhead = paco_instance.overhead  # Assume PACO stores an overall overhead metric
    except AttributeError:
        overhead = None

    return {
        'num_ants': num_ants,
        'batch_size': batch_size,
        'best_fitness': result[1],  # (solution, fitness, routes)
        'runtime': total_runtime,
        'overhead': overhead
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

def run_sensitivity_analysis(instances, param_grid, base_params, num_runs, output_dir):
    """
    Runs sensitivity analysis for PACO across specified instances and parameter variations.

    Args:
        instances (dict): Dictionary mapping instance scale/type to list of filenames.
                          e.g., {'small_C': ['C101_co_25.txt'], 'large_R': ['R101_co_100.txt']}
        param_grid (dict): Dictionary where keys are parameter names to vary ('num_ants', 'alpha', etc.)
                           and values are lists of values to test for that parameter.
        base_params (dict): Dictionary of fixed PACO parameters.
        num_runs (int): Number of times to run each configuration.
        output_dir (str): Directory to save the results CSV.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results_file = os.path.join(output_dir, 'paco_sensitivity_results.csv')
    
    # Prepare header for CSV
    param_names = list(param_grid.keys())
    header = ['instance_scale', 'instance_type', 'instance_name'] + param_names + ['run', 'best_fitness', 'runtime', 'overhead']
    
    # Generate all parameter combinations to test
    varying_param_values = [param_grid[k] for k in param_names]
    param_combinations = list(itertools.product(*varying_param_values))

    print(f"Starting PACO Sensitivity Analysis. Results will be saved to: {results_file}")
    print(f"Base parameters: {base_params}")
    print(f"Varying parameters: {param_grid}")
    print(f"Number of runs per config: {num_runs}")

    with open(results_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for scale_type, instance_files in instances.items():
            scale, type_ = scale_type.split('_') # e.g., 'small_C' -> 'small', 'C'
            for instance_name in instance_files:
                instance_path = f"data/{scale}/{instance_name}"
                print(f"\n--- Loading Instance: {instance_path} ---")
                try:
                    problem = Problem()
                    problem.load_data(instance_path)
                except FileNotFoundError:
                    print(f"Error: Instance file not found at {instance_path}. Skipping.")
                    continue
                
                print(f"--- Running Experiments for {instance_name} ---")
                for combo in param_combinations:
                    current_params = base_params.copy()
                    param_combo_dict = {}
                    for i, param_name in enumerate(param_names):
                        current_params[param_name] = combo[i]
                        param_combo_dict[param_name] = combo[i]
                    
                    print(f"  Config: {param_combo_dict}")
                    
                    for run in range(num_runs):
                        print(f"    Run {run + 1}/{num_runs}...", end="")
                        try:
                            # Ensure all necessary params are passed
                            run_result = run_paco_instance(
                                problem=problem,
                                num_ants=current_params.get('num_ants', 1000), # Provide defaults if not varied
                                batch_size=current_params.get('batch_size', 100),
                                num_iterations=current_params.get('num_iterations', 100),
                                alpha=current_params.get('alpha', 1.0),
                                beta=current_params.get('beta', 2.0),
                                evaporation_rate=current_params.get('evaporation_rate', 0.1),
                                Q=current_params.get('Q', 1.0),
                                elitist_num=current_params.get('elitist_num', 10)
                            )
                            
                            # Prepare row for CSV
                            row_data = [scale, type_, instance_name] + list(combo) + \
                                       [run + 1, run_result['best_fitness'], run_result['runtime'], run_result['overhead']]
                            writer.writerow(row_data)
                            f.flush() # Write to file periodically
                            print(f" Fitness: {run_result['best_fitness']:.2f}, Time: {run_result['runtime']:.2f}s")
                        
                        except Exception as e:
                            print(f" Error during run: {e}")
                            # Optionally write an error row
                            error_row = [scale, type_, instance_name] + list(combo) + [run + 1, 'ERROR', str(e), None]
                            writer.writerow(error_row)
                            f.flush()

    print("\nSensitivity Analysis Complete.")

def speed_up(config_path='param.yaml'):
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
    paco_params = exp_config['paco_params']
    samples = exp_config['samples']
    num_runs = exp_config.get('num_runs', 1)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results_file = os.path.join(output_dir, 'paco_scaling_vs_cores.csv')
    header = ['instance_scale', 'instance_name', 'core_count', 'num_ants', 'batch_size', 'runtime', 'speedup', 'efficiency']

    print(f"Starting PACO scaling vs core analysis. Results will be saved to: {results_file}")
    with open(results_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for scale_name, scale_config in samples.items():
            instance_files = scale_config['instances']
            data_dir = scale_config['data_dir']
            for instance_name in instance_files:
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

def paco_sensitivity():
    # 1. Define Instances to Test
    instances_to_test = {
        # Small Scale (25 nodes)
        '25_C': ['C101_co_25.txt', 'C201_co_25.txt'],
        '25_R': ['R101_co_25.txt', 'R201_co_25.txt'],
        '25_RC': ['RC101_co_25.txt', 'RC201_co_25.txt'],
        # Medium Scale (50 nodes) - Sampled
        '50_C': ['C101_co_50.txt'],
        '50_R': ['R101_co_50.txt'],
        '50_RC': ['RC101_co_50.txt'],
        # Large Scale (100 nodes) - Sampled
        '100_C': ['C101_co_100.txt'],
        '100_R': ['R101_co_100.txt'],
        '100_RC': ['RC101_co_100.txt'],
    }

    # 2. Define Parameter Grid for Sensitivity Analysis
    parameter_grid = {
        'num_ants': [200, 500, 2000],          # Vary number of ants
        'evaporation_rate': [0.1, 0.2, 0.5],    # Vary evaporation rate
        'alpha': [0.5, 1.0, 2.0],               # Vary alpha
        'beta': [0.5, 1.0, 2.0],                 # Vary beta
        'batch_size': [10, 20, 50]      # Fixed batch size
    }

    # 3. Define Base (Fixed) Parameters for PACO
    base_parameters = {
        'num_iterations': 100,  # Fixed number of iterations
        'Q': 1.0,               # Fixed Q
        'elitist_num': 10       # Fixed number of elitist ants
        # Note: The parameters defined in parameter_grid will override these during the loops
    }

    # 4. Set Number of Runs for Averaging
    number_of_runs = 5 # Run each configuration 5 times

    # 5. Define Output Directory
    output_directory = "output/sensitivity_analysis"

    # 6. Run the Analysis
    run_sensitivity_analysis(
        instances=instances_to_test,
        param_grid=parameter_grid,
        base_params=base_parameters,
        num_runs=number_of_runs,
        output_dir=output_directory
    )

class ParallelExperiment:
    def __init__(self, problem, other_params=None, num_runs=3):
        """
        :param problem: The Problem instance for PACO.
        :param other_params: Additional PACO parameters (e.g., alpha, beta, evaporation_rate, Q, etc.)
        :param num_runs: Number of independent runs per configuration.
        """
        self.problem = problem
        self.other_params = other_params if other_params is not None else {}
        self.num_runs = num_runs
        self.results = []  # List to store results from each run

    def run_experiment_varying_ants(self, fixed_batch_size, ants_values):
        """
        Runs experiments varying the number of ants (with fixed batch size) sequentially.
        :param fixed_batch_size: Constant batch size used for each run.
        :param ants_values: List of num_ants values to test.
        :return: List of result dictionaries.
        """
        print("Starting experiment: Varying number of ants (fixed batch size = {})".format(fixed_batch_size))
        experiment_results = []
        for num_ants in ants_values:
            for run in range(self.num_runs):
                res = run_paco_instance(self.problem, num_ants, fixed_batch_size, self.other_params)
                experiment_results.append(res)
                print("Run (ants={}, batch_size={}, run={}) | Best Fitness: {} | Runtime: {:.2f}s".format(
                    num_ants, fixed_batch_size, run+1, res['best_fitness'], res['runtime']))
                # Write to files for checkpointing: 
        self.results = experiment_results
        return experiment_results

    def run_experiment_varying_batch_size(self, fixed_num_ants, batch_sizes):
        """
        Runs experiments varying the batch size (with fixed number of ants) sequentially.

        :param fixed_num_ants: Constant number of ants for each run.
        :param batch_sizes: List of batch size values to test.
        :return: List of result dictionaries.
        """
        print("Starting experiment: Varying batch size (fixed num_ants = {})".format(fixed_num_ants))
        experiment_results = []
        for batch_size in batch_sizes:
            for run in range(self.num_runs):
                res = run_paco_instance(self.problem, fixed_num_ants, batch_size, self.other_params)
                experiment_results.append(res)
                print("Run (ants={}, batch_size={}, run={}) | Best Fitness: {} | Runtime: {:.2f}s".format(
                    fixed_num_ants, batch_size, run+1, res['best_fitness'], res['runtime']))
        self.results = experiment_results
        return experiment_results

    def write_csv_report(self, filename):
        """
        Writes raw experimental results to a CSV file.

        :param filename: CSV filename.
        """
        if not self.results:
            print("No results available to write.")
            return
        header = list(self.results[0].keys())
        with open(filename, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for row in self.results:
                writer.writerow(row)
        print("CSV report written to:", filename)

    def aggregate_results(self, param_name):
        """
        Aggregates results based on a given parameter (either 'num_ants' or 'batch_size').

        :param param_name: Parameter name to aggregate by.
        :return: Dictionary mapping parameter value to aggregated metrics.
        """
        agg = {}
        for res in self.results:
            key = res[param_name]
            if key not in agg:
                agg[key] = {'best_fitness': [], 'runtime': [], 'overhead': []}
            agg[key]['best_fitness'].append(res['best_fitness'])
            agg[key]['runtime'].append(res['runtime'])
            if res['overhead'] is not None:
                agg[key]['overhead'].append(res['overhead'])
        summary = {}
        for key, metrics in agg.items():
            summary[key] = {
                'min_best_fitness': min(metrics['best_fitness']),
                'avg_best_fitness': mean(metrics['best_fitness']),
                'std_best_fitness': stdev(metrics['best_fitness']) if len(metrics['best_fitness']) > 1 else 0,
                'avg_runtime': mean(metrics['runtime']),
                'std_runtime': stdev(metrics['runtime']) if len(metrics['runtime']) > 1 else 0,
                'avg_overhead': mean(metrics['overhead']) if metrics['overhead'] else None,
                'std_overhead': stdev(metrics['overhead']) if len(metrics['overhead']) > 1 else None
            }
        return summary

    def write_aggregated_report(self, summary, filename, param_name):
        """
        Writes aggregated experimental data into a CSV file.

        :param summary: Dictionary from aggregate_results().
        :param filename: CSV filename.
        :param param_name: The parameter used for grouping (e.g., 'num_ants' or 'batch_size').
        """
        header = [param_name, 'avg_best_fitness', 'std_best_fitness',
                  'min_best_fitness', 'avg_runtime', 'std_runtime', 'avg_overhead', 'std_overhead']
        with open(filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for key, metrics in sorted(summary.items()):
                writer.writerow([key,
                                 metrics['min_best_fitness'],
                                 metrics['avg_best_fitness'],
                                 metrics['std_best_fitness'],
                                 metrics['avg_runtime'],
                                 metrics['std_runtime'],
                                 metrics['avg_overhead'],
                                 metrics['std_overhead']])
        print("Aggregated report written to:", filename)
          
# Example usage:
if __name__ == '__main__':
    speed_up()
