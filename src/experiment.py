import csv
import time
from statistics import mean, stdev

from meta.solver import Problem
from meta.ACO import PACO

def run_paco_instance(problem, num_ants, batch_size, other_params):
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
    result = paco_instance.optimize(verbose=False)
    end_time = time.time()
    total_runtime = end_time - start_time

    try:
        overhead = paco_instance.overhead  # Assume PACO stores an overall overhead metric
    except AttributeError:
        overhead = None

    # paco_instance.cleanup()

    return {
        'num_ants': num_ants,
        'batch_size': batch_size,
        'best_fitness': result[1],  # (solution, fitness, routes)
        'runtime': total_runtime,
        'overhead': overhead
    }


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
                  'avg_runtime', 'std_runtime', 'avg_overhead', 'std_overhead']
        with open(filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for key, metrics in sorted(summary.items()):
                writer.writerow([key,
                                 metrics['avg_best_fitness'],
                                 metrics['std_best_fitness'],
                                 metrics['avg_runtime'],
                                 metrics['std_runtime'],
                                 metrics['avg_overhead'],
                                 metrics['std_overhead']])
        print("Aggregated report written to:", filename)


# Example usage:
if __name__ == '__main__':
    # from problem_module import Problem
    # Initialize your problem instance appropriately.
    instance = Problem()
    instance.load_data("data/50/C101_co_50.txt")

    extra_params = {'alpha': 1.0, 'beta': 1.0, 'evaporation_rate': 0.1, 'Q': 1.0, 'num_iterations': 100}

    experiment = ParallelExperiment(problem=instance,
                                    other_params=extra_params,
                                    num_runs=10)

    # Experiment 1: Vary number of ants (with fixed batch size)
    ants_values = [100, 200, 500, 1000]
    fixed_batch_size = 100
    results_ants = experiment.run_experiment_varying_ants(fixed_batch_size=fixed_batch_size,
                                                           ants_values=ants_values)
    experiment.write_csv_report("test_1.csv")
    summary_ants = experiment.aggregate_results(param_name='num_ants')
    experiment.write_aggregated_report(summary_ants, "test_1_summary.csv", param_name='num_ants')

    # Experiment 2: Vary batch size (with fixed number of ants)
    batch_sizes = [50, 100, 200, 400]
    fixed_num_ants = 1000
    results_batch = experiment.run_experiment_varying_batch_size(fixed_num_ants=fixed_num_ants,
                                                                 batch_sizes=batch_sizes)
    experiment.write_csv_report("test_2.csv")
    summary_batch = experiment.aggregate_results(param_name='batch_size')
    experiment.write_aggregated_report(summary_batch, "test_2_summary.csv", param_name='batch_size')
