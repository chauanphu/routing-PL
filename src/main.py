import time
from meta.solver import Problem
from meta.GreyWolf import GreyWolfOptimization as GWO
from meta.PSO import ParticleSwarmOptimization as PSO
import numpy as np
from scipy.stats import ttest_ind

# Create a problem instance
instance = Problem()
instance.load_data("data/25/C101_co_25.txt")

start_time = time.time()
# Load the solver
gwo = GWO(instance.evaluate, num_wolves=10, num_iterations=100, search_space=instance.get_search_space())
pso = PSO(instance.evaluate, num_particles=10, num_iterations=100, search_space=instance.get_search_space())
solver = [gwo, pso]

# Initialize lists to store results
gwo_results = []
pso_results = []

# Repeat the experiment 50 times
for _ in range(50):
    print(f"Experiment {_ + 1}")
    for s in solver:
        start = time.time()
        s.optimize(verbose=False)
        end = time.time()
        if isinstance(s, GWO):
            gwo_results.append((s.global_best_fitness, end - start))
        elif isinstance(s, PSO):
            pso_results.append((s.global_best_fitness, end - start))

# Calculate mean and standard deviation
gwo_fitness = [result[0] for result in gwo_results]
gwo_time = [result[1] for result in gwo_results]
pso_fitness = [result[0] for result in pso_results]
pso_time = [result[1] for result in pso_results]

print("Number of experiments: 50")
print("GWO Results:")
print(f"Mean Fitness: {np.mean(gwo_fitness)}")
print(f"Std Fitness: {np.std(gwo_fitness)}")
print(f"Mean Time: {np.mean(gwo_time)}")
print(f"Std Time: {np.std(gwo_time)}")

print("PSO Results:")
print(f"Mean Fitness: {np.mean(pso_fitness)}")
print(f"Std Fitness: {np.std(pso_fitness)}")
print(f"Mean Time: {np.mean(pso_time)}")
print(f"Std Time: {np.std(pso_time)}")

# Perform hypothesis testing
t_stat_fitness, p_value_fitness = ttest_ind(gwo_fitness, pso_fitness)
t_stat_time, p_value_time = ttest_ind(gwo_time, pso_time)

print("Hypothesis Testing Results:")
print("Fitness Comparison:")
print(f"T-statistic: {t_stat_fitness:.2f}, P-value: {p_value_fitness:.2f}")
if p_value_fitness < 0.05:
    print("Reject the null hypothesis: There is a significant difference in fitness between GWO and PSO.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference in fitness between GWO and PSO.")

print("Time Comparison:")
print(f"T-statistic: {t_stat_time:.2f}, P-value: {p_value_time:.2f}")
if p_value_time < 0.05:
    print("Reject the null hypothesis: There is a significant difference in time between GWO and PSO.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference in time between GWO and PSO.")
