import random
import math
import time
import matplotlib.pyplot as plt
import numpy as np
from problems import rastrigin

class Bee:
    def __init__(self, position):
        self.position = position
        self.fitness = 0

    def evaluate(self, objective_function):
        self.fitness = objective_function(self.position)

class BeeColonyOptimization:
    def __init__(self, objective_function, num_bees, num_iterations, search_space, max_trials):
        self.objective_function = objective_function
        self.num_bees = num_bees
        self.num_iterations = num_iterations
        self.search_space = search_space
        self.max_trials = max_trials
        self.best_solution = None
        self.best_fitness = float('inf')
        self.bees = [Bee(self._initialize_bee()) for _ in range(num_bees)]
        self.fitness_history = []

    def _initialize_bee(self):
        return [random.uniform(self.search_space[i][0], self.search_space[i][1]) for i in range(len(self.search_space))]

    def _mutate(self, position):
        return [position[i] + random.uniform(-1, 1) * (self.search_space[i][1] - self.search_space[i][0]) for i in range(len(position))]

    def optimize(self):
        for iteration in range(self.num_iterations):
            for bee in self.bees:
                bee.evaluate(self.objective_function)
                if bee.fitness < self.best_fitness:
                    self.best_fitness = bee.fitness
                    self.best_solution = bee.position

            for bee in self.bees:
                trials = 0
                while trials < self.max_trials:
                    new_position = self._mutate(bee.position)
                    new_bee = Bee(new_position)
                    new_bee.evaluate(self.objective_function)
                    if new_bee.fitness < bee.fitness:
                        bee.position = new_bee.position
                        bee.fitness = new_bee.fitness
                        break
                    trials += 1

            self.fitness_history.append(self.best_fitness)
            print(f"Iteration {iteration + 1}: Best Fitness = {self.best_fitness}")

        return self.best_solution, self.best_fitness

    def plot_search_space(self):
        if len(self.search_space) != 2:
            print("Search space must be 2D for visualization.")
            return

        x = np.linspace(self.search_space[0][0], self.search_space[0][1], 100)
        y = np.linspace(self.search_space[1][0], self.search_space[1][1], 100)
        X, Y = np.meshgrid(x, y)
        Z = np.array([[self.objective_function([xi, yi]) for xi, yi in zip(x_row, y_row)] for x_row, y_row in zip(X, Y)])

        plt.figure(figsize=(10, 6))
        plt.contourf(X, Y, Z, levels=50, cmap='viridis')
        plt.colorbar(label='Fitness')
        plt.scatter(self.best_solution[0], self.best_solution[1], color='red', label='Best Solution')
        plt.title('Search Space and Best Solution')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()

    def plot_fitness_history(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.num_iterations + 1), self.fitness_history, marker='o', linestyle='-', color='b')
        plt.title('Fitness History')
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness')
        plt.grid(True)
        plt.show()

# Parameters
num_bees = 20
num_iterations = 100
search_space = [(-5.12, 5.12), (-5.12, 5.12)]  # 2-dimensional search space for visualization
max_trials = 10

start_time = time.time()
# Run BCO
bco = BeeColonyOptimization(rastrigin, num_bees, num_iterations, search_space, max_trials)
best_solution, best_fitness = bco.optimize()
end_time = time.time()

print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)
print(f"Time taken: {end_time - start_time:.6f} seconds")

# Plot the search space and best solution
bco.plot_search_space()

# Plot the fitness history
bco.plot_fitness_history()