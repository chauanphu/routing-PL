import random

from meta.solver import Solver

class Particle:
    def __init__(self, position, velocity):
        self.position = position                # Current position
        self.velocity = velocity                # Current velocity
        self.best_position = position.copy()    # Personal best position so far
        self.best_fitness = float('inf')        # Personal best fitness

    def evaluate(self, objective_function):
        """
        Evaluate the particle's current fitness and update its personal best.
        """
        fitness = objective_function(self.position)
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_position = self.position.copy()
        return fitness

class ParticleSwarmOptimization(Solver):
    def __init__(self, objective_function, num_particles, num_iterations, search_space,
                 w=0.5, c1=1.5, c2=1.5):
        """
        Initializes the PSO instance.
        
        Parameters:
            objective_function (callable): The function to be minimized.
            num_particles (int): The number of particles in the swarm.
            num_iterations (int): The number of iterations to run the optimization.
            search_space (list of tuples): The bounds for each dimension.
            w (float): Inertia weight.
            c1 (float): Cognitive (personal) coefficient.
            c2 (float): Social (global) coefficient.
        """
        super().__init__(objective_function, num_iterations=num_iterations, search_space=search_space)
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.swarm = [self._initialize_particle() for _ in range(num_particles)]

    def _initialize_particle(self):
        """
        Initializes a particle with random position and velocity.
        """
        position = [random.uniform(self.search_space[i][0], self.search_space[i][1])
                    for i in range(len(self.search_space))]
        # Velocity can be initialized in the range of the search space dimensions
        velocity = [random.uniform(-abs(self.search_space[i][1] - self.search_space[i][0]),
                                   abs(self.search_space[i][1] - self.search_space[i][0]))
                    for i in range(len(self.search_space))]
        return Particle(position, velocity)

    def optimize(self, verbose=True):
        """
        Runs the PSO algorithm and returns the best solution and its fitness.
        """
        self.fitness_history = []
        for iteration in range(self.num_iterations):
            for particle in self.swarm:
                # Evaluate fitness and update personal best
                fitness = particle.evaluate(self.objective_function)
                # Update global best if needed
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position.copy()

            # Update velocity and position for each particle
            for particle in self.swarm:
                new_velocity = []
                new_position = []
                for i in range(len(self.search_space)):
                    r1 = random.random()
                    r2 = random.random()
                    # Velocity update rule:
                    # v = w*v + c1*r1*(personal_best - current_position) + c2*r2*(global_best - current_position)
                    inertia = self.w * particle.velocity[i]
                    cognitive = self.c1 * r1 * (particle.best_position[i] - particle.position[i])
                    social = self.c2 * r2 * (self.global_best_position[i] - particle.position[i])
                    v_new = inertia + cognitive + social
                    new_velocity.append(v_new)

                    # Update the position
                    pos_new = particle.position[i] + v_new
                    # Ensure the new position is within the search space bounds
                    lower_bound, upper_bound = self.search_space[i]
                    pos_new = max(lower_bound, min(pos_new, upper_bound))
                    new_position.append(pos_new)
                particle.velocity = new_velocity
                particle.position = new_position

            self.fitness_history.append(self.global_best_fitness)
            if verbose:
                print(f"Iteration {iteration + 1}: Global Best Fitness = {self.global_best_fitness}")
        return self.global_best_position, self.global_best_fitness