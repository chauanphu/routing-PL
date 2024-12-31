import random
from typing import List, Tuple
import numpy as np
from meta.PSO.Particle import PSOParticle
from utils.config import GA_NUMMAX, GA_NUMMIN, MAX_ITER, GAMMA, GA_MINITER, BETA, GA_PSMIN, GA_PSMAX, GA_MAXITR, GA_MODE

from utils.load_data import OrderItem, Vehicle

class GA:
    def __init__(self, PSO_ITER: int, population: list[PSOParticle], num_vehicle: int, num_orders: int):
        self.population: List[PSOParticle] = population
        self.GA_NUM = int(GA_NUMMAX -((PSO_ITER / MAX_ITER) ** GAMMA) * (GA_NUMMAX - GA_NUMMIN)) # Number of individuals to select from PSO
        self.GA_PS = int(GA_PSMIN + ((PSO_ITER / MAX_ITER) ** GAMMA) * (GA_PSMAX - GA_PSMIN)) # Population size of GA
        self.GA_MAXITER = int(GA_MINITER + ((PSO_ITER / MAX_ITER) ** BETA) * (GA_MAXITR - GA_MINITER)) # Number of iterations of GA
        self.num_vehicle = num_vehicle
        self.num_orders = num_orders
        self.ELITIST_SIZE = 10

    def evolve(self, orders: List[OrderItem], vehicles: List[Vehicle]):
        sub_idxes = self.select_subpopulation()
        selected_population = [self.population[i] for i in sub_idxes]
        for ind_idx, individual in zip(sub_idxes, selected_population):
            current_best_fitness = individual.p_fitness
            current_best_position = individual.positions
            new_best_particle = None
            for _ in range(self.GA_MAXITER):
                # Keep the best individual
                chromosomes = self.initialize(individual=current_best_position)
                sorted_elites, sorted_fitness, idxes = self.select_elites(chromosomes, orders, vehicles)
                elites = sorted_elites[:self.ELITIST_SIZE]
                # Apply selection, crossover, and mutation
                parent1, parent2 = self.select(elites, sorted_fitness[:self.ELITIST_SIZE])
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                # Replace the worst individual with the new individual if the new individual is better
                new_child1 = PSOParticle(orders, vehicles, child1)
                new_child1.decode()
                new_child1.update_fitness()
                new_child2 = PSOParticle(orders, vehicles, child2)
                new_child2.decode()
                new_child2.update_fitness()

                if new_child1.p_fitness < new_child2.p_fitness:
                    new_child1 = new_child2
                if new_child1.p_fitness < current_best_fitness:
                    current_best_fitness = new_child1.p_fitness
                    current_best_position = new_child1.positions
                    new_best_particle = new_child1
            
            # Compare the new individual with the old individual
            if new_best_particle is not None:
                print("Updated", ind_idx, end=' ')
                self.population[ind_idx] = new_best_particle

        return self.population
    
    def initialize(self, individual: np.ndarray) -> np.ndarray:
        """
        Randomly initialize vector with GA_PS particle
        """
        chromosomes = [PSOParticle.random_position(self.num_orders, self.num_vehicle) for _ in range(self.GA_PS)]
        chromosomes[0] = individual.copy()
        return chromosomes

    def select_subpopulation(self) -> list[int]:
        """
        Select a subpopulation from the population
        
        """
        # Return the best individuals
        return sorted(
            range(len(self.population)), 
            key=lambda i: self.population[i].p_fitness,
            reverse=GA_MODE == 'worst_selection' # If the reverse is True, sort in descending order (worst selection)
        )[:self.GA_NUM]

    def select(self, chromosomes: np.ndarray, fitness: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select two parents from the population using roulette wheel selection
        
        """
        # Normalize the fitness
        fitness = np.array(fitness)
        fitness = fitness - np.min(fitness)
        chromosomes = np.array(chromosomes)
        if np.sum(fitness) == 0:
            return chromosomes[0], chromosomes[1]
        fitness = fitness / np.sum(fitness)
        # Cumulative sum of the fitness
        cumsum = np.cumsum(fitness)
        selected = set()
        while len(selected) < 2:
            r = random.random()
            for i, f in enumerate(cumsum):
                if r < f:
                    selected.add(i)
                    break
        selected = list(selected)
        return chromosomes[selected[0]], chromosomes[selected[1]]

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray, crossover_rate=0.8) -> Tuple[np.ndarray, np.ndarray]:
        """
        2-point crossover two parents, each parent is a vector of length 2 * num_orders
        
        The first half is segment is vehicle assignment, the second half is the priority
        """
        # Select two random points
        parent1 = parent1.copy()
        parent2 = parent2.copy()
        if random.random() > crossover_rate:
            return parent1, parent2
        half_point = len(parent1) // 2
        assignment_point = random.randint(0, half_point)
        priority_point = random.randint(half_point, len(parent1) - 1)
        # Perform crossover seperately for vehicle assignment and priority
        new_assignment = np.concatenate([parent1[:assignment_point], parent2[assignment_point:half_point]])
        new_priority = np.concatenate([parent1[half_point:priority_point], parent2[priority_point:]])
        child1 = np.concatenate([new_assignment, new_priority])
        new_assignment = np.concatenate([parent2[:assignment_point], parent1[assignment_point:half_point]])
        new_priority = np.concatenate([parent2[half_point:priority_point], parent1[priority_point:]])
        child2 = np.concatenate([new_assignment, new_priority])
        return child1, child2
    
    def mutate(self, chromosome: np.ndarray, mutation_rate=0.1) -> np.ndarray:
        """
        Mutate the chromosome, swapping vehicle assignment and priority
        
        """
        if random.random() > mutation_rate:
            return chromosome
        chromosome = chromosome.copy()
        half_point = len(chromosome) // 2 - 1
        first_segment = chromosome[:half_point]
        second_segment = chromosome[half_point:]
        # Swap 2 random points in vehicle assignment segment the swap 2 random points in priority segment
        v1_idx, v2_idx = random.sample(range(len(first_segment)), 2)
        p1_idx, p2_idx = random.sample(range(len(second_segment)), 2)
        first_segment[v1_idx], first_segment[v2_idx] = first_segment[v2_idx], first_segment[v1_idx]
        second_segment[p1_idx], second_segment[p2_idx] = second_segment[p2_idx], second_segment[p1_idx]

        chromosome = np.concatenate([first_segment, second_segment])

        return chromosome
    
    def select_elites(self, positions: np.ndarray, orders: List[OrderItem], vehicles: List[Vehicle]):
        """
        Select the best individual from the population
        
        """
        fitness = []
        for position in positions:
            individual = PSOParticle(orders, vehicles, position)
            individual.decode()
            individual.update_fitness()
            fitness.append(individual.p_fitness)
        # Select k best individuals
        idxes = sorted(range(len(fitness)), key=lambda i: fitness[i])
        return [positions[i] for i in idxes], [fitness[i] for i in idxes], idxes