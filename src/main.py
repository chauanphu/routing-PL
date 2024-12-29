from utils.load_data import load_voratas_vrp
import logging
from matplotlib import pyplot as plt
from meta.PSO.solver import PSOSolver
from utils.config import POPULATION_SIZE, MAX_ITER, INFEASIBILITY_PENALTY
logging.basicConfig(level=logging.INFO)

def pso_main():
    orders, locations, vehicles  = load_voratas_vrp('Aar1')
    print("Number of orders:", len(orders))
    print("Number of locations:", len(locations))
    print("Number of vehicles:", len(vehicles))
    # visualize_orders(orders, vehicles=vehicles, title="Aac2", output_name="Aac2")
    psoSolver = PSOSolver(n_particles=POPULATION_SIZE, n_iterations=MAX_ITER, orders=orders, vehicles=vehicles)
    psoSolver.init_swarm()
    history = psoSolver.solve()
    # Plot the line of the best fitness and the personal best fitness
    plt.plot(history['fitness'], label='Global Best')   
    # Plot the horizontal line of the infeasibility penalty
    plt.axhline(y=INFEASIBILITY_PENALTY, color='r', linestyle='-', label='Infeasibility Penalty')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.title('Fitness of the best particle')
    plt.legend()
    plt.show()
    plt.savefig('output/pso_fitness.png')

    # Logging the change of the first particle fitness into a file.txt
    with open('output/pso_particle_history.txt', 'w') as f:
        for i in range(len(history['particle_fitness'][1])):
            f.write(f"{i+1}: {history['particle_fitness'][1][i]}\n")
    psoSolver.print_best_solution()
    print("Done")

if __name__ == "__main__":
    pso_main()
