from utils.load_data import load_voratas_vrp
import logging
from matplotlib import pyplot as plt
from meta.PSO.solver import PSOSolver
from utils.config import POPULATION_SIZE, MAX_ITER, INFEASIBILITY_PENALTY, INSTANCE, GA_ENABLED, GA_MODE
logging.basicConfig(level=logging.INFO)

def pso_main():
    orders, locations, vehicles  = load_voratas_vrp(INSTANCE)
    print("Number of orders:", len(orders))
    print("Number of locations:", len(locations))
    print("Number of vehicles:", len(vehicles))
    print("Instance:", INSTANCE)
    print("GA Enabled:", GA_ENABLED)
    print("GA Mode:", 'Mode A' if GA_MODE == 'best_selection' else 'Mode B')
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
    title = f'Fitness of {INSTANCE} using PSO{("-GA " + ("MODE A" if GA_MODE == "best_selection" else "MODE B")) if GA_ENABLED else ""}'
    file_name = f'{INSTANCE}{("-GA" + ("-A" if GA_MODE == "best_selection" else "-B")) if GA_ENABLED else ""}'
    plt.title(title)
    plt.legend()
    plt.show()
    plt.savefig(f'output/{file_name}.plot.png')

    # Save the history to a file
    with open(f'output/{file_name}.history.txt', 'w') as f:
        f.write(str(history['fitness']).replace("[", "").replace("]", ""))
    # Print the best solution
    psoSolver.print_best_solution(f'output/{file_name}.solution.txt')
    print("Done")

if __name__ == "__main__":
    pso_main()
