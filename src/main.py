from utils.load_data import load_voratas_vrp
import logging
from matplotlib import pyplot as plt
from utils.solver import PSOSolver
from utils.config import POPULATION_SIZE, MAX_ITER, ALLOW_EARLY
logging.basicConfig(level=logging.INFO)
import networkx as nx

def pso_main():
    orders, locations, vehicles  = load_voratas_vrp('Aac1')
    print("Number of orders:", len(orders))
    print("Number of locations:", len(locations))
    print("Number of vehicles:", len(vehicles))
    # visualize_orders(orders, vehicles=vehicles, title="Aac2", output_name="Aac2")
    psoSolver = PSOSolver(n_particles=POPULATION_SIZE, n_iterations=MAX_ITER)
    psoSolver.init_swarm(orders=orders, vehicles=vehicles)
    history = psoSolver.solve()
    # Plot the line of the best fitness and the personal best fitness
    plt.plot(history['fitness'], label='Global Best')   
    plt.plot(history['p_fitness'], label='Personal Best')

    plt.show()
    plt.savefig('output/pso_fitness.png')

    # Logging the change of the first particle fitness into a file.txt
    with open('output/pso_particle_history.txt', 'w') as f:
        for i in range(len(history['particle_fitness'][1])):
            f.write(f"{i+1}: {history['particle_fitness'][1][i]}\n")
    # Logging the best solution into a file.txt
    # with open('output/pso.best_solution.txt', 'w') as f:
    #     if not psoSolver.final_solution:
    #         f.write("No solution found\n")
    #         return
    #     f.write(f"Best Fitness: {psoSolver.g_fitness}\n")
    #     f.write(f"Number of vehicles used: {len(psoSolver.final_solution)}\n")
    #     f.write(f"Number of orders: {sum([len(o.orders) for o in psoSolver.final_solution])}\n")
    #     f.write("Best Solution\n")
    #     f.write("-"*20 + "\n")
    #     for order_set in psoSolver.final_solution:
    #         if not order_set.orders:
    #             continue
    #         f.write("Orders\n")
    #         for order in order_set.orders.values():
    #             f.write(f"- {order}\n")
    #         f.write("Route\n")
    #         if nx.is_directed_acyclic_graph(order_set):
    #             route, route_cost = order_set.weighted_topological_sort(weight="weight", allow_early=ALLOW_EARLY)
    #             f.write(f"- {route}\n")
    #         f.write("-"*10 + "\n")
            
    print("Done")

if __name__ == "__main__":
    pso_main()
