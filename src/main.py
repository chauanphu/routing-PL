from utils.load_data import load_voratas_vrp
import logging
from matplotlib import pyplot as plt
from utils.solver import PSOSolver
logging.basicConfig(level=logging.INFO)
import yaml

# def sa_main():
#     # Load the Solomon VRP benchmark file
#     vehicle_info, customer_data = load_solomon_vrp('benchmark/C101.txt')
#     print("Vehicle Info:", vehicle_info.__dict__)
#     # Convert to orders
#     orders = convert_to_orders(customer_data)
#     print(f"Number of orders: {len(orders)}")
#     end_depot = Location(0, customer_data[0].xcoord, customer_data[0].ycoord)
#     initial_solution: SASolution = init_solution(orders, end_depot, vehicle_info.capacity)
#     print("Initial Solution:", initial_solution.total_distance())
#     solver = SimulatedAnnealingSolver(initial_solution, initial_temp=1000, cooling_rate=0.95, stopping_temp=1)
#     best_solution, history = solver.anneal()
#     # Save history to a JSON file
#     with open('output/history.json', 'w') as f:
#         json.dump(history, f)

def pso_main():
    orders, locations, vehicles  = load_voratas_vrp('benchmark/mdpdr/Aac1.txt', 20)
    print("Number of orders:", len(orders))
    print("Number of locations:", len(locations))
    print("Number of vehicles:", len(vehicles))
    # visualize_orders(orders, vehicles=vehicles, title="Aac2", output_name="Aac2")
    psoSolver = PSOSolver(n_particles=200, n_iterations=50)
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
    print("Done")

if __name__ == "__main__":
    pso_main()
