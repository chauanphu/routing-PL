from utils.load_data import Location, convert_to_orders, load_solomon_vrp, load_voratas_vrp, visualize_orders
import logging

from utils.solver import PSOSolver
logging.basicConfig(level=logging.INFO)

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
    psoSolver = PSOSolver()
    psoSolver.init_swarm(orders=orders, vehicles=vehicles)
    psoSolver.fitness()
    # for particle in psoSolver.particles:
    #     print(particle)
    #     print("-"*10)

if __name__ == "__main__":
    pso_main()
