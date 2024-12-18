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