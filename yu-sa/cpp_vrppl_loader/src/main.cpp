#include "InstanceParser.h"
#include "Solver.h"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <instance_file>" << std::endl;
        return 1;
    }
    std::string filename = argv[1];
    VRPInstance instance = InstanceParser::parse(filename);
    std::cout << "Loaded instance: " << filename << std::endl;
    std::cout << "Customers: " << instance.num_customers << std::endl;
    std::cout << "Lockers: " << instance.num_lockers << std::endl;
    std::cout << "Vehicles: " << instance.num_vehicles << ", Capacity: " << instance.vehicle_capacity << std::endl;
    if (instance.depot) {
        std::cout << "Depot at (" << instance.depot->x << ", " << instance.depot->y << ")" << std::endl;
    }
    for (size_t i = 0; i < std::min<size_t>(instance.customers.size(), 3); ++i) {
        auto& c = instance.customers[i];
        std::cout << "Customer " << c->id << ": demand=" << c->demand << ", type=" << c->customer_type << ", loc=(" << c->x << ", " << c->y << ")" << std::endl;
    }
    // Call the pseudo-solver
    Solution sol = Solver::solve(instance);
    std::cout << "\nPseudo-solver result:" << std::endl;
    std::cout << "Objective value: " << sol.objective_value << std::endl;
    if (sol.routes.empty()) {
        std::cout << "No feasible solution found." << std::endl;
    } else {
        for (size_t i = 0; i < sol.routes.size(); ++i) {
            std::cout << "Route " << i+1 << ": ";
            for (int nid : sol.routes[i]) std::cout << nid << " ";
            std::cout << std::endl;
        }
    }
    return 0;
}
