#include "FA.h"
#include "../core/SolverFactory.h"
#include "../utils.h"
#include <algorithm>
#include <limits>
#include <cmath>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <random>
#include <numeric>
#include <unordered_map>

namespace {
    SolverRegistrar<FA> registrar("fa");
}

struct FAParams {
    double lock_probability;
};

Solution FA::solve(const VRPInstance& instance, const YAML::Node& params_node, bool history, int verbose) {
    FAParams params;
    if(params_node["lock_probability"]) {
        params.lock_probability = params_node["lock_probability"].as<double>();
    } else {
        params.lock_probability = 0.5; // Default value
    }


    int n = instance.num_customers;
    std::mt19937 gen(std::random_device{}());

    // Initialize a random permutation of customers
    std::vector<int> customer_perm(n);
    std::iota(customer_perm.begin(), customer_perm.end(), 1); // Fill with 1, 2, ..., n
    std::shuffle(customer_perm.begin(), customer_perm.end(), gen);

    // Initialize customer-to-node mapping
    std::unordered_map<int, int> customer2node;
    utils::initialize_customer2node(instance, customer2node, params.lock_probability, gen);

    // Evaluate the initial solution
    Solution random_sol = Solver::evaluate(instance, customer_perm, customer2node, false, verbose);

    if (history) {
        std::filesystem::create_directories("../output/experiment");
        std::ofstream csv("../output/experiment/fa.cvr.csv");
        csv << "iter,best_objective\n";
        csv << "0," << random_sol.objective_value << "\n";
        csv.close();
    }

    return random_sol;
}