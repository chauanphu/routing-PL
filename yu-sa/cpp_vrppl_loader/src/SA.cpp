#include "SA.h"
#include <algorithm>
#include <random>
#include <vector>

Solution SA::solve(const VRPInstance& instance) {
    Solution sol;
    int n = instance.num_customers;
    // First array: permutation of customer indices (1-based)
    std::vector<int> customer_perm(n);
    for (int i = 0; i < n; ++i) customer_perm[i] = i + 1;
    // Second array: assigned delivery node for each customer
    std::vector<int> delivery_nodes(n);
    for (int i = 0; i < n; ++i) {
        int cust_type = instance.customers[i]->customer_type;
        if (cust_type == 1) {
            // Type-I: assigned to their own node
            delivery_nodes[i] = instance.customers[i]->id;
        } else if (cust_type == 2) {
            // Type-II: assign to preferred locker (first available)
            for (size_t j = 0; j < instance.customer_preferences[i].size(); ++j) {
                if (instance.customer_preferences[i][j]) {
                    delivery_nodes[i] = instance.lockers[j]->id;
                    break;
                }
            }
        } else {
            // Type-III: assign randomly to a preferred locker
            std::vector<int> prefs;
            for (size_t j = 0; j < instance.customer_preferences[i].size(); ++j) {
                if (instance.customer_preferences[i][j])
                    prefs.push_back(instance.lockers[j]->id);
            }
            if (!prefs.empty()) {
                std::random_device rd;
                std::mt19937 g(rd());
                std::uniform_int_distribution<> dis(0, prefs.size() - 1);
                delivery_nodes[i] = prefs[dis(g)];
            } else {
                delivery_nodes[i] = instance.customers[i]->id; // fallback
            }
        }
    }
    // For demonstration, just output the two arrays as a single route
    sol.routes.push_back(customer_perm);
    sol.delivery_nodes = delivery_nodes;
    // Fill the permutation-invariant map
    for (int i = 0; i < n; ++i) {
        int cust_id = i + 1; // 1-based
        sol.customer_to_delivery_node[cust_id] = delivery_nodes[i];
    }
    sol.objective_value = 0.0; // Not computed yet
    return sol;
}
