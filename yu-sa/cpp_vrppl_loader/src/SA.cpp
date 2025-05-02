#include "SA.h"
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <iostream>

// Helper: initialize the initial solution for SA
static void initialize_solution(const VRPInstance& instance, std::vector<int>& customer_perm, std::unordered_map<int, int>& customer2node, double p = 0.5) {
    int n = instance.num_customers;
    customer_perm.clear();
    customer2node.clear();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> prob(0.0, 1.0);
    // Step 1: assign delivery nodes for each customer as before
    std::vector<int> assigned_delivery_node(n, -1);
    for (int i = 0; i < n; ++i) {
        auto c = instance.customers[i];
        if (c->customer_type == 1) {
            assigned_delivery_node[i] = c->id;
        } else if (c->customer_type == 2) {
            int assigned = -1;
            if (!instance.customer_preferences.empty() && i < instance.customer_preferences.size()) {
                for (size_t j = 0; j < instance.customer_preferences[i].size(); ++j) {
                    if (instance.customer_preferences[i][j] == 1) {
                        assigned = instance.lockers[j]->id;
                        break;
                    }
                }
            }
            if (assigned != -1) {
                assigned_delivery_node[i] = assigned;
            } else if (!instance.lockers.empty()) {
                assigned_delivery_node[i] = instance.lockers[0]->id;
            } else {
                assigned_delivery_node[i] = c->id; // fallback
            }
        } else if (c->customer_type == 3) {
            double r = prob(gen);
            int assigned = -1;
            if (r < p) {
                if (!instance.customer_preferences.empty() && i < instance.customer_preferences.size()) {
                    for (size_t j = 0; j < instance.customer_preferences[i].size(); ++j) {
                        if (instance.customer_preferences[i][j] == 1) {
                            assigned = instance.lockers[j]->id;
                            break;
                        }
                    }
                }
                if (assigned != -1) {
                    assigned_delivery_node[i] = assigned;
                } else if (!instance.lockers.empty()) {
                    assigned_delivery_node[i] = instance.lockers[0]->id;
                } else {
                    assigned_delivery_node[i] = c->id; // fallback
                }
            } else {
                assigned_delivery_node[i] = c->id;
            }
        }
        customer2node[c->id] = assigned_delivery_node[i];
    }
    // Step 2: nearest neighbor assignment for permutation
    std::vector<bool> assigned(n, false);
    int current_node = 0; // depot
    for (int step = 0; step < n; ++step) {
        double min_dist = std::numeric_limits<double>::max();
        int next_customer = -1;
        for (int i = 0; i < n; ++i) {
            if (assigned[i]) continue;
            int delivery_node = assigned_delivery_node[i];
            double dist = instance.distance_matrix[current_node][delivery_node];
            if (dist < min_dist) {
                min_dist = dist;
                next_customer = i;
            }
        }
        if (next_customer == -1) break;
        assigned[next_customer] = true;
        customer_perm.push_back(next_customer + 1); // customer IDs are 1-based
        current_node = assigned_delivery_node[next_customer];
    }
}

Solution SA::solve(const VRPInstance& instance) {
    int n = instance.num_customers;
    std::vector<int> customer_perm;
    std::unordered_map<int, int> customer2node;
    initialize_solution(instance, customer_perm, customer2node);
    // Evaluate using the framework
    // Print out customer_perm
    std::cout << "Customer permutation: ";
    for (const auto& customer : customer_perm) {
        std::cout << customer << " ";
    }
    std::cout << std::endl;

    // // Print out customer2node
    // std::cout << "Customer to node mapping:" << std::endl;
    // for (const auto& pair : customer2node) {
    //     std::cout << "  Customer " << pair.first << " -> Node " << pair.second << std::endl;
    // }
    return Solver::evaluate(instance, customer_perm, customer2node);
}
