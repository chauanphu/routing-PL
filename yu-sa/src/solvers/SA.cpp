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
            if (r >= p) {
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

Solution SA::iterate(const VRPInstance& instance, std::vector<int> customer_perm, std::unordered_map<int, int> customer2node, const SAParams& params) {
    Solution sigma_best = Solver::evaluate(instance, customer_perm, customer2node);
    Solution sigma_current = sigma_best;
    int R = 0;
    double T = params.T0;
    bool FBS = false;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> prob(0.0, 1.0);
    int n = customer_perm.size();
    // std::cout << "SA Start: initial objective = " << sigma_best.objective_value << std::endl;
    while (R < params.patience && T > params.Tf) {
        for (int iter = 0; iter < params.max_iter; ++iter) {
            std::vector<int> new_perm = customer_perm;
            // Neighborhood move
            double r = prob(gen);
            if (r <= 1.0/3) {
                int i = gen() % n, j = gen() % n;
                if (i != j) std::swap(new_perm[i], new_perm[j]);
                // std::cout << "[Iter " << iter << "] Swap: " << i << " <-> " << j << std::endl;
            } else if (r <= 2.0/3) {
                int i = gen() % n, j = gen() % n;
                if (i != j) {
                    int val = new_perm[i];
                    new_perm.erase(new_perm.begin() + i);
                    new_perm.insert(new_perm.begin() + j, val);
                }
                // std::cout << "[Iter " << iter << "] Insertion: " << i << " -> " << j << std::endl;
            } else {
                int i = gen() % n, j = gen() % n;
                if (i > j) std::swap(i, j);
                if (i != j) std::reverse(new_perm.begin() + i, new_perm.begin() + j + 1);
                // std::cout << "[Iter " << iter << "] Inversion: " << i << " - " << j << std::endl;
            }
            // Evaluate
            Solution sigma_new = Solver::evaluate(instance, new_perm, customer2node);
            double theta = sigma_new.objective_value - sigma_current.objective_value;
            // std::cout << "  Current obj: " << sigma_current.objective_value << ", New obj: " << sigma_new.objective_value << ", Best obj: " << sigma_best.objective_value << std::endl;
            if (theta <= 0) {
                sigma_current = sigma_new;
                customer_perm = new_perm;
                // std::cout << "  Accepted (improved or equal)" << std::endl;
            } else {
                double r2 = prob(gen);
                double accept_prob = std::exp(-theta / (params.beta * T));
                if (r2 < accept_prob) {
                    sigma_current = sigma_new;
                    customer_perm = new_perm;
                    // std::cout << "  Accepted (worse, prob=" << accept_prob << ")" << std::endl;
                } else {
                    // std::cout << "  Rejected (worse)" << std::endl;
                }
            }
            if (sigma_new.objective_value < sigma_best.objective_value) {
                sigma_best = sigma_new;
                R = 0;
                FBS = true;
                // std::cout << "  New best found: " << sigma_best.objective_value << std::endl;
            }
        }
        T *= params.alpha;
        // std::cout << "Temperature decreased to " << T << ", R = " << R << std::endl;
        if (FBS) {
            FBS = false;
        } else {
            R += 1;
        }
    }
    return sigma_best;
}

Solution SA::solve(const VRPInstance& instance, const SAParams& params) {
    int n = instance.num_customers;
    std::vector<int> customer_perm;
    std::unordered_map<int, int> customer2node;
    initialize_solution(instance, customer_perm, customer2node, params.p);
    return iterate(instance, customer_perm, customer2node, params);
}
