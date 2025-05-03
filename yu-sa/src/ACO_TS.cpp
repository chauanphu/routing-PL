#include "ACO_TS.h"
#include <algorithm>
#include <limits>
#include <cmath>
#include <iostream>

// Helper: initialize delivery node mapping for each customer
static void initialize_customer2node(const VRPInstance& instance, std::unordered_map<int, int>& customer2node, double p, std::mt19937& gen) {
    int n = instance.num_customers;
    std::uniform_real_distribution<> prob(0.0, 1.0);
    for (int i = 0; i < n; ++i) {
        auto c = instance.customers[i];
        if (c->customer_type == 1) {
            customer2node[c->id] = c->id;
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
                customer2node[c->id] = assigned;
            } else if (!instance.lockers.empty()) {
                customer2node[c->id] = instance.lockers[0]->id;
            } else {
                customer2node[c->id] = c->id;
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
                    customer2node[c->id] = assigned;
                } else if (!instance.lockers.empty()) {
                    customer2node[c->id] = instance.lockers[0]->id;
                } else {
                    customer2node[c->id] = c->id;
                }
            } else {
                customer2node[c->id] = c->id;
            }
        }
    }
}

// Helper: construct a random initial permutation with depot delimiters
static std::vector<int> random_delim_permutation(int n, int m, std::mt19937& gen) {
    std::vector<int> perm(n);
    for (int i = 0; i < n; ++i) perm[i] = i + 1;
    std::shuffle(perm.begin(), perm.end(), gen);
    // Insert m+1 depot delimiters (0) at start, between, and end
    std::vector<int> result;
    int per_route = (n + m - 1) / m;
    int idx = 0;
    for (int v = 0; v < m; ++v) {
        result.push_back(0);
        for (int j = 0; j < per_route && idx < n; ++j) {
            result.push_back(perm[idx++]);
        }
    }
    result.push_back(0);
    return result;
}

// Helper: ACO construction using transition probability (delimeter-based)
static std::vector<int> aco_construct_permutation(const VRPInstance& instance, const std::unordered_map<int, int>& customer2node, const std::vector<std::vector<double>>& tau, double alpha, double beta, std::mt19937& gen) {
    int n = instance.num_customers;
    int m = instance.num_vehicles;
    int depot_id = 0;
    std::vector<int> unvisited;
    for (int i = 1; i <= n; ++i) unvisited.push_back(i);
    std::vector<int> perm;
    perm.push_back(depot_id); // start at depot
    int curr_node = depot_id;
    int load = 0;
    double time = 0;
    int vehicles_used = 1;
    while (!unvisited.empty() && vehicles_used <= m) {
        // Build feasible neighborhood
        std::vector<int> feasible;
        for (int cust_id : unvisited) {
            int delivery_node = customer2node.at(cust_id);
            int demand = 0;
            int early = 0, late = 0;
            if (delivery_node <= instance.num_customers) {
                auto c = instance.customers[delivery_node-1];
                demand = c->demand;
                early = c->early_time;
                late = c->late_time;
            } else {
                auto l = instance.lockers[delivery_node-1-instance.num_customers];
                auto c = instance.customers[cust_id-1];
                demand = c->demand;
                early = l->early_time;
                late = l->late_time;
            }
            double arr_time = time + instance.distance_matrix[curr_node][delivery_node];
            if (load + demand <= instance.vehicle_capacity && arr_time <= late) {
                feasible.push_back(cust_id);
            }
        }
        if (feasible.empty()) {
            // No feasible customer, close route
            perm.push_back(depot_id);
            curr_node = depot_id;
            load = 0;
            time = 0;
            vehicles_used++;
            continue;
        }
        // Compute probabilities
        std::vector<double> probs;
        double sum_prob = 0.0;
        for (int cust_id : feasible) {
            int delivery_node = customer2node.at(cust_id);
            double tau_ij = tau[curr_node][delivery_node];
            double eta_ij = 1.0 / (instance.distance_matrix[curr_node][delivery_node] + 1e-6);
            double val = std::pow(tau_ij, alpha) * std::pow(eta_ij, beta);
            probs.push_back(val);
            sum_prob += val;
        }
        // Normalize
        for (double& p : probs) p /= sum_prob;
        // Roulette wheel selection
        std::discrete_distribution<> dist(probs.begin(), probs.end());
        int idx = dist(gen);
        int next_cust = feasible[idx];
        int delivery_node = customer2node.at(next_cust);
        // Update state
        perm.push_back(next_cust);
        load += (delivery_node <= instance.num_customers)
            ? instance.customers[delivery_node-1]->demand
            : instance.customers[next_cust-1]->demand;
        double arr_time = time + instance.distance_matrix[curr_node][delivery_node];
        if (delivery_node <= instance.num_customers) {
            time = std::max(arr_time, (double)instance.customers[delivery_node-1]->early_time);
        } else {
            time = std::max(arr_time, (double)instance.lockers[delivery_node-1-instance.num_customers]->early_time);
        }
        curr_node = delivery_node;
        // Remove from unvisited
        unvisited.erase(std::remove(unvisited.begin(), unvisited.end(), next_cust), unvisited.end());
    }
    // End with depot
    if (perm.back() != depot_id) perm.push_back(depot_id);
    return perm;
}

// Tabu Search for VRPPL (exchange, insert, 2-opt)
static Solution tabu_search(const VRPInstance& instance, const Solution& init_sol, const ACOParams& params, std::vector<std::vector<double>>& tau, int max_iter = 100, int max_no_improve = 20) {
    int n = instance.num_customers;
    int tabu_tenure = std::max(1, n / 2);
    std::mt19937 gen(std::random_device{}());
    Solution best = init_sol;
    Solution curr = init_sol;
    int no_improve = 0;
    // Tabu list: customer_id -> iteration when tabu expires
    std::unordered_map<int, int> tabu_list;
    for (int iter = 0; iter < max_iter && no_improve < max_no_improve; ++iter) {
        Solution best_candidate;
        best_candidate.objective_value = 1e12;
        std::vector<int> best_move; // [type, route1, pos1, route2, pos2]
        // 1. Exchange between routes
        for (size_t r1 = 0; r1 < curr.routes.size(); ++r1) {
            for (size_t i = 1; i + 1 < curr.routes[r1].size(); ++i) {
                int cust1 = curr.routes[r1][i];
                if (cust1 == 0) continue;
                for (size_t r2 = 0; r2 < curr.routes.size(); ++r2) {
                    if (r1 == r2) continue;
                    for (size_t j = 1; j + 1 < curr.routes[r2].size(); ++j) {
                        int cust2 = curr.routes[r2][j];
                        if (cust2 == 0) continue;
                        // Swap
                        auto new_routes = curr.routes;
                        std::swap(new_routes[r1][i], new_routes[r2][j]);
                        // Flatten to permutation with delimiters
                        std::vector<int> perm;
                        for (const auto& route : new_routes) {
                            for (int v : route) perm.push_back(v);
                        }
                        Solution cand = Solver::evaluate(instance, perm, curr.customer2node, true);
                        if (cand.objective_value < best_candidate.objective_value && (tabu_list[cust1] < iter || cand.objective_value < best.objective_value)) {
                            best_candidate = cand;
                            best_move = {1, (int)r1, (int)i, (int)r2, (int)j};
                        }
                    }
                }
            }
        }
        // 2. Insert customer from one route to another
        for (size_t r1 = 0; r1 < curr.routes.size(); ++r1) {
            for (size_t i = 1; i + 1 < curr.routes[r1].size(); ++i) {
                int cust = curr.routes[r1][i];
                if (cust == 0) continue;
                for (size_t r2 = 0; r2 < curr.routes.size(); ++r2) {
                    if (r1 == r2) continue;
                    for (size_t j = 1; j < curr.routes[r2].size(); ++j) {
                        // Insert cust from r1,i to r2,j
                        auto new_routes = curr.routes;
                        int val = new_routes[r1][i];
                        new_routes[r1].erase(new_routes[r1].begin() + i);
                        new_routes[r2].insert(new_routes[r2].begin() + j, val);
                        // Flatten
                        std::vector<int> perm;
                        for (const auto& route : new_routes) {
                            for (int v : route) perm.push_back(v);
                        }
                        Solution cand = Solver::evaluate(instance, perm, curr.customer2node, true);
                        if (cand.objective_value < best_candidate.objective_value && (tabu_list[cust] < iter || cand.objective_value < best.objective_value)) {
                            best_candidate = cand;
                            best_move = {2, (int)r1, (int)i, (int)r2, (int)j};
                        }
                    }
                }
            }
        }
        // 3. 2-opt within each route
        for (size_t r = 0; r < curr.routes.size(); ++r) {
            for (size_t i = 1; i + 2 < curr.routes[r].size(); ++i) {
                for (size_t j = i + 1; j + 1 < curr.routes[r].size(); ++j) {
                    auto new_routes = curr.routes;
                    std::reverse(new_routes[r].begin() + i, new_routes[r].begin() + j + 1);
                    std::vector<int> perm;
                    for (const auto& route : new_routes) {
                        for (int v : route) perm.push_back(v);
                    }
                    Solution cand = Solver::evaluate(instance, perm, curr.customer2node, true);
                    if (cand.objective_value < best_candidate.objective_value) {
                        best_candidate = cand;
                        best_move = {3, (int)r, (int)i, (int)j};
                    }
                }
            }
        }
        // If no improvement, break
        if (best_candidate.objective_value >= curr.objective_value) {
            no_improve++;
        } else {
            no_improve = 0;
        }
        // Apply best move
        if (!best_move.empty()) {
            if (best_move[0] == 1) { // exchange
                int r1 = best_move[1], i = best_move[2], r2 = best_move[3], j = best_move[4];
                tabu_list[curr.routes[r1][i]] = iter + tabu_tenure;
                tabu_list[curr.routes[r2][j]] = iter + tabu_tenure;
            } else if (best_move[0] == 2) { // insert
                int r1 = best_move[1], i = best_move[2];
                tabu_list[curr.routes[r1][i]] = iter + tabu_tenure;
            }
            curr = best_candidate;
            // Pheromone update for each move
            for (const auto& route : curr.routes) {
                for (size_t i = 1; i < route.size(); ++i) {
                    int u = route[i-1], v = route[i];
                    tau[u][v] += params.Q / curr.objective_value;
                }
            }
            if (curr.objective_value < best.objective_value) {
                best = curr;
            }
        } else {
            break;
        }
    }
    return best;
}

Solution ACO_TS::solve(const VRPInstance& instance, const ACOParams& params) {
    int n = instance.num_customers;
    int m = instance.num_vehicles;
    int num_nodes = n + instance.num_lockers + 1;
    std::mt19937 gen(std::random_device{}());
    // Pheromone matrix (node x node)
    std::vector<std::vector<double>> tau(num_nodes, std::vector<double>(num_nodes, 1.0));
    Solution best_sol;
    best_sol.objective_value = 1e12;
    int stagnation = 0;
    int stagnation_limit = 3;
    for (int iter = 0; iter < params.num_iterations; ++iter) {
        std::vector<Solution> ant_sols(params.num_ants);
        for (int k = 0; k < params.num_ants; ++k) {
            // 1. Assign delivery nodes
            std::unordered_map<int, int> customer2node;
            initialize_customer2node(instance, customer2node, params.p, gen);
            // 2. Build a solution (permutation with delimiters)
            std::vector<int> perm = aco_construct_permutation(instance, customer2node, tau, params.alpha, params.beta, gen);
            // 3. Construct solution using delimiter-based evaluation
            Solution sol = Solver::evaluate(instance, perm, customer2node, true);
            ant_sols[k] = sol;
        }
        // Find best ant
        auto best_ant = std::min_element(ant_sols.begin(), ant_sols.end(), [](const Solution& a, const Solution& b) {
            return a.objective_value < b.objective_value;
        });
        if (best_ant->objective_value < best_sol.objective_value) {
            best_sol = *best_ant;
            stagnation = 0;
        } else {
            stagnation++;
        }
        // Pheromone evaporation
        for (int i = 0; i < num_nodes; ++i)
            for (int j = 0; j < num_nodes; ++j)
                tau[i][j] *= (1.0 - params.rho);
        // Pheromone update (only best ant)
        for (const auto& route : best_ant->routes) {
            for (size_t i = 1; i < route.size(); ++i) {
                int u = route[i-1], v = route[i];
                tau[u][v] += params.Q / best_ant->objective_value;
            }
        }
        // Tabu Search integration
        if (stagnation >= stagnation_limit) {
            best_sol = tabu_search(instance, best_sol, params, tau);
            stagnation = 0;
        }
        // Optionally print progress
        // std::cout << "[ACO] Iter " << iter << ": Best = " << best_sol.objective_value << std::endl;
    }
    return best_sol;
}
