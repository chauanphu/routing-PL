#include "ThreeDACO.h"
#include <algorithm>
#include <limits>
#include <cmath>
#include <iostream>
#include <unordered_map>

// Helper: Build feasibility mask for delivery mode
static void build_feasibility_mask(const VRPInstance& instance, std::vector<std::vector<std::vector<int>>>& mask) {
    int n = instance.num_customers;
    mask.assign(n+1, std::vector<std::vector<int>>(n+1, std::vector<int>(2, 1)));
    for (int j = 1; j <= n; ++j) {
        auto c = instance.customers[j-1];
        if (c->customer_type == 1) {
            for (int i = 0; i <= n; ++i) mask[i][j][1] = 0; // No locker
        } else if (c->customer_type == 2) {
            for (int i = 0; i <= n; ++i) mask[i][j][0] = 0; // No home
        }
    }
}

// Construct solution for one ant using 3D pheromone, compatible with C++ core
static std::tuple<std::vector<std::tuple<int,int,int>>, std::vector<int>, std::unordered_map<int,int>, double> construct_solution(
    const VRPInstance& instance,
    const std::vector<std::vector<std::vector<double>>>& pheromones,
    const std::vector<std::vector<std::vector<int>>>& mask,
    double alpha, double beta, std::mt19937& gen) {
    int n = instance.num_customers;
    std::vector<int> unvisited(n);
    for (int i = 0; i < n; ++i) unvisited[i] = i+1;
    std::vector<std::tuple<int,int,int>> transitions;
    std::vector<int> customer_perm;
    std::unordered_map<int,int> customer2node;
    int prev = 0;
    int load = 0;
    double time = 0;
    int vehicles_used = 1;
    double total_dist = 0.0;
    while (!unvisited.empty() && vehicles_used <= instance.num_vehicles) {
        std::vector<std::tuple<int,int,double>> candidates; // (j, mode, value)
        for (int idx = 0; idx < (int)unvisited.size(); ++idx) {
            int j = unvisited[idx];
            auto c = instance.customers[j-1];
            std::vector<int> allowed;
            if (c->customer_type == 1) allowed = {0};
            else if (c->customer_type == 2) allowed = {1};
            else allowed = {0,1};
            for (int d : allowed) {
                int delivery_node;
                if (d == 0) {
                    delivery_node = j; // home
                } else {
                    int assigned = -1;
                    if (!instance.customer_preferences.empty() && (j-1) < (int)instance.customer_preferences.size()) {
                        for (size_t lid = 0; lid < instance.customer_preferences[j-1].size(); ++lid) {
                            if (instance.customer_preferences[j-1][lid] == 1) {
                                assigned = instance.lockers[lid]->id;
                                break;
                            }
                        }
                    }
                    if (assigned == -1 && !instance.lockers.empty())
                        assigned = instance.lockers[0]->id;
                    if (assigned == -1) assigned = j; // fallback
                    delivery_node = assigned;
                }
                int demand = c->demand;
                int early, late;
                if (delivery_node <= instance.num_customers) {
                    auto cc = instance.customers[delivery_node-1];
                    early = cc->early_time;
                    late = cc->late_time;
                } else {
                    auto l = instance.lockers[delivery_node-1-instance.num_customers];
                    early = l->early_time;
                    late = l->late_time;
                }
                double dist = instance.distance_matrix[prev][delivery_node];
                double arr_time = time + dist;
                if (load + demand <= instance.vehicle_capacity && arr_time <= late) {
                    double tau = pheromones[prev][j][d];
                    double heuristic = 1.0 / (dist + 1e-6);
                    double value = std::pow(tau, alpha) * std::pow(heuristic, beta);
                    candidates.emplace_back(j, d, value);
                }
            }
        }
        if (candidates.empty()) {
            // No feasible, close route
            customer_perm.push_back(0); // depot delimiter
            prev = 0;
            load = 0;
            time = 0;
            vehicles_used++;
            continue;
        }
        // Probabilistic selection
        double sum = 0.0;
        for (auto& tup : candidates) sum += std::get<2>(tup);
        std::vector<double> probs;
        for (auto& tup : candidates) probs.push_back(std::get<2>(tup)/sum);
        std::discrete_distribution<> dist(probs.begin(), probs.end());
        int sel = dist(gen);
        int j = std::get<0>(candidates[sel]);
        int d = std::get<1>(candidates[sel]);
        auto c = instance.customers[j-1];
        int delivery_node;
        if (d == 0) {
            delivery_node = j;
        } else {
            int assigned = -1;
            if (!instance.customer_preferences.empty() && (j-1) < (int)instance.customer_preferences.size()) {
                for (size_t lid = 0; lid < instance.customer_preferences[j-1].size(); ++lid) {
                    if (instance.customer_preferences[j-1][lid] == 1) {
                        assigned = instance.lockers[lid]->id;
                        break;
                    }
                }
            }
            if (assigned == -1 && !instance.lockers.empty())
                assigned = instance.lockers[0]->id;
            if (assigned == -1) assigned = j;
            delivery_node = assigned;
        }
        transitions.emplace_back(prev, j, d);
        customer_perm.push_back(j);
        customer2node[j] = delivery_node;
        double move_dist = instance.distance_matrix[prev][delivery_node];
        total_dist += move_dist;
        load += c->demand;
        double arr_time = time + move_dist;
        if (delivery_node <= instance.num_customers) {
            auto cc = instance.customers[delivery_node-1];
            time = std::max(arr_time, (double)cc->early_time);
        } else {
            auto l = instance.lockers[delivery_node-1-instance.num_customers];
            time = std::max(arr_time, (double)l->early_time);
        }
        prev = delivery_node;
        unvisited.erase(std::remove(unvisited.begin(), unvisited.end(), j), unvisited.end());
    }
    customer_perm.push_back(0); // end with depot
    return {transitions, customer_perm, customer2node, total_dist};
}

static void update_pheromones(
    std::vector<std::vector<std::vector<double>>>& pheromones,
    const std::vector<std::tuple<std::vector<std::tuple<int,int,int>>, double>>& solutions,
    double evaporation_rate, double Q, int num_elitist) {
    int n = pheromones.size();
    int m = pheromones[0].size();
    int d = pheromones[0][0].size();
    // Evaporation
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            for (int k = 0; k < d; ++k)
                pheromones[i][j][k] *= (1.0 - evaporation_rate);
    // Sort and select top-t
    auto sorted = solutions;
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b){ return std::get<1>(a) < std::get<1>(b); });
    for (int s = 0; s < std::min(num_elitist, (int)sorted.size()); ++s) {
        const auto& transitions = std::get<0>(sorted[s]);
        double fitness = std::get<1>(sorted[s]);
        double deposit = Q / (fitness > 0 ? fitness : 1e-8);
        for (const auto& t : transitions) {
            int i,j,o; std::tie(i,j,o) = t;
            pheromones[i][j][o] += deposit;
        }
    }
}

Solution ThreeDACO::solve(const VRPInstance& instance, const ThreeDACOParams& params) {
    int n = instance.num_customers;
    std::mt19937 gen(std::random_device{}());
    // 3D pheromone matrix (n+1 x n+1 x 2)
    std::vector<std::vector<std::vector<double>>> pheromones(n+1, std::vector<std::vector<double>>(n+1, std::vector<double>(2, 2.0)));
    std::vector<std::vector<std::vector<int>>> mask;
    build_feasibility_mask(instance, mask);
    // Apply mask to pheromones (zero out infeasible)
    for (int j = 1; j <= n; ++j) {
        auto c = instance.customers[j-1];
        if (c->customer_type == 1) for (int i = 0; i <= n; ++i) pheromones[i][j][1] = 0.0;
        else if (c->customer_type == 2) for (int i = 0; i <= n; ++i) pheromones[i][j][0] = 0.0;
    }
    double best_fitness = 1e12;
    Solution best_sol;
    for (int iter = 0; iter < params.num_iterations; ++iter) {
        std::vector<std::tuple<std::vector<std::tuple<int,int,int>>, double>> ant_solutions;
        for (int ant = 0; ant < params.num_ants; ++ant) {
            auto [transitions, customer_perm, customer2node, fitness] = construct_solution(instance, pheromones, mask, params.alpha, params.beta, gen);
            Solution sol = Solver::evaluate(instance, customer_perm, customer2node, false); // Use greedy-insertion (construct_routes)
            ant_solutions.emplace_back(transitions, sol.objective_value);
            if (sol.objective_value < best_fitness && !sol.routes.empty()) {
                best_fitness = sol.objective_value;
                best_sol = sol;
            }
        }
        // Update pheromones
        update_pheromones(pheromones, ant_solutions, params.evaporation_rate, params.Q, params.num_elitist);
    }
    return best_sol;
}
