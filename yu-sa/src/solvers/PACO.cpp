#include "PACO.h"
#include <omp.h>
#include <algorithm>
#include <limits>
#include <fstream>
#include <yaml-cpp/yaml.h>
#include <random>
#include <numeric>

PACOParams PACO::load_params(const std::string& filename) {
    PACOParams params;
    YAML::Node config = YAML::LoadFile(filename);
    params.m = config["m"].as<int>();
    params.alpha = config["alpha"].as<double>();
    params.beta = config["beta"].as<double>();
    params.rho = config["rho"].as<double>();
    params.Q = config["Q"].as<double>();
    params.I = config["I"].as<int>();
    params.t = config["t"].as<int>();
    params.p = config["p"].as<int>();
    return params;
}

// Helper: PACO construction using 3D-ACO transition probability (with delimiters)
static std::pair<std::vector<int>, std::unordered_map<int, int>> paco_construct_permutation(const VRPInstance& instance,
    const std::vector<std::vector<std::vector<double>>>& tau,
    double alpha, double beta, std::mt19937& gen) {
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
    int last_locker = -1;
    std::unordered_map<int, int> customer2node;
    while (!unvisited.empty() && vehicles_used <= m) {
        // Build feasible neighborhood: (customer, option)
        std::vector<std::tuple<int, int>> feasible;
        for (int cust_id : unvisited) {
            for (int o = 0; o < 2; ++o) {
                int delivery_node = (o == 0) ? cust_id : 0;
                // Only allow home (o=0) if customer type is not II, locker (o=1) if not type I
                auto c = instance.customers[cust_id-1];
                if ((o == 0 && c->customer_type == 2) || (o == 1 && c->customer_type == 1)) continue;
                if (o == 1) {
                    // Locker assignment: pick first preferred locker or first locker
                    int assigned = -1;
                    if (!instance.customer_preferences.empty() && cust_id-1 < instance.customer_preferences.size()) {
                        for (size_t j = 0; j < instance.customer_preferences[cust_id-1].size(); ++j) {
                            if (instance.customer_preferences[cust_id-1][j] == 1) {
                                assigned = instance.lockers[j]->id;
                                break;
                            }
                        }
                    }
                    if (assigned == -1 && !instance.lockers.empty()) assigned = instance.lockers[0]->id;
                    if (assigned == -1) continue; // no locker available
                    delivery_node = assigned;
                }
                int demand = c->demand;
                int early = 0, late = 0;
                if (delivery_node <= instance.num_customers) {
                    auto cc = instance.customers[delivery_node-1];
                    early = cc->early_time;
                    late = cc->late_time;
                } else {
                    auto l = instance.lockers[delivery_node-1-instance.num_customers];
                    early = l->early_time;
                    late = l->late_time;
                }
                double arr_time = time + instance.distance_matrix[curr_node][delivery_node];
                // Locker revisit violation: if last_locker == delivery_node and o==1, skip
                if (o == 1 && last_locker == delivery_node) continue;
                if (load + demand <= instance.vehicle_capacity && arr_time <= late) {
                    feasible.emplace_back(cust_id, o);
                }
            }
        }
        if (feasible.empty()) {
            // No feasible customer, close route
            perm.push_back(depot_id);
            curr_node = depot_id;
            load = 0;
            time = 0;
            vehicles_used++;
            last_locker = -1;
            continue;
        }
        // Compute probabilities
        std::vector<double> probs;
        double sum_prob = 0.0;
        for (auto [cust_id, o] : feasible) {
            int delivery_node = (o == 0) ? cust_id : 0;
            if (o == 1) {
                int assigned = -1;
                if (!instance.customer_preferences.empty() && cust_id-1 < instance.customer_preferences.size()) {
                    for (size_t j = 0; j < instance.customer_preferences[cust_id-1].size(); ++j) {
                        if (instance.customer_preferences[cust_id-1][j] == 1) {
                            assigned = instance.lockers[j]->id;
                            break;
                        }
                    }
                }
                if (assigned == -1 && !instance.lockers.empty()) assigned = instance.lockers[0]->id;
                delivery_node = assigned;
            }
            double tau_ijo = tau[curr_node][delivery_node][o];
            double eta_ijo = 1.0 / (instance.distance_matrix[curr_node][delivery_node] + 1e-6);
            double val = std::pow(tau_ijo, alpha) * std::pow(eta_ijo, beta);
            probs.push_back(val);
            sum_prob += val;
        }
        for (double& p : probs) p /= sum_prob;
        std::discrete_distribution<> dist(probs.begin(), probs.end());
        int idx = dist(gen);
        int next_cust, next_o;
        std::tie(next_cust, next_o) = feasible[idx];
        int delivery_node = (next_o == 0) ? next_cust : 0;
        if (next_o == 1) {
            int assigned = -1;
            if (!instance.customer_preferences.empty() && next_cust-1 < instance.customer_preferences.size()) {
                for (size_t j = 0; j < instance.customer_preferences[next_cust-1].size(); ++j) {
                    if (instance.customer_preferences[next_cust-1][j] == 1) {
                        assigned = instance.lockers[j]->id;
                        break;
                    }
                }
            }
            if (assigned == -1 && !instance.lockers.empty()) assigned = instance.lockers[0]->id;
            delivery_node = assigned;
        }
        perm.push_back(next_cust);
        customer2node[next_cust] = delivery_node;
        load += instance.customers[next_cust-1]->demand;
        double arr_time = time + instance.distance_matrix[curr_node][delivery_node];
        if (delivery_node <= instance.num_customers) {
            time = std::max(arr_time, (double)instance.customers[delivery_node-1]->early_time);
        } else {
            time = std::max(arr_time, (double)instance.lockers[delivery_node-1-instance.num_customers]->early_time);
        }
        curr_node = delivery_node;
        if (next_o == 1) last_locker = delivery_node; else last_locker = -1;
        unvisited.erase(std::remove(unvisited.begin(), unvisited.end(), next_cust), unvisited.end());
    }
    if (perm.back() != depot_id) perm.push_back(depot_id);
    return {perm, customer2node};
}

Solution PACO::solve(const VRPInstance& instance, const PACOParams& params) {
    // Problem size
    int n = instance.num_customers + 1; // including depot
    int m = params.m;
    int p = params.p;
    int t = params.t;
    int I = params.I;
    // 3D pheromone matrix: tau[i][j][o]
    std::vector<std::vector<std::vector<double>>> tau(n, std::vector<std::vector<double>>(n, std::vector<double>(2, 1.0)));
    // Build 3D feasibility mask M[i][j][o] according to prompt
    std::vector<std::vector<std::vector<int>>> mask(n, std::vector<std::vector<int>>(n, std::vector<int>(2, 1)));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            // o = 0: home, o = 1: locker
            // Prevent self-delivery
            mask[i][j][0] = (i != j) ? 1 : 0;
            mask[i][j][1] = (i != j) ? 1 : 0;
            // Only allow home for non-type-II
            if (j > 0 && j <= instance.num_customers) {
                auto c = instance.customers[j-1];
                if (c->customer_type == 2) mask[i][j][0] = 0; // type-II: no home
                if (c->customer_type == 1) mask[i][j][1] = 0; // type-I: no locker
            }
        }
    }
    // Apply mask once
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int o = 0; o < 2; ++o)
                if (!mask[i][j][o]) tau[i][j][o] = 0.0;
    Solution global_best;
    double global_best_obj = std::numeric_limits<double>::max();
    // Main loop
    for (int iter = 0; iter < I; ++iter) {
        std::vector<Solution> all_solutions(m);
        std::vector<double> all_objs(m);
        // Parallel region: each thread builds m/p ants
        #pragma omp parallel num_threads(p)
        {
            int tid = omp_get_thread_num();
            int m_p = m / p;
            int start = tid * m_p;
            int end = (tid == p-1) ? m : start + m_p;
            std::random_device rd; std::mt19937 gen(rd() + tid);
            for (int k = start; k < end; ++k) {
                auto [perm, customer2node] = paco_construct_permutation(instance, tau, params.alpha, params.beta, gen);
                Solution sol = Solver::evaluate(instance, perm, customer2node, true);
                all_solutions[k] = sol;
                all_objs[k] = sol.objective_value;
            }
        }
        // Select top-t ants
        std::vector<int> idx(m);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin()+t, idx.end(), [&](int a, int b){ return all_objs[a] < all_objs[b]; });
        // Pheromone evaporation
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                for (int o = 0; o < 2; ++o)
                    tau[i][j][o] *= (1.0 - params.rho);
        // Pheromone update by top-t ants
        for (int rank = 0; rank < t; ++rank) {
            int k = idx[rank];
            const auto& perm = all_solutions[k].customer_permutation;
            const auto& customer2node = all_solutions[k].customer2node;
            for (size_t s = 1; s < perm.size(); ++s) {
                int prev = perm[s-1];
                int curr = perm[s];
                if (prev == 0 && curr == 0) continue; // skip depot-depot
                if (curr == 0) continue; // depot delimiter
                // Determine delivery option o
                int o = (customer2node.at(curr) == curr) ? 0 : 1;
                tau[prev][customer2node.at(curr)][o] += params.Q / all_objs[k];
            }
        }
        // Update global best
        if (*std::min_element(all_objs.begin(), all_objs.end()) < global_best_obj) {
            int best_idx = std::min_element(all_objs.begin(), all_objs.end()) - all_objs.begin();
            global_best = all_solutions[best_idx];
            global_best_obj = all_objs[best_idx];
        }
    }
    // Return best solution
    return global_best;
}