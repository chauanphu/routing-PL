#include "PACO.h"
#include <omp.h>
#include <algorithm>
#include <limits>
#include <fstream>
#include <yaml-cpp/yaml.h>
#include <random>
#include <numeric>
#include <iostream>
#include <filesystem>
#include <cmath>

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
// Helper: PACO ant solution construction using 3D-ACO transition rule
static std::pair<std::vector<int>, std::unordered_map<int, int>>
paco_construct_solution(const VRPInstance& instance,
    const std::vector<std::vector<std::vector<double>>>& tau,
    double alpha, double beta, std::mt19937& gen) {
    int n = instance.num_customers;
    int depot_id = 0;
    std::vector<int> unvisited;
    for (int i = 1; i <= n; ++i) unvisited.push_back(i);
    std::vector<int> perm;
    std::unordered_map<int, int> customer2node;
    int prev_index = depot_id;
    while (!unvisited.empty()) {
        std::vector<std::tuple<int, int, double>> candidate_options; // (j, d, value)
        for (int j : unvisited) {
            auto c = instance.customers[j-1];
            std::vector<int> allowed;
            if (c->customer_type == 1) allowed = {0};
            else if (c->customer_type == 2) allowed = {1};
            else allowed = {0, 1};
            for (int d : allowed) {
                if (tau[prev_index][j][d] == 0.0) continue; // skip infeasible
                double value = std::pow(tau[prev_index][j][d], alpha);
                candidate_options.emplace_back(j, d, value);
            }
        }
        double sum = 0.0;
        for (auto& opt : candidate_options) sum += std::get<2>(opt);
        std::vector<double> probs;
        for (auto& opt : candidate_options)
            probs.push_back(sum > 0 ? std::get<2>(opt) / sum : 1.0 / candidate_options.size());
        if (probs.empty()) break;
        std::discrete_distribution<> dist(probs.begin(), probs.end());
        int idx = dist(gen);
        int j = std::get<0>(candidate_options[idx]);
        int d = std::get<1>(candidate_options[idx]);
        // Locker assignment
        int delivery_node = (d == 0) ? j : -1;
        if (d == 1) {
            int assigned = -1;
            if (!instance.customer_preferences.empty() && j-1 < (int)instance.customer_preferences.size()) {
                for (size_t k = 0; k < instance.customer_preferences[j-1].size(); ++k) {
                    if (instance.customer_preferences[j-1][k] == 1) {
                        assigned = instance.lockers[k]->id;
                        break;
                    }
                }
            }
            if (assigned == -1 && !instance.lockers.empty()) assigned = instance.lockers[0]->id;
            delivery_node = assigned;
        }
        perm.push_back(j);
        customer2node[j] = delivery_node;
        prev_index = j;
        unvisited.erase(std::remove(unvisited.begin(), unvisited.end(), j), unvisited.end());
    }
    // Do NOT add depot at the end
    return {perm, customer2node};
}

Solution PACO::solve(const VRPInstance& instance, const PACOParams& params, bool history) {
    // std::cout << "[PACO] Starting solve..." << std::endl;
    int n = instance.num_customers + 1; // including depot
    int m = params.m;
    int p = params.p;
    int t = params.t;
    int I = params.I;
    // std::cout << "[PACO] n=" << n << ", m=" << m << ", p=" << p << ", t=" << t << ", I=" << I << std::endl;
    std::vector<std::vector<std::vector<double>>> tau(n, std::vector<std::vector<double>>(n, std::vector<double>(2, 1.0)));
    std::vector<std::vector<std::vector<int>>> mask(n, std::vector<std::vector<int>>(n, std::vector<int>(2, 1)));
    // std::cout << "[PACO] Building feasibility mask..." << std::endl;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            mask[i][j][0] = (i != j) ? 1 : 0;
            mask[i][j][1] = (i != j) ? 1 : 0;
            if (j > 0 && j <= instance.num_customers) {
                auto c = instance.customers[j-1];
                if (c->customer_type == 2) mask[i][j][0] = 0;
                if (c->customer_type == 1) mask[i][j][1] = 0;
            }
        }
    }
    // std::cout << "[PACO] Applying mask to pheromone matrix..." << std::endl;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int o = 0; o < 2; ++o)
                if (!mask[i][j][o]) tau[i][j][o] = 0.0;
    Solution global_best;
    double global_best_obj = std::numeric_limits<double>::max();
    std::vector<double> convergence_history;
    if (history) convergence_history.push_back(global_best_obj);
    double rho_ini = params.rho;
    int non_improved = 0;
    while (non_improved < I) {
        // std::cout << "[PACO] Iteration " << iter << std::endl;
        std::vector<Solution> all_solutions(m);
        std::vector<double> all_objs(m);
        #pragma omp parallel num_threads(p)
        {
            int tid = omp_get_thread_num();
            int m_p = m / p;
            int start = tid * m_p;
            int end = (tid == p-1) ? m : start + m_p;
            std::random_device rd; std::mt19937 gen(rd() + tid);
            for (int k = start; k < end; ++k) {
                auto [perm, customer2node] = paco_construct_solution(instance, tau, params.alpha, params.beta, gen);
                std::vector<int> best_perm = perm;
                double best_obj = Solver::evaluate(instance, best_perm, customer2node, false).objective_value;
                std::vector<int> curr_perm = best_perm;
                double curr_obj = best_obj;
                std::uniform_real_distribution<> prob(0.0, 1.0);
                for (int ls_iter = 0; ls_iter < params.LS; ++ls_iter) {
                    std::vector<int> neighbor = curr_perm;
                    int n = neighbor.size();
                    if (n > 1) {
                        double r = std::uniform_real_distribution<>(0.0, 1.0)(gen);
                        if (r < 1.0/3) {
                            int i = gen() % n, j = gen() % n;
                            if (i != j) std::swap(neighbor[i], neighbor[j]);
                        } else if (r < 2.0/3) {
                            int i = gen() % n, j = gen() % n;
                            if (i != j) {
                                int val = neighbor[i];
                                neighbor.erase(neighbor.begin() + i);
                                neighbor.insert(neighbor.begin() + j, val);
                            }
                        } else {
                            int i = gen() % n, j = gen() % n;
                            if (i > j) std::swap(i, j);
                            if (i != j) std::reverse(neighbor.begin() + i, neighbor.begin() + j + 1);
                        }
                        double neighbor_obj = Solver::evaluate(instance, neighbor, customer2node, false).objective_value;
                        // Accept only if neighbor is better
                        if (neighbor_obj < curr_obj) {
                            curr_perm = neighbor;
                            curr_obj = neighbor_obj;
                            if (curr_obj < best_obj) {
                                best_perm = curr_perm;
                                best_obj = curr_obj;
                            }
                        }
                    }
                }
                Solution sol = Solver::evaluate(instance, best_perm, customer2node, false);
                all_solutions[k] = sol;
                all_objs[k] = sol.objective_value;
            }
        }
        // std::cout << "[PACO] Finished ant construction." << std::endl;
        std::vector<int> idx(m);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin()+t, idx.end(), [&](int a, int b){ return all_objs[a] < all_objs[b]; });
        // std::cout << "[PACO] Finished sorting top-t ants." << std::endl;
        // Adaptive evaporation rate (new formula)
        auto sigmoid = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
        double rho = (sigmoid((non_improved - 0.25*I) / 20.0)) * (1.0 - rho_ini) + rho_ini;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                for (int o = 0; o < 2; ++o)
                    tau[i][j][o] *= (1.0 - rho);
        // std::cout << "[PACO] Pheromone evaporation done." << std::endl;
        for (int rank = 0; rank < t; ++rank) {
            int k = idx[rank];
            const auto& perm = all_solutions[k].customer_permutation;
            const auto& customer2node = all_solutions[k].customer2node;
            for (size_t s = 1; s < perm.size(); ++s) {
                int prev = perm[s-1];
                int curr = perm[s];
                // Skip depot nodes when updating pheromone and accessing customer2node
                if (prev == 0 || curr == 0) continue;
                auto it = customer2node.find(curr);
                if (it == customer2node.end()) continue; // skip if not found
                int o = (it->second == curr) ? 0 : 1;
                tau[prev][curr][o] += params.Q / all_objs[k];
            }
        }
        // std::cout << "[PACO] Pheromone update done." << std::endl;
        double min_obj = *std::min_element(all_objs.begin(), all_objs.end());
        if (min_obj < global_best_obj) {
            int best_idx = std::min_element(all_objs.begin(), all_objs.end()) - all_objs.begin();
            global_best = all_solutions[best_idx];
            global_best_obj = min_obj;
            non_improved = 0;
        } else {
            non_improved++;
        }
        if (history) convergence_history.push_back(global_best_obj);
        // Best solution so far, Non-improved iterations, and Evaporation rate
        if (history)  std::cout << "[PACO] Iteration " << non_improved << ": Best solution so far: " << global_best_obj
                  << ", Non-improved iterations: " << non_improved
                  << ", Evaporation rate: " << rho << std::endl;
    }
    if (history) {
        std::filesystem::create_directories("../output/experiment");
        std::ofstream csv("../output/experiment/paco.cvr.csv");
        csv << "iter,best_objective\n";
        for (size_t i = 0; i < convergence_history.size(); ++i) {
            csv << i << "," << convergence_history[i] << "\n";
        }
        csv.close();
    }
    // std::cout << "[PACO] Done. Returning best solution." << std::endl;
    return global_best;
}