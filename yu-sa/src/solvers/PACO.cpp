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
#include <set>

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

// --- GA operators for elitist ants ---
// Order Crossover (OX) for permutations
static std::vector<int> order_crossover(const std::vector<int>& parent1, const std::vector<int>& parent2, std::mt19937& gen) {
    int n = parent1.size();
    std::uniform_int_distribution<> dist(0, n - 1);
    int a = dist(gen), b = dist(gen);
    if (a > b) std::swap(a, b);
    std::vector<int> child(n, -1);
    std::set<int> in_child;
    // Copy slice from parent1
    for (int i = a; i <= b; ++i) {
        child[i] = parent1[i];
        in_child.insert(parent1[i]);
    }
    // Fill from parent2
    int j = (b + 1) % n;
    for (int k = 0; k < n; ++k) {
        int idx = (b + 1 + k) % n;
        if (in_child.count(parent2[idx]) == 0) {
            child[j] = parent2[idx];
            in_child.insert(parent2[idx]);
            j = (j + 1) % n;
        }
    }
    return child;
}

// Swap mutation
static void mutate(std::vector<int>& perm, std::mt19937& gen, double mutation_rate = 0.2) {
    std::uniform_real_distribution<> prob(0.0, 1.0);
    if (prob(gen) < mutation_rate && perm.size() > 1) {
        std::uniform_int_distribution<> dist(0, perm.size() - 1);
        int i = dist(gen), j = dist(gen);
        if (i != j) std::swap(perm[i], perm[j]);
    }
}

Solution PACO::solve(const VRPInstance& instance, const PACOParams& params, bool history, int verbose) {
    if (verbose >= 1) std::cout << "[PACO] Starting solve..." << std::endl;
    int n_nodes = instance.num_customers + 1; // including depot, for pheromone matrix indexing
    int m = params.m; // total number of ants
    int p = params.p; // number of parallel threads
    int t = params.t; // number of elite solutions for pheromone update
    int I = params.I; // max non-improved iterations

    if (verbose >= 2) std::cout << "[PACO] n_nodes=" << n_nodes << ", m=" << m << ", p=" << p << ", t=" << t << ", I=" << I << std::endl;
    std::vector<std::vector<std::vector<double>>> tau(n_nodes, std::vector<std::vector<double>>(n_nodes, std::vector<double>(2, 1.0)));
    std::vector<std::vector<std::vector<int>>> mask(n_nodes, std::vector<std::vector<int>>(n_nodes, std::vector<int>(2, 1)));

    if (verbose >= 2) std::cout << "[PACO] Building feasibility mask..." << std::endl;
    for (int i = 0; i < n_nodes; ++i) {
        for (int j = 0; j < n_nodes; ++j) {
            mask[i][j][0] = (i != j) ? 1 : 0; // Direct delivery
            mask[i][j][1] = (i != j) ? 1 : 0; // Locker delivery
            if (j > 0 && j <= instance.num_customers) { // j is a customer node (1 to num_customers)
                auto c = instance.customers[j-1]; // Assuming customers vector is 0-indexed
                if (c->customer_type == 2) mask[i][j][0] = 0; // Customer type 2 cannot be direct
                if (c->customer_type == 1) mask[i][j][1] = 0; // Customer type 1 cannot be locker
            }
        }
    }
    if (verbose >= 2) std::cout << "[PACO] Applying mask to pheromone matrix..." << std::endl;
    for (int i = 0; i < n_nodes; ++i)
        for (int j = 0; j < n_nodes; ++j)
            for (int o = 0; o < 2; ++o)
                if (!mask[i][j][o]) tau[i][j][o] = 0.0;

    Solution global_best;
    double global_best_obj = std::numeric_limits<double>::max();
    std::vector<double> convergence_history;
    if (history) convergence_history.push_back(global_best_obj);

    double rho_ini = params.rho;
    int non_improved = 0;
    int iter = 0;
    double rho = rho_ini;
    auto sigmoid = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };

    std::random_device rd_global_seed; // For seeding thread-local RNGs

    while (non_improved < I) {
        if (verbose >= 1) std::cout << "[PACO] Iteration " << iter << std::endl;
        
        std::vector<Solution> iteration_final_solutions; // Collects solutions from all threads after their local GA

        #pragma omp parallel num_threads(p)
        {
            int tid = omp_get_thread_num();
            // Calculate ants per thread
            int ants_per_thread = m / p;
            int remainder_ants = m % p;
            int start_ant_idx_global = tid * ants_per_thread + std::min(tid, remainder_ants);
            int num_ants_this_thread = ants_per_thread * (non_improved + 1) + (tid < remainder_ants ? 1 : 0);

            std::mt19937 thread_rng(rd_global_seed() + tid + iter); // Thread-local RNG

            std::vector<Solution> thread_initial_ant_solutions(num_ants_this_thread);
            std::vector<double> thread_initial_ant_objs(num_ants_this_thread);

            if (num_ants_this_thread > 0) {
                // 1. Ant construction and Local Search for this thread's ants
                for (int k_local = 0; k_local < num_ants_this_thread; ++k_local) {
                    auto [perm, customer2node_map] = paco_construct_solution(instance, tau, params.alpha, params.beta, thread_rng);
                    
                    std::vector<int> best_perm_ls = perm;
                    Solution initial_eval_sol = Solver::evaluate(instance, best_perm_ls, customer2node_map, false);
                    double best_obj_ls = initial_eval_sol.objective_value;
                    
                    std::vector<int> curr_perm_ls = best_perm_ls;
                    double curr_obj_ls = best_obj_ls;
                    
                    for (int ls_iter_count = 0; ls_iter_count < params.LS; ++ls_iter_count) {
                        std::vector<int> neighbor_perm = curr_perm_ls;
                        int n_ls = neighbor_perm.size();
                        if (n_ls > 1) {
                            std::uniform_real_distribution<> op_dist(0.0, 1.0);
                            double r_ls = op_dist(thread_rng);
                            std::uniform_int_distribution<> idx_dist(0, n_ls - 1);
                            int i_ls = idx_dist(thread_rng);
                            int j_ls = idx_dist(thread_rng);

                            if (r_ls < 1.0/3.0) { // Swap
                                if (i_ls != j_ls) std::swap(neighbor_perm[i_ls], neighbor_perm[j_ls]);
                            } else if (r_ls < 2.0/3.0) { // Insert
                                if (i_ls != j_ls && !neighbor_perm.empty()) {
                                    int val_to_move = neighbor_perm[i_ls];
                                    neighbor_perm.erase(neighbor_perm.begin() + i_ls);
                                    // Adjust j_ls to be a valid index for insertion
                                    int insert_at_idx = j_ls;
                                    if (neighbor_perm.empty()) {
                                        insert_at_idx = 0;
                                    } else if (insert_at_idx > neighbor_perm.size()) {
                                        insert_at_idx = neighbor_perm.size(); 
                                    }

                                    if (insert_at_idx == 0 || neighbor_perm.empty()) {
                                       neighbor_perm.insert(neighbor_perm.begin(), val_to_move);
                                    } else if (insert_at_idx >= neighbor_perm.size()) {
                                       neighbor_perm.push_back(val_to_move);
                                    }
                                    else {
                                       neighbor_perm.insert(neighbor_perm.begin() + insert_at_idx, val_to_move);
                                    }
                                }
                            } else { // Reverse (2-opt style)
                                if (i_ls > j_ls) std::swap(i_ls, j_ls);
                                if (i_ls != j_ls && i_ls < neighbor_perm.size() && j_ls < neighbor_perm.size() && (j_ls + 1) <= neighbor_perm.size()) {
                                   std::reverse(neighbor_perm.begin() + i_ls, neighbor_perm.begin() + j_ls + 1);
                                }
                            }
                            Solution neighbor_sol_eval = Solver::evaluate(instance, neighbor_perm, customer2node_map, false);
                            double neighbor_obj_ls = neighbor_sol_eval.objective_value;
                            if (neighbor_obj_ls < curr_obj_ls) {
                                curr_perm_ls = neighbor_perm;
                                curr_obj_ls = neighbor_obj_ls;
                                if (curr_obj_ls < best_obj_ls) {
                                    best_perm_ls = curr_perm_ls;
                                    best_obj_ls = curr_obj_ls;
                                }
                            }
                        }
                    }
                    Solution final_ant_sol = Solver::evaluate(instance, best_perm_ls, customer2node_map, false);
                    thread_initial_ant_solutions[k_local] = final_ant_sol;
                    thread_initial_ant_objs[k_local] = final_ant_sol.objective_value;
                }

                // 2. Initialize GA population for this thread from its best ants
                int ga_population_size_for_thread = std::max(1, t / p); 
                if (t < p && t > 0) ga_population_size_for_thread = 1; 
                if (t == 0) ga_population_size_for_thread = 0; 
                ga_population_size_for_thread = std::min(ga_population_size_for_thread, num_ants_this_thread);

                std::vector<Solution> current_ga_pop_thread; // This thread's GA population
                if (num_ants_this_thread > 0 && ga_population_size_for_thread > 0) {
                    std::vector<int> ant_indices(num_ants_this_thread);
                    std::iota(ant_indices.begin(), ant_indices.end(), 0);
                    std::partial_sort(ant_indices.begin(), ant_indices.begin() + ga_population_size_for_thread, ant_indices.end(),
                                      [&](int a, int b){ return thread_initial_ant_objs[a] < thread_initial_ant_objs[b]; });
                    for (int i = 0; i < ga_population_size_for_thread; ++i) {
                        current_ga_pop_thread.push_back(thread_initial_ant_solutions[ant_indices[i]]);
                    }
                }

                // 3. Iterative GA with Tournament Selection for this thread
                std::vector<Solution> solutions_from_thread_ga; // Final output of this thread's GA

                if (!current_ga_pop_thread.empty()) {
                    const int TOURNAMENT_K_VALUE = 2; // Tournament size

                    // Lambda for tournament selection
                    auto select_parent_tournament = 
                        [&](const std::vector<Solution>& pop, int k_tournament, std::mt19937& rng) -> int {
                        if (pop.empty()) return -1;
                        if (pop.size() == 1) return 0;
                        int effective_k = std::min(k_tournament, (int)pop.size());
                        if (effective_k <= 0) return 0; 

                        int best_idx_in_tournament = -1;
                        double best_obj_in_tournament = std::numeric_limits<double>::max();
                        std::uniform_int_distribution<> dist(0, pop.size() - 1);

                        for (int i = 0; i < effective_k; ++i) {
                            int current_participant_idx = dist(rng);
                            if (pop[current_participant_idx].objective_value < best_obj_in_tournament) {
                                best_obj_in_tournament = pop[current_participant_idx].objective_value;
                                best_idx_in_tournament = current_participant_idx;
                            }
                        }
                        return (best_idx_in_tournament != -1) ? best_idx_in_tournament : dist(rng); // Fallback
                    };

                    // Determine GA generations for this thread
                    int ga_gens_base = 2;         // Base generations
                    int ga_gens_bonus_max = 5;  // Max bonus generations
                    int num_ga_generations = ga_gens_base;
                    if (I > 0) { // I is max non-improved iterations for main ACO loop
                        double non_improve_ratio = static_cast<double>(non_improved) / I;
                        num_ga_generations += static_cast<int>(std::round(ga_gens_bonus_max * non_improve_ratio));
                    }
                    num_ga_generations = std::min(num_ga_generations, ga_gens_base + ga_gens_bonus_max);
                    num_ga_generations = std::max(1, num_ga_generations); // At least 1 generation

                    for (int gen = 0; gen < num_ga_generations; ++gen) {
                        if (current_ga_pop_thread.empty() || ga_population_size_for_thread == 0) break;

                        std::vector<Solution> offspring_population;
                        offspring_population.reserve(ga_population_size_for_thread);

                        for (int i = 0; i < ga_population_size_for_thread; ++i) {
                            if (current_ga_pop_thread.empty()) break;
                            int p1_local_idx = select_parent_tournament(current_ga_pop_thread, TOURNAMENT_K_VALUE, thread_rng);
                            int p2_local_idx = select_parent_tournament(current_ga_pop_thread, TOURNAMENT_K_VALUE, thread_rng);

                            if (p1_local_idx == -1 || p2_local_idx == -1) continue;

                            const auto& parent1_sol = current_ga_pop_thread[p1_local_idx];
                            const auto& parent2_sol = current_ga_pop_thread[p2_local_idx];
                            
                            auto child_permutation = order_crossover(parent1_sol.customer_permutation, parent2_sol.customer_permutation, thread_rng);
                            mutate(child_permutation, thread_rng, 0.2); // Hardcoded mutation rate
                            
                            auto child_customer2node = parent1_sol.customer2node; // Inherit map
                            Solution child_solution = Solver::evaluate(instance, child_permutation, child_customer2node, false);
                            offspring_population.push_back(child_solution);
                        }
                        // Generational replacement with elitism: combine parents and offspring, sort, then truncate
                        current_ga_pop_thread.insert(current_ga_pop_thread.end(), offspring_population.begin(), offspring_population.end());
                        std::sort(current_ga_pop_thread.begin(), current_ga_pop_thread.end(), 
                                  [](const Solution& a, const Solution& b) {
                                      return a.objective_value < b.objective_value;
                                  });
                        if (current_ga_pop_thread.size() > ga_population_size_for_thread) {
                            current_ga_pop_thread.resize(ga_population_size_for_thread);
                        }
                    }
                    solutions_from_thread_ga = current_ga_pop_thread; // Final population after GA
                } else {
                    solutions_from_thread_ga.clear(); // No initial GA population
                }
                
                // 4. Determine this thread's contribution to the global solution pool
                std::vector<Solution> thread_contribution_to_global = solutions_from_thread_ga;
                // Fallback: if GA produced nothing, contribute best initial ants
                if (thread_contribution_to_global.empty() && !thread_initial_ant_solutions.empty()) {
                    std::sort(thread_initial_ant_solutions.begin(), thread_initial_ant_solutions.end(),
                              [](const Solution& a, const Solution& b) { return a.objective_value < b.objective_value; });
                    int num_to_contribute_fallback = std::min((int)thread_initial_ant_solutions.size(), std::max(1, ga_population_size_for_thread));
                    if (num_to_contribute_fallback > 0 && num_to_contribute_fallback <= thread_initial_ant_solutions.size()) {
                         thread_contribution_to_global.assign(thread_initial_ant_solutions.begin(), 
                                                              thread_initial_ant_solutions.begin() + num_to_contribute_fallback);
                    }
                }

                // 5. Add this thread's final solutions to the global pool for this iteration
                #pragma omp critical
                {
                    iteration_final_solutions.insert(iteration_final_solutions.end(),
                                                     thread_contribution_to_global.begin(), 
                                                     thread_contribution_to_global.end());
                }
            } // if (num_ants_this_thread > 0)
        } // End of #pragma omp parallel
        if (verbose >= 2) std::cout << "[PACO] Finished parallel ant construction and GA." << std::endl;

        // Sort all collected candidate solutions from all threads
        std::sort(iteration_final_solutions.begin(), iteration_final_solutions.end(), 
                  [](const Solution& a, const Solution& b) {
            return a.objective_value < b.objective_value;
        });
        if (verbose >= 2 && !iteration_final_solutions.empty()) {
            std::cout << "[PACO] Total solutions after parallel GA: " << iteration_final_solutions.size() 
                      << ", Best obj this iter: " << iteration_final_solutions[0].objective_value << std::endl;
        } else if (verbose >=2 && iteration_final_solutions.empty()) {
            std::cout << "[PACO] No solutions generated this iteration." << std::endl;
        }

        // Pheromone evaporation
        rho = rho_ini + (0.1 - rho_ini) * sigmoid(8 * (static_cast<double>(non_improved) / (std::max(1,I) / 2.0) - 0.5)); // Avoid division by zero if I=0
        for (int i = 0; i < n_nodes; ++i)
            for (int j = 0; j < n_nodes; ++j)
                for (int o = 0; o < 2; ++o)
                    tau[i][j][o] *= (1.0 - rho);
        if (verbose >= 2) std::cout << "[PACO] Pheromone evaporation done. Rho = " << rho << std::endl;

        // Pheromone update using the top 't' solutions from iteration_final_solutions
        int num_solutions_for_pheromone_update = std::min((int)iteration_final_solutions.size(), t);
        for (int rank = 0; rank < num_solutions_for_pheromone_update; ++rank) {
            const auto& sol_for_update = iteration_final_solutions[rank];
            const auto& perm = sol_for_update.customer_permutation; // Assumed to be customer IDs
            const auto& customer2node = sol_for_update.customer2node;
            double obj = sol_for_update.objective_value;

            if (!perm.empty()) {
                int first_cust = perm[0]; // Assuming perm holds customer IDs (1..N)
                auto it_map_first = customer2node.find(first_cust);
                if (it_map_first != customer2node.end() && first_cust > 0 && first_cust < n_nodes) {
                    int delivery_type_first = (it_map_first->second == first_cust) ? 0 : 1;
                    if (mask[0][first_cust][delivery_type_first]) { // Check mask
                        tau[0][first_cust][delivery_type_first] += params.Q / obj;
                    }
                }
            }
            for (size_t s = 0; s + 1 < perm.size(); ++s) {
                int u = perm[s];
                int v = perm[s+1];
                auto it_map_v = customer2node.find(v);
                if (it_map_v != customer2node.end() && u > 0 && u < n_nodes && v > 0 && v < n_nodes) {
                    int delivery_type_v = (it_map_v->second == v) ? 0 : 1;
                     if (mask[u][v][delivery_type_v]) { // Check mask
                        tau[u][v][delivery_type_v] += params.Q / obj;
                    }
                }
            }
            if (!perm.empty()) {
                int last_cust = perm.back();
                if (last_cust > 0 && last_cust < n_nodes) {
                    if (mask[last_cust][0][0]) { // Assuming direct return to depot (type 0)
                        tau[last_cust][0][0] += params.Q / obj;
                    }
                }
            }
        }

        // Update global best solution
        if (!iteration_final_solutions.empty()) {
            if (iteration_final_solutions[0].objective_value < global_best_obj) {
                global_best = iteration_final_solutions[0];
                global_best_obj = iteration_final_solutions[0].objective_value;
                non_improved = 0;
                if (verbose >= 1) std::cout << "[PACO] New global best: " << global_best_obj << std::endl;
            } else {
                non_improved++;
            }
        } else {
             non_improved++; // No solutions, count as non-improvement
        }

        if (history) convergence_history.push_back(global_best_obj);
        if (verbose >= 1) std::cout << "[PACO] Iteration " << iter
            << ": Best solution so far: " << global_best_obj
            << ", Non-improved iterations: " << non_improved
            << ", Evaporation rate: " << rho << std::endl;
        iter++;
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
    if (verbose >= 1) std::cout << "[PACO] Done. Returning best solution." << std::endl;
    return global_best;
}