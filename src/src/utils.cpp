#include "utils.h"
#include <unordered_set>

namespace utils {

// Helper: initialize delivery node mapping for each customer
void initialize_customer2node(const VRPInstance& instance, std::unordered_map<int, int>& customer2node, double p, std::mt19937& gen) {
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
std::vector<int> random_delim_permutation(int n, int m, std::mt19937& gen) {
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

void initialize_solution(
    const VRPInstance& instance,
    std::vector<int>& customer_perm,
    std::unordered_map<int, int>& customer2node,
    double p,
    bool delimiters) {
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
    // Step 2: nearest neighbor assignment for permutation; optionally insert delimiters
    std::vector<bool> assigned(n, false);
    if (delimiters) {
        int current_node = 0; // depot
        int m = instance.num_vehicles > 0 ? instance.num_vehicles : 1;
        int per_route = (n + m - 1) / m;
        int assigned_count = 0;
        for (int v = 0; v < m; ++v) {
            customer_perm.push_back(0); // depot delimiter at start of each route
            int route_count = 0;
            // Nearest neighbor insertion
            while (route_count < per_route && assigned_count < n) {
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
                ++route_count;
                ++assigned_count;
            }
            // Nearest neighbor insertion
            current_node = 0; // reset to depot for next route
        }
        customer_perm.push_back(0); // depot delimiter at end
    } else {
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
}

// Random initialization with constraint-aware delimiter insertion.
void random_init(const VRPInstance& instance, std::vector<int>& customer_perm, std::unordered_map<int,int>& customer2node, double p) {
    int n = instance.num_customers;
    int m = instance.num_vehicles > 0 ? instance.num_vehicles : 1;
    customer_perm.clear();
    customer2node.clear();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> prob(0.0, 1.0);
    // Step 1: randomly assign delivery nodes similar to initialize_solution but random locker choice when allowed
    std::vector<int> assigned_delivery_node(n, -1);
    for (int i = 0; i < n; ++i) {
        auto c = instance.customers[i];
        if (c->customer_type == 1) {
            assigned_delivery_node[i] = c->id;
        } else if (c->customer_type == 2) {
            // pick a preferred locker randomly among feasible, else first locker, else itself
            std::vector<int> prefs;
            if (!instance.customer_preferences.empty() && i < (int)instance.customer_preferences.size()) {
                for (size_t j = 0; j < instance.customer_preferences[i].size(); ++j) {
                    if (instance.customer_preferences[i][j] == 1) prefs.push_back(instance.lockers[j]->id);
                }
            }
            if (!prefs.empty()) {
                std::uniform_int_distribution<> pick(0, (int)prefs.size()-1);
                assigned_delivery_node[i] = prefs[pick(gen)];
            } else if (!instance.lockers.empty()) {
                std::uniform_int_distribution<> pick(0, (int)instance.lockers.size()-1);
                assigned_delivery_node[i] = instance.lockers[pick(gen)]->id;
            } else {
                assigned_delivery_node[i] = c->id;
            }
        } else if (c->customer_type == 3) {
            double r = prob(gen);
            if (r < p) {
                assigned_delivery_node[i] = c->id; // home
            } else {
                std::vector<int> prefs;
                if (!instance.customer_preferences.empty() && i < (int)instance.customer_preferences.size()) {
                    for (size_t j = 0; j < instance.customer_preferences[i].size(); ++j) {
                        if (instance.customer_preferences[i][j] == 1) prefs.push_back(instance.lockers[j]->id);
                    }
                }
                if (!prefs.empty()) {
                    std::uniform_int_distribution<> pick(0, (int)prefs.size()-1);
                    assigned_delivery_node[i] = prefs[pick(gen)];
                } else if (!instance.lockers.empty()) {
                    std::uniform_int_distribution<> pick(0, (int)instance.lockers.size()-1);
                    assigned_delivery_node[i] = instance.lockers[pick(gen)]->id;
                } else {
                    assigned_delivery_node[i] = c->id;
                }
            }
        }
        customer2node[c->id] = assigned_delivery_node[i];
    }
    // Step 2: nearest neighbor heuristic with constraint-aware insertion and delimiters
    const int depot_id = 0;
    int cap = instance.vehicle_capacity;
    int vehicle = 0;
    std::vector<char> remaining(n, 1); // 1 if not yet inserted
    int remaining_count = n;
    while (remaining_count > 0 && vehicle < m) {
        customer_perm.push_back(0); // start route
        int load = 0;
        int time = 0;
        int curr_node = depot_id;
        std::unordered_set<int> visited_lockers;
        int last_node = depot_id;
        bool progress = true;
        while (progress) {
            progress = false;
            double best_dist = std::numeric_limits<double>::max();
            int best_idx = -1;
            int best_delivery_node = -1;
            int best_demand = 0; int best_early=0; int best_late=0; bool best_is_locker=false;
            for (int i = 0; i < n; ++i) {
                if (!remaining[i]) continue;
                int cust_id = i + 1;
                int delivery_node = customer2node[cust_id];
                bool is_locker = (delivery_node > instance.num_customers);
                int demand=0, early=0, late=std::numeric_limits<int>::max();
                if (delivery_node <= instance.num_customers) {
                    auto c = instance.customers[delivery_node-1];
                    demand = c->demand; early = c->early_time; late = c->late_time;
                } else {
                    auto l = instance.lockers[delivery_node-1-instance.num_customers];
                    auto c = instance.customers[cust_id-1];
                    demand = c->demand; early = l->early_time; late = l->late_time;
                }
                int arr_time = time + (int)instance.distance_matrix[curr_node][delivery_node];
                bool duplicate_locker = false;
                if (is_locker && visited_lockers.count(delivery_node) && delivery_node != last_node) duplicate_locker = true;
                if (load + demand > cap || arr_time > late || duplicate_locker) continue; // infeasible
                double dist = instance.distance_matrix[curr_node][delivery_node];
                if (dist < best_dist) {
                    best_dist = dist;
                    best_idx = i;
                    best_delivery_node = delivery_node;
                    best_demand = demand; best_early = early; best_late = late; best_is_locker = is_locker;
                }
            }
            if (best_idx != -1) {
                // insert chosen customer
                int cust_id = best_idx + 1;
                remaining[best_idx] = 0; --remaining_count;
                customer_perm.push_back(cust_id);
                int arr_time = time + (int)instance.distance_matrix[curr_node][best_delivery_node];
                time = std::max(arr_time, best_early);
                load += best_demand;
                curr_node = best_delivery_node;
                if (best_is_locker) visited_lockers.insert(best_delivery_node);
                last_node = best_delivery_node;
                progress = true;
            }
        }
        // close route
        customer_perm.push_back(0);
        ++vehicle;
    }
    // If vehicles exhausted but customers remain, they are left unserved (could be penalized by evaluation)
}
} // namespace utils