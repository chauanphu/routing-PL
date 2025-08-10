#include "utils.h"

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

} // namespace utils