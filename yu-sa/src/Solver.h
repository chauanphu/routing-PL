#pragma once
#include "VRPInstance.h"
#include <vector>
#include <unordered_map>

struct Solution {
    std::vector<std::vector<int>> routes; // Each route is a sequence of node IDs
    double objective_value;
    std::vector<int> delivery_nodes; // Assigned delivery node for each customer (index = customer-1)
    std::unordered_map<int, int> customer2node; // customer_id -> delivery_node_id
};

class Solver {
public:
    // Construct routes and compute objective given permutation and customer2node mapping
    static Solution evaluate(const VRPInstance& instance,
                            const std::vector<int>& customer_perm,
                            const std::unordered_map<int, int>& customer2node);
};
