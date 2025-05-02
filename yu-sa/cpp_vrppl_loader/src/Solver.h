#pragma once
#include "VRPInstance.h"
#include <vector>
#include <unordered_map>

struct Solution {
    std::vector<std::vector<int>> routes; // Each route is a sequence of node IDs
    double objective_value;
    std::vector<int> delivery_nodes; // Assigned delivery node for each customer (index = customer-1)
    std::unordered_map<int, int> customer_to_delivery_node; // customer_id -> delivery_node_id

    // Given a customer index in the permutation, return the assigned delivery node
    int get_delivery_node(int perm_index) const {
        if (perm_index < 0 || perm_index >= (int)routes[0].size())
            return -1;
        int cust_id = routes[0][perm_index]; // 1-based customer ID
        auto it = customer_to_delivery_node.find(cust_id);
        if (it != customer_to_delivery_node.end())
            return it->second;
        return -1;
    }
};

class Solver {
public:
    static Solution solve(const VRPInstance& instance);
};
