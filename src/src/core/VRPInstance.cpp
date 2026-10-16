#include <cmath>
#include <cstdlib>
#include <cstring>
#include "VRPInstance.h"

VRPInstance::VRPInstance()
    : num_customers(0), num_lockers(0), num_vehicles(0), vehicle_capacity(0) {}

// Helper: compute Euclidean distance between two nodes
static double euclidean(const Node& a, const Node& b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    double d = std::sqrt(dx * dx + dy * dy);
    // Set VRPPL_DISTANCE=trunc1 to truncate distances to 1 decimal (convention of the published VRPPL BKS).
    static const bool trunc1 = [] {
        const char* m = std::getenv("VRPPL_DISTANCE");
        return m && std::strcmp(m, "trunc1") == 0;
    }();
    return trunc1 ? std::trunc(d * 10) / 10 : d;
}

// Build the distance matrix after all nodes are parsed
void VRPInstance::build_distance_matrix() {
    int total_nodes = 1 + num_customers + num_lockers;
    distance_matrix.assign(total_nodes, std::vector<double>(total_nodes, 0.0));
    std::vector<const Node*> nodes;
    nodes.push_back(depot.get());
    for (const auto& c : customers) nodes.push_back(c.get());
    for (const auto& l : lockers) nodes.push_back(l.get());
    for (int i = 0; i < total_nodes; ++i) {
        for (int j = 0; j < total_nodes; ++j) {
            distance_matrix[i][j] = euclidean(*nodes[i], *nodes[j]);
        }
    }
}
