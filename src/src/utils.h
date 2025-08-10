#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <unordered_map>
#include <random>
#include <algorithm>

#include "core/VRPInstance.h" // adjust path relative to src/src

namespace utils {

// Initialize delivery node mapping for each customer type.
void initialize_customer2node(const VRPInstance& instance, std::unordered_map<int, int>& customer2node, double p, std::mt19937& gen);

// Construct a random customer permutation with depot delimiters (0) separating routes.
std::vector<int> random_delim_permutation(int n, int m, std::mt19937& gen);

} // namespace utils

#endif // UTILS_H
