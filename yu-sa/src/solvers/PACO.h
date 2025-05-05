#pragma once
#include "Solver.h"
#include <vector>
#include <tuple>
#include <string>

// PACO hyperparameters
struct PACOParams {
    int m;      // Number of ants
    double alpha;
    double beta;
    double rho;
    double Q;
    int I;     // Number of iterations
    int t;     // Top ants for elitist update
    int p;     // Number of processors (threads)
};

class PACO : public Solver {
public:
    static Solution solve(const VRPInstance& instance, const PACOParams& params);
    static PACOParams load_params(const std::string& filename);
};
