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
    int p = 32;     // Number of processors (threads)
    int LS = 10;    // Local search strategy (0: none, 1: 2-opt, 2: 3-opt)
};

class PACO : public Solver {
public:
    static Solution solve(const VRPInstance& instance, const PACOParams& params, bool history = false, int verbose = 0);
    static PACOParams load_params(const std::string& filename);
};
