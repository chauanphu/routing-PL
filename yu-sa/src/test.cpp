#include "core/InstanceParser.h"
#include "solvers/Solver.h"
#include "solvers/SA.h"
#include "solvers/GA.h"
#include "solvers/ACO_TS.h"
#include "solvers/PACO.h"
#include <iostream>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <cmath>

struct ExperimentParams {
    std::string solver_name;
    std::string exp_size;
    std::string instance_file;
    SAParams sa_params;
    GAParams ga_params;
    ACOTSParams aco_params;
    PACOParams paco_params;
    std::string output_csv;
    int num_runs;
    std::string data_dir;
    YAML::Node config;
};

ExperimentParams parse_params(int argc, char* argv[]) {
    ExperimentParams params;
    params.exp_size = "small";
    params.num_runs = 1;
    params.output_csv = "";
    params.data_dir = "";
    std::string params_yaml = "";

    // Parse arguments by key
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--solver" && i + 1 < argc) {
            params.solver_name = argv[++i];
        } else if ((key == "--size" || key == "--exp-size") && i + 1 < argc) {
            params.exp_size = argv[++i];
        } else if (key == "--params" && i + 1 < argc) {
            params_yaml = argv[++i];
        } else if (key == "--instances" && i + 1 < argc) {
            params.data_dir = argv[++i];
        } else if (key == "--num-runs" && i + 1 < argc) {
            params.num_runs = std::stoi(argv[++i]);
        } else if (key == "--output" && i + 1 < argc) {
            params.output_csv = argv[++i];
        } else if (key == "--instance-file" && i + 1 < argc) {
            params.instance_file = argv[++i];
        } else if ((key == "--verbose" || key == "-v") && i + 1 < argc) {
            ++i; // skip verbosity value
        }
    }

    // Check required arguments
    if (params.solver_name.empty()) {
        std::cerr << "Usage: ./test --solver <name> --params <param_file.yaml> [--instances <dir>] [--num-runs <int>] [--output <output_file.csv>] [--size <experiment_size>] [--instance-file <file>] [--verbose <level> | -v <level>]" << std::endl;
        exit(1);
    }
    if (params_yaml.empty()) {
        params_yaml = params.solver_name + ".param.yaml";
    }
    params.config = YAML::LoadFile(params_yaml);
    if (!params.config[params.exp_size]) {
        std::cerr << "Experiment size '" << params.exp_size << "' not found in " << params_yaml << std::endl;
        exit(1);
    }
    // Set from YAML if not set by CLI
    if (params.data_dir.empty() && params.config[params.exp_size]["data_dir"])
        params.data_dir = params.config[params.exp_size]["data_dir"].as<std::string>();
    if (params.output_csv.empty() && params.config[params.exp_size]["output_csv"])
        params.output_csv = params.config[params.exp_size]["output_csv"].as<std::string>();
    if (params.num_runs == 1 && params.config[params.exp_size]["num_runs"])
        params.num_runs = params.config[params.exp_size]["num_runs"].as<int>();

    // Solver-specific params
    if (params.solver_name == "sa") {
        auto sa_params_node = params.config[params.exp_size]["params"];
        params.sa_params.max_iter = sa_params_node["max_iter"].as<int>();
        params.sa_params.T0 = sa_params_node["T0"].as<double>();
        params.sa_params.Tf = sa_params_node["Tf"].as<double>();
        params.sa_params.alpha = sa_params_node["alpha"].as<double>();
        params.sa_params.beta = sa_params_node["beta"].as<double>();
        params.sa_params.patience = sa_params_node["patience"].as<int>();
        params.sa_params.p = sa_params_node["p"].as<double>();
    } else if (params.solver_name == "ga") {
        auto ga_params_node = params.config[params.exp_size]["params"];
        if (!ga_params_node) {
            std::cerr << "ga_params not found in " << params_yaml << std::endl;
            exit(1);
        }
        params.ga_params.population_size = ga_params_node["population_size"].as<int>();
        params.ga_params.generations = ga_params_node["generations"].as<int>();
        params.ga_params.crossover_rate = ga_params_node["crossover_rate"].as<double>();
        params.ga_params.mutation_rate = ga_params_node["mutation_rate"].as<double>();
        params.ga_params.p = ga_params_node["p"].as<double>();
    } else if (params.solver_name == "aco-ts") {
        auto aco_params_node = params.config[params.exp_size]["params"];
        if (!aco_params_node) {
            std::cerr << "aco-ts not found in " << params_yaml << std::endl;
            exit(1);
        }
        params.aco_params.num_ants = aco_params_node["num_ants"].as<int>();
        params.aco_params.num_iterations = aco_params_node["num_iterations"].as<int>();
        params.aco_params.alpha = aco_params_node["alpha"].as<double>();
        params.aco_params.beta = aco_params_node["beta"].as<double>();
        params.aco_params.rho = aco_params_node["evaporation_rate"].as<double>();
        params.aco_params.Q = aco_params_node["Q"].as<double>();
        params.aco_params.p = aco_params_node["p"].as<double>();
    } else if (params.solver_name == "paco") {
        auto paco_node = params.config[params.exp_size]["params"];
        params.paco_params.m = paco_node["m"].as<int>();
        params.paco_params.I = paco_node["I"].as<int>();
        params.paco_params.alpha = paco_node["alpha"].as<double>();
        params.paco_params.beta = paco_node["beta"].as<double>();
        params.paco_params.rho = paco_node["rho"].as<double>();
        params.paco_params.Q = paco_node["Q"].as<double>();
        params.paco_params.t = paco_node["t"].as<int>();
        params.paco_params.p = paco_node["p"].as<int>();
    } else {
        std::cerr << "Unknown solver: " << params.solver_name << std::endl;
        exit(1);
    }
    return params;
}

void print_params(const ExperimentParams& params) {
    if (params.solver_name == "sa") {
        std::cout << "Simulated Annealing Parameters:" << std::endl;
        std::cout << "  max_iter: " << params.sa_params.max_iter << std::endl;
        std::cout << "  T0: " << params.sa_params.T0 << std::endl;
        std::cout << "  Tf: " << params.sa_params.Tf << std::endl;
        std::cout << "  alpha: " << params.sa_params.alpha << std::endl;
        std::cout << "  beta: " << params.sa_params.beta << std::endl;
        std::cout << "  patience: " << params.sa_params.patience << std::endl;
        std::cout << "  p: " << params.sa_params.p << std::endl;
    } else if (params.solver_name == "ga") {
        std::cout << "Genetic Algorithm Parameters:" << std::endl;
        std::cout << "  population_size: " << params.ga_params.population_size << std::endl;
        std::cout << "  generations: " << params.ga_params.generations << std::endl;
        std::cout << "  crossover_rate: " << params.ga_params.crossover_rate << std::endl;
        std::cout << "  mutation_rate: " << params.ga_params.mutation_rate << std::endl;
        std::cout << "  p: " << params.ga_params.p << std::endl;
    } else if (params.solver_name == "aco-ts") {
        std::cout << "ACO-TS Parameters:" << std::endl;
        auto aco_params_node = params.config[params.exp_size]["params"];
        std::cout << "  num_ants: " << aco_params_node["num_ants"].as<int>() << std::endl;
        std::cout << "  num_iterations: " << aco_params_node["num_iterations"].as<int>() << std::endl;
        std::cout << "  alpha: " << aco_params_node["alpha"].as<double>() << std::endl;
        std::cout << "  beta: " << aco_params_node["beta"].as<double>() << std::endl;
        std::cout << "  evaporation_rate: " << aco_params_node["evaporation_rate"].as<double>() << std::endl;
        std::cout << "  Q: " << aco_params_node["Q"].as<double>() << std::endl;
        std::cout << "  p: " << aco_params_node["p"].as<double>() << std::endl;
    } else if (params.solver_name == "paco") {
        std::cout << "3D ACO Parameters:" << std::endl;
        auto paco_node = params.config[params.exp_size]["params"];
        std::cout << "  num_ants: " << paco_node["m"].as<int>() << std::endl;
        std::cout << "  num_iterations: " << paco_node["I"].as<int>() << std::endl;
        std::cout << "  alpha: " << paco_node["alpha"].as<double>() << std::endl;
        std::cout << "  beta: " << paco_node["beta"].as<double>() << std::endl;
        std::cout << "  evaporation_rate: " << paco_node["rho"].as<double>() << std::endl;
        std::cout << "  Q: " << paco_node["Q"].as<double>() << std::endl;
        std::cout << "  t: " << paco_node["t"].as<int>() << std::endl;
        std::cout << "  p: " << paco_node["p"].as<int>() << std::endl;
    }
}

// Helper: get first instance file in a directory
std::string get_first_instance(const std::string& dir) {
    std::vector<std::string> files;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.is_regular_file()) {
            files.push_back(entry.path().string());
        }
    }

    // Sort files based on the format <{C | R | RC}><101 -> 209>_co_<size>.txt
    std::sort(files.begin(), files.end(), [](const std::string& a, const std::string& b) {
        auto extract_key = [](const std::string& filename) -> std::pair<char, int> {
            std::string fname = filename.substr(filename.find_last_of("/\\") + 1);
            char prefix = fname[0];
            size_t num_start = 1;
            // Handle RC prefix
            if (fname[0] == 'R' && fname[1] == 'C') {
                prefix = 'Z'; // RC after R and C
                num_start = 2;
            }
            size_t num_end = fname.find("_co_");
            if (num_end == std::string::npos || num_end <= num_start) return {prefix, 9999};
            std::string num_str = fname.substr(num_start, num_end - num_start);
            int num = 9999;
            try {
                num = std::stoi(num_str);
            } catch (...) {}
            return {prefix, num};
        };

        auto [prefix_a, num_a] = extract_key(a);
        auto [prefix_b, num_b] = extract_key(b);
        if (prefix_a != prefix_b) return prefix_a < prefix_b;
        return num_a < num_b;
    });

    return files.empty() ? "" : files.front();
}

// Helper: Print distances between consecutive nodes in a route for debugging
void print_route_distances(const VRPInstance& instance, const std::vector<int>& route) {
    std::cout << "Route: ";
    for (size_t i = 0; i < route.size(); ++i) {
        std::cout << route[i];
        if (i + 1 < route.size()) std::cout << " -> ";
    }
    std::cout << std::endl;
    double total = 0.0;
    for (size_t i = 0; i + 1 < route.size(); ++i) {
        int from = route[i];
        int to = route[i+1];
        double dist = instance.distance_matrix[from][to];
        std::cout << "  " << from << " -> " << to << ": " << dist << std::endl;
        total += dist;
    }
    std::cout << "Total distance: " << total << std::endl;
}

int main(int argc, char* argv[]) {
    ExperimentParams params = parse_params(argc, argv);
    print_params(params);
    std::string size = params.exp_size;
    if (!params.config[size]) {
        std::cout << "Experiment size not found: " << size << std::endl;
        return 1;
    }
    std::string data_dir = params.config[size]["data_dir"].as<std::string>();
    std::string instance_file;
    if (!params.instance_file.empty()) {
        instance_file = params.instance_file;
    } else {
        instance_file = get_first_instance(data_dir);
    }
    if (instance_file.empty()) {
        std::cout << "No instance found for size: " << size << std::endl;
        return 1;
    }
    std::cout << "\nTesting size: " << size << ", instance: " << instance_file << std::endl;
    VRPInstance instance = InstanceParser::parse(instance_file);
    instance.build_distance_matrix();
    Solution sol;
    auto start = std::chrono::high_resolution_clock::now();
    if (params.solver_name == "sa") {
        sol = SA::solve(instance, params.sa_params, true);
    } else if (params.solver_name == "ga") {
        sol = GA::solve(instance, params.ga_params);
    } else if (params.solver_name == "aco-ts") {
        ACOTSParams aco_params;
        auto aco_node = params.config[size]["params"];
        aco_params.num_ants = aco_node["num_ants"].as<int>();
        aco_params.num_iterations = aco_node["num_iterations"].as<int>();
        aco_params.alpha = aco_node["alpha"].as<double>();
        aco_params.beta = aco_node["beta"].as<double>();
        aco_params.rho = aco_node["evaporation_rate"].as<double>();
        aco_params.Q = aco_node["Q"].as<double>();
        aco_params.p = aco_node["p"].as<double>();
        sol = ACO_TS::solve(instance, aco_params, true);
    } else if (params.solver_name == "paco") {
        auto paco_node = params.config[size]["params"];
        params.paco_params.m = paco_node["m"].as<int>();
        params.paco_params.I = paco_node["I"].as<int>();
        params.paco_params.alpha = paco_node["alpha"].as<double>();
        params.paco_params.beta = paco_node["beta"].as<double>();
        params.paco_params.rho = paco_node["rho"].as<double>();
        params.paco_params.Q = paco_node["Q"].as<double>();
        params.paco_params.t = paco_node["t"].as<int>();
        params.paco_params.p = paco_node["p"].as<int>();
        int verbose = 0;
        for (int i = 1; i < argc; ++i) {
            std::string key = argv[i];
            if ((key == "--verbose" || key == "-v") && i + 1 < argc) {
                verbose = std::stoi(argv[i + 1]);
                break;
            }
        }
        sol = PACO::solve(instance, params.paco_params, true, verbose);
    } else {
        std::cerr << "Unknown solver: " << params.solver_name << std::endl;
        exit(1);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double runtime = std::chrono::duration<double>(end - start).count();
    std::cout << "  Obj = " << sol.objective_value << ", Vehicles = " << sol.routes.size() << ", Time = " << runtime << "s" << std::endl;
    // // Print routes and customer permutation
    // std::cout << "Customer permutation: ";
    // for (size_t i = 0; i < sol.customer_permutation.size(); ++i) {
    //     std::cout << sol.customer_permutation[i];
    //     if (i + 1 < sol.customer_permutation.size()) std::cout << ", ";
    // }
    // // Print distances for debugging
    // std::cout << "\nRoute distances:" << std::endl;
    // for (size_t i = 0; i < sol.routes.size(); ++i) {
    //     std::cout << "Route " << i+1 << ":" << std::endl;
    //     print_route_distances(instance, sol.routes[i]);
    // }
    std::cout << std::endl;
    return 0;
}
