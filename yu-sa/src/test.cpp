#include "core/InstanceParser.h"
#include "solvers/Solver.h"
#include "solvers/SA.h"
#include "solvers/GA.h"
#include "solvers/ACO_TS.h"
#include "solvers/ThreeDACO.h"
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
    ThreeDACOParams paco_params;
    std::string output_csv;
    int num_runs;
    std::string data_dir;
    YAML::Node config;
};

ExperimentParams parse_params(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <solver> <experiment_size> [instance_file] [params_yaml]" << std::endl;
        std::cout << "  solver: sa (Simulated Annealing) | ga (Genetic Algorithm) | aco-ts (ACO-TS) | paco (3D ACO)" << std::endl;
        std::cout << "  experiment_size: small | medium | large" << std::endl;
        std::cout << "  instance_file: (optional) path to a single instance file" << std::endl;
        std::cout << "  params_yaml: (optional) path to params yaml file, defaults to <solver>.param.yaml" << std::endl;
        exit(1);
    }
    ExperimentParams params;
    params.solver_name = argv[1];
    params.exp_size = argv[2];
    params.instance_file = (argc >= 4) ? argv[3] : "";
    std::string params_yaml = (argc >= 5) ? argv[4] : ("../" + params.solver_name + ".param.yaml");
    params.config = YAML::LoadFile(params_yaml);
    if (!params.config[params.exp_size]) {
        std::cerr << "Experiment size '" << params.exp_size << "' not found in " << params_yaml << std::endl;
        exit(1);
    }
    if (params.solver_name == "sa") {
        auto sa_params_node = params.config[params.exp_size]["sa_params"];
        params.sa_params.max_iter = sa_params_node["max_iter"].as<int>();
        params.sa_params.T0 = sa_params_node["T0"].as<double>();
        params.sa_params.Tf = sa_params_node["Tf"].as<double>();
        params.sa_params.alpha = sa_params_node["alpha"].as<double>();
        params.sa_params.beta = sa_params_node["beta"].as<double>();
        params.sa_params.patience = sa_params_node["patience"].as<int>();
        params.sa_params.p = sa_params_node["p"].as<double>();
    } else if (params.solver_name == "ga") {
        auto ga_params_node = params.config[params.exp_size]["ga_params"];
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
        auto aco_params_node = params.config[params.exp_size]["paco_params"];
        if (!aco_params_node) {
            std::cerr << "paco_params not found in " << params_yaml << std::endl;
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
        auto aco_params_node = params.config[params.exp_size]["paco_params"];
        if (!aco_params_node) {
            std::cerr << "paco_params not found in " << params_yaml << std::endl;
            exit(1);
        }
        params.paco_params.num_ants = aco_params_node["num_ants"].as<int>();
        params.paco_params.num_iterations = aco_params_node["num_iterations"].as<int>();
        params.paco_params.alpha = aco_params_node["alpha"].as<double>();
        params.paco_params.beta = aco_params_node["beta"].as<double>();
        params.paco_params.evaporation_rate = aco_params_node["evaporation_rate"].as<double>();
        params.paco_params.Q = aco_params_node["Q"].as<double>();
    } else {
        std::cerr << "Unknown solver: " << params.solver_name << std::endl;
        exit(1);
    }
    params.output_csv = "../" + params.config[params.exp_size]["output_csv"].as<std::string>();
    params.num_runs = params.config[params.exp_size]["num_runs"].as<int>();
    params.data_dir = params.config[params.exp_size]["data_dir"].as<std::string>();
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
        auto aco_params_node = params.config[params.exp_size]["paco_params"];
        std::cout << "  num_ants: " << aco_params_node["num_ants"].as<int>() << std::endl;
        std::cout << "  num_iterations: " << aco_params_node["num_iterations"].as<int>() << std::endl;
        std::cout << "  alpha: " << aco_params_node["alpha"].as<double>() << std::endl;
        std::cout << "  beta: " << aco_params_node["beta"].as<double>() << std::endl;
        std::cout << "  evaporation_rate: " << aco_params_node["evaporation_rate"].as<double>() << std::endl;
        std::cout << "  Q: " << aco_params_node["Q"].as<double>() << std::endl;
        std::cout << "  p: " << aco_params_node["p"].as<double>() << std::endl;
    } else if (params.solver_name == "paco") {
        std::cout << "3D ACO Parameters:" << std::endl;
        auto aco_params_node = params.config[params.exp_size]["paco_params"];
        std::cout << "  num_ants: " << aco_params_node["num_ants"].as<int>() << std::endl;
        std::cout << "  num_iterations: " << aco_params_node["num_iterations"].as<int>() << std::endl;
        std::cout << "  alpha: " << aco_params_node["alpha"].as<double>() << std::endl;
        std::cout << "  beta: " << aco_params_node["beta"].as<double>() << std::endl;
        std::cout << "  evaporation_rate: " << aco_params_node["evaporation_rate"].as<double>() << std::endl;
        std::cout << "  Q: " << aco_params_node["Q"].as<double>() << std::endl;
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

int main(int argc, char* argv[]) {
    ExperimentParams params = parse_params(argc, argv);
    print_params(params);
    std::string size = params.exp_size;
    if (!params.config[size]) {
        std::cout << "Experiment size not found: " << size << std::endl;
        return 1;
    }
    std::string data_dir = "../" + params.config[size]["data_dir"].as<std::string>();
    std::string instance_file = get_first_instance(data_dir);
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
        sol = SA::solve(instance, params.sa_params);
    } else if (params.solver_name == "ga") {
        sol = GA::solve(instance, params.ga_params);
    } else if (params.solver_name == "aco-ts") {
        ACOTSParams aco_params;
        auto aco_node = params.config[size]["paco_params"];
        aco_params.num_ants = aco_node["num_ants"].as<int>();
        aco_params.num_iterations = aco_node["num_iterations"].as<int>();
        aco_params.alpha = aco_node["alpha"].as<double>();
        aco_params.beta = aco_node["beta"].as<double>();
        aco_params.rho = aco_node["evaporation_rate"].as<double>();
        aco_params.Q = aco_node["Q"].as<double>();
        aco_params.p = aco_node["p"].as<double>();
        sol = ACO_TS::solve(instance, aco_params);
    } else if (params.solver_name == "paco") {
        ThreeDACOParams paco_params;
        auto paco_node = params.config[size]["paco_params"];
        paco_params.num_ants = paco_node["num_ants"].as<int>();
        paco_params.num_iterations = paco_node["num_iterations"].as<int>();
        paco_params.alpha = paco_node["alpha"].as<double>();
        paco_params.beta = paco_node["beta"].as<double>();
        paco_params.evaporation_rate = paco_node["evaporation_rate"].as<double>();
        paco_params.Q = paco_node["Q"].as<double>();
        sol = ThreeDACO::solve(instance, paco_params);
    } else {
        std::cerr << "Unknown solver: " << params.solver_name << std::endl;
        exit(1);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double runtime = std::chrono::duration<double>(end - start).count();
    std::cout << "  Obj = " << sol.objective_value << ", Vehicles = " << sol.routes.size() << ", Time = " << runtime << "s" << std::endl;
    return 0;
}
