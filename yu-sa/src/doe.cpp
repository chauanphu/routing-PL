// doe.cpp: Design of Experiments (DoE) for hyperparameter tuning of GA/SA
#include "core/InstanceParser.h"
#include "solvers/Solver.h"
#include "solvers/SA.h"
#include "solvers/GA.h"
#include <iostream>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <chrono>
#include <filesystem>
#include <vector>
#include <string>
#include <cmath>

struct DOEParams {
    std::string solver_name;
    std::string exp_size;
    std::string instance_file;
    std::string param_grid_file; // YAML/CSV file with parameter grid
    std::string output_csv;
    std::string data_dir;
    int num_runs = 1;
};

// Helper: Load parameter grid from YAML file
std::vector<YAML::Node> load_param_grid(const std::string& filename, const std::string& solver_name) {
    std::vector<YAML::Node> grid;
    YAML::Node root = YAML::LoadFile(filename);
    if (!root[solver_name]) {
        std::cerr << "Solver section '" << solver_name << "' not found in grid file." << std::endl;
        exit(1);
    }
    for (const auto& node : root[solver_name]) {
        grid.push_back(node);
    }
    return grid;
}

// Helper: Run experiment for a single parameter set
void run_single_experiment(const DOEParams& params, const YAML::Node& param_set, std::ofstream& ofs) {
    std::vector<std::string> instance_files;
    for (const auto& entry : std::filesystem::directory_iterator(params.data_dir)) {
        if (entry.is_regular_file()) {
            instance_files.push_back(entry.path().string());
        }
    }
    std::sort(instance_files.begin(), instance_files.end());
    for (const auto& file : instance_files) {
        std::vector<double> distances, runtimes;
        int best_num_vehicles = 0;
        double best_distance = 1e12;
        std::cout << "Current instance file: " << file << std::endl;
        for (int run = 0; run < params.num_runs; ++run) {
            auto start = std::chrono::high_resolution_clock::now();
            VRPInstance instance = InstanceParser::parse(file);
            instance.build_distance_matrix();
            Solution sol;
            // std::cout << "- Run: " << run + 1 << std::endl;
            if (params.solver_name == "sa") {
                SAParams sa_params;
                sa_params.max_iter = param_set["max_iter"].as<int>();
                sa_params.T0 = param_set["T0"].as<double>();
                sa_params.Tf = param_set["Tf"].as<double>();
                sa_params.alpha = param_set["alpha"].as<double>();
                sa_params.beta = param_set["beta"].as<double>();
                sa_params.patience = param_set["patience"].as<int>();
                sa_params.p = param_set["p"].as<double>();
                sol = SA::solve(instance, sa_params);
            } else if (params.solver_name == "ga") {
                GAParams ga_params;
                ga_params.population_size = param_set["population_size"].as<int>();
                ga_params.generations = param_set["generations"].as<int>();
                ga_params.crossover_rate = param_set["crossover_rate"].as<double>();
                ga_params.mutation_rate = param_set["mutation_rate"].as<double>();
                ga_params.p = param_set["p"].as<double>();
                sol = GA::solve(instance, ga_params);
            } else {
                std::cerr << "Unknown solver: " << params.solver_name << std::endl;
                exit(1);
            }
            auto end = std::chrono::high_resolution_clock::now();
            double runtime = std::chrono::duration<double>(end - start).count();
            runtimes.push_back(runtime);
            distances.push_back(sol.objective_value);
            if (sol.objective_value < best_distance) {
                best_distance = sol.objective_value;
                best_num_vehicles = sol.routes.size();
            }
        }
        double avg_dist = 0, std_dist = 0, avg_runtime = 0, std_runtime = 0;
        for (double d : distances) avg_dist += d;
        avg_dist /= distances.size();
        for (double d : distances) std_dist += (d - avg_dist) * (d - avg_dist);
        std_dist = sqrt(std_dist / distances.size());
        for (double t : runtimes) avg_runtime += t;
        avg_runtime /= runtimes.size();
        for (double t : runtimes) std_runtime += (t - avg_runtime) * (t - avg_runtime);
        std_runtime = sqrt(std_runtime / runtimes.size());
        std::string instance_name = std::filesystem::path(file).filename().string();
        std::cout << "  Best Distance: " << best_distance << ", Avg Distance: " << avg_dist << ", Std Distance: " << std_dist << std::endl;
        // Output: param values, instance, results
        ofs << params.solver_name;
        for (const auto& it : param_set) {
            ofs << "," << it.first.as<std::string>() << "=" << it.second;
        }
        ofs << "," << instance_name << "," << best_num_vehicles << "," << best_distance << "," << avg_dist << "," << std_dist << "," << avg_runtime << "," << std_runtime << "\n";
        ofs.flush();
    }
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cout << "Usage: " << argv[0] << " <solver> <experiment_size> <param_grid_yaml> <data_dir> [num_runs] [output_csv]" << std::endl;
        std::cout << "  solver: sa | ga" << std::endl;
        std::cout << "  experiment_size: small | medium | large (for reference only)" << std::endl;
        std::cout << "  param_grid_yaml: YAML file with parameter grid (see example)" << std::endl;
        std::cout << "  data_dir: directory with instance files" << std::endl;
        std::cout << "  num_runs: (optional) number of runs per instance/param set (default 1)" << std::endl;
        std::cout << "  output_csv: (optional) output CSV file (default: doe_results.csv)" << std::endl;
        return 1;
    }
    DOEParams params;
    params.solver_name = argv[1];
    params.exp_size = argv[2];
    params.param_grid_file = argv[3];
    params.data_dir = argv[4];
    params.num_runs = (argc >= 6) ? std::stoi(argv[5]) : 1;
    params.output_csv = (argc >= 7) ? argv[6] : "doe_results.csv";
    std::vector<YAML::Node> param_grid = load_param_grid(params.param_grid_file, params.solver_name);
    std::ofstream ofs(params.output_csv);
    ofs << "solver,param_set,instance_name,Num Vehicles,Best Distance,AVG Distance,Std Distance,AVG Runtime (s),Std Runtime (s)\n";
    for (const auto& param_set : param_grid) {
        std::cout << "Running experiment with parameter set:\n" << param_set << std::endl;
        run_single_experiment(params, param_set, ofs);
    }
    ofs.close();
    std::cout << "DoE completed. Results saved to: " << params.output_csv << std::endl;
    return 0;
}
