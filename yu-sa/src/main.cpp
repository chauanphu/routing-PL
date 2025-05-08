#include "core/InstanceParser.h"
#include "solvers/Solver.h"
#include "solvers/SA.h"
#include "solvers/GA.h"
#include "solvers/ACO_TS.h"
#include "solvers/PACO.h" // Add PACO include
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
    PACOParams paco_params; // Add PACO params
    std::string output_csv;
    int num_runs;
    std::string data_dir;
    YAML::Node config;
};

ExperimentParams parse_params(int argc, char* argv[]) {
    ExperimentParams params;
    params.exp_size = "small";
    params.num_runs = 5;
    params.output_csv = "../output/solutions/output.csv";
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
        } else if ((key == "--verbose" || key == "-v") && i + 1 < argc) {
            // Optionally handle verbosity here if needed
            ++i;
        }
    }

    // Check required arguments
    if (params.solver_name.empty() || params_yaml.empty()) {
        std::cerr << "Usage: ./main --solver <name> --params <param_file.yaml> [--instances <dir>] [--num-runs <int>] [--output <output_file.csv>] [--size <experiment_size>] [--verbose <level> | -v <level>]" << std::endl;
        exit(1);
    }

    params.config = YAML::LoadFile(params_yaml);
    if (!params.config[params.exp_size]) {
        std::cerr << "Experiment size '" << params.exp_size << "' not found in " << params_yaml << std::endl;
        exit(1);
    }

    // If not set by CLI, get from YAML
    if (params.data_dir.empty() && params.config[params.exp_size]["data_dir"])
        params.data_dir = params.config[params.exp_size]["data_dir"].as<std::string>();
    if (params.output_csv == "../output/solutions/output.csv" && params.config[params.exp_size]["output_csv"])
        params.output_csv = params.config[params.exp_size]["output_csv"].as<std::string>();
    if (params.num_runs == 5 && params.config[params.exp_size]["num_runs"])
        params.num_runs = params.config[params.exp_size]["num_runs"].as<int>();

    // ...existing solver param parsing code...
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
        auto aco_params_node = params.config[params.exp_size]["aco-ts"];
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
        params.aco_params.stagnation_limit = aco_params_node["stagnation_limit"] ? aco_params_node["stagnation_limit"].as<int>() : 10;
        params.aco_params.p = aco_params_node["p"] ? aco_params_node["p"].as<double>() : 0.1;
    } else if (params.solver_name == "paco") {
        auto paco_params_node = params.config[params.exp_size]["params"];
        if (!paco_params_node) {
            std::cerr << "params not found in " << params_yaml << std::endl;
            exit(1);
        }
        params.paco_params.m = paco_params_node["m"].as<int>();
        params.paco_params.alpha = paco_params_node["alpha"].as<double>();
        params.paco_params.beta = paco_params_node["beta"].as<double>();
        params.paco_params.rho = paco_params_node["rho"].as<double>();
        params.paco_params.Q = paco_params_node["Q"].as<double>();
        params.paco_params.I = paco_params_node["I"].as<int>();
        params.paco_params.t = paco_params_node["t"].as<int>();
        params.paco_params.p = paco_params_node["p"].as<int>();
        params.paco_params.LS = paco_params_node["LS"].as<int>();
    } else {
        std::cerr << "Unknown solver: " << params.solver_name << std::endl;
        exit(1);
    }
    return params;
}

std::vector<std::string> load_instance_files(const ExperimentParams& params) {
    std::vector<std::string> instance_files;
    std::string data_dir = params.data_dir;
    for (const auto& entry : std::filesystem::directory_iterator(data_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            instance_files.push_back(entry.path().string());
        }
    }
    std::sort(instance_files.begin(), instance_files.end(), [](const std::string& a, const std::string& b) {
        auto get_type = [](const std::string& name) {
            if (name.find("RC") != std::string::npos) return 2;
            if (name.find("R") != std::string::npos) return 1;
            return 0; // C
        };
        auto get_num = [](const std::string& name) {
            size_t start = name.find_first_of("0123456789");
            size_t end = name.find_first_not_of("0123456789", start);
            return std::stoi(name.substr(start, end - start));
        };
        int type_a = get_type(a);
        int type_b = get_type(b);
        if (type_a != type_b) return type_a < type_b;
        int num_a = get_num(a);
        int num_b = get_num(b);
        return num_a < num_b;
    });
    // Print all loaded instance files
    std::cout << "Loaded instance files (" << instance_files.size() << "):" << std::endl;
    for (const auto& f : instance_files) {
        std::cout << "  " << std::filesystem::path(f).filename().string() << std::endl;
    }
    return instance_files;
}

void print_params(const ExperimentParams& params) {
    std::cout << "Loaded parameters for experiment size: " << params.exp_size << std::endl;
    if (params.solver_name == "sa") {
        std::cout << "  max_iter: " << params.sa_params.max_iter << std::endl;
        std::cout << "  T0: " << params.sa_params.T0 << std::endl;
        std::cout << "  Tf: " << params.sa_params.Tf << std::endl;
        std::cout << "  alpha: " << params.sa_params.alpha << std::endl;
        std::cout << "  beta: " << params.sa_params.beta << std::endl;
        std::cout << "  patience: " << params.sa_params.patience << std::endl;
        std::cout << "  p: " << params.sa_params.p << std::endl;
    } else if (params.solver_name == "ga") {
        std::cout << "  population_size: " << params.ga_params.population_size << std::endl;
        std::cout << "  generations: " << params.ga_params.generations << std::endl;
        std::cout << "  crossover_rate: " << params.ga_params.crossover_rate << std::endl;
        std::cout << "  mutation_rate: " << params.ga_params.mutation_rate << std::endl;
        std::cout << "  p: " << params.ga_params.p << std::endl;
    } else if (params.solver_name == "aco-ts") {
        std::cout << "  num_ants: " << params.aco_params.num_ants << std::endl;
        std::cout << "  num_iterations: " << params.aco_params.num_iterations << std::endl;
        std::cout << "  alpha: " << params.aco_params.alpha << std::endl;
        std::cout << "  beta: " << params.aco_params.beta << std::endl;
        std::cout << "  rho: " << params.aco_params.rho << std::endl;
        std::cout << "  Q: " << params.aco_params.Q << std::endl;
        std::cout << "  stagnation_limit: " << params.aco_params.stagnation_limit << std::endl;
        std::cout << "  p: " << params.aco_params.p << std::endl;
    } else if (params.solver_name == "paco") {
        std::cout << "  m: " << params.paco_params.m << std::endl;
        std::cout << "  alpha: " << params.paco_params.alpha << std::endl;
        std::cout << "  beta: " << params.paco_params.beta << std::endl;
        std::cout << "  rho: " << params.paco_params.rho << std::endl;
        std::cout << "  Q: " << params.paco_params.Q << std::endl;
        std::cout << "  I: " << params.paco_params.I << std::endl;
        std::cout << "  t: " << params.paco_params.t << std::endl;
        std::cout << "  p: " << params.paco_params.p << std::endl;
        std::cout << "  LS: " << params.paco_params.LS << std::endl;
    }
    std::cout << "  num_runs: " << params.num_runs << std::endl;
    std::cout << "  output_csv: " << params.output_csv << std::endl;
    std::cout << "  data_dir: " << params.data_dir << std::endl;
}

void run_experiment(const ExperimentParams& params, const std::vector<std::string>& instance_files) {
    std::ofstream ofs(params.output_csv);
    ofs << "instance_name,Num Vehicles,Best Distance,AVG Distance,Std Distance,AVG Runtime (s),Std Runtime (s)\n";
    for (const auto& file : instance_files) {
        std::cout << "\nProcessing instance: " << file << std::endl;
        std::vector<double> distances, runtimes;
        int best_num_vehicles = 0;
        double best_distance = 1e12;
        for (int run = 0; run < params.num_runs; ++run) {
            auto start = std::chrono::high_resolution_clock::now();
            VRPInstance instance = InstanceParser::parse(file);
            instance.build_distance_matrix();
            Solution sol;
            if (params.solver_name == "sa") {
                sol = SA::solve(instance, params.sa_params, false);
            } else if (params.solver_name == "ga") {
                sol = GA::solve(instance, params.ga_params);
            } else if (params.solver_name == "aco-ts") {
                sol = ACO_TS::solve(instance, params.aco_params);
            } else if (params.solver_name == "paco") {
                sol = PACO::solve(instance, params.paco_params);
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
            std::cout << "  Run " << (run+1) << ": Obj = " << sol.objective_value << ", Vehicles = " << sol.routes.size() << ", Time = " << runtime << "s" << std::endl;
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
        ofs << instance_name << "," << best_num_vehicles << "," << best_distance << "," << avg_dist << "," << std_dist << "," << avg_runtime << "," << std_runtime << "\n";
        ofs.flush();
        std::cout << "Finished: " << instance_name << " Best: " << best_distance << " Avg: " << avg_dist << std::endl;
        std::cout << "Results saved to: " << params.output_csv << std::endl;
    }
    ofs.close();
}

int main(int argc, char* argv[]) {
    ExperimentParams params = parse_params(argc, argv);
    std::vector<std::string> instance_files = load_instance_files(params);
    print_params(params);
    run_experiment(params, instance_files);
    return 0;
}
