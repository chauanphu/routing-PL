// bao.cpp: Bayesian Optimization for Solver Hyperparameter Tuning
// Usage: ./bao --solver <name> --params <param_file.yaml> --instances <dir> [--max-evals <int>] [--runs-per-instance <int>] [--output <output_file.yaml>]

#include "core/InstanceParser.h"
#include "solvers/Solver.h"
#include "solvers/SA.h"
#include "solvers/GA.h"
#include "solvers/ACO_TS.h"
#include "solvers/PACO.h"
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <variant>
#include <random>
#include <chrono>
#include <numeric>
#include <limits>
#include <filesystem>
#include <fstream>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <stdexcept>
#include <cstdlib>
#include <nlohmann/json.hpp> // For JSON serialization (add to yu-sa/CMakeLists.txt if not present)

// --- Type Definitions ---
using ParamValue = std::variant<int, double>;
using Configuration = std::map<std::string, ParamValue>;

struct ParamDef {
    std::string type; // "int" or "float"
    ParamValue min_val;
    ParamValue max_val;
    ParamValue default_val;
    ParamValue step;
    bool tune = true;
};
using ParamSpace = std::map<std::string, ParamDef>;

// --- Helper Functions ---
ParamSpace load_param_space(const std::string& filename) {
    ParamSpace space;
    YAML::Node config = YAML::LoadFile(filename);
    if (!config["parameters"]) {
        throw std::runtime_error("YAML file missing 'parameters' key: " + filename);
    }
    for (const auto& param_node : config["parameters"]) {
        std::string name = param_node.first.as<std::string>();
        ParamDef def;
        def.type = param_node.second["type"].as<std::string>();
        def.tune = !param_node.second["tune"] || param_node.second["tune"].as<bool>();
        if (def.type == "int") {
            def.min_val = param_node.second["range"].size() > 0 ? param_node.second["range"][0].as<int>() : param_node.second["default"].as<int>();
            def.max_val = param_node.second["range"].size() > 1 ? param_node.second["range"][1].as<int>() : param_node.second["default"].as<int>();
            def.default_val = param_node.second["default"].as<int>();
            def.step = param_node.second["step"].as<int>();
        } else if (def.type == "float") {
            def.min_val = param_node.second["range"].size() > 0 ? param_node.second["range"][0].as<double>() : param_node.second["default"].as<double>();
            def.max_val = param_node.second["range"].size() > 1 ? param_node.second["range"][1].as<double>() : param_node.second["default"].as<double>();
            def.default_val = param_node.second["default"].as<double>();
            def.step = param_node.second["step"].as<double>();
        } else {
            throw std::runtime_error("Unsupported parameter type: " + def.type + " for parameter " + name);
        }
        space[name] = def;
    }
    std::cout << "Parameter space loaded successfully." << std::endl;
    return space;
}

Configuration get_default_configuration(const ParamSpace& space) {
    Configuration config;
    for (const auto& pair : space) {
        config[pair.first] = pair.second.default_val;
    }
    return config;
}

// Helper: Clamp value within range
template <typename T>
T clamp(T val, T min_val, T max_val) {
    return std::max(min_val, std::min(val, max_val));
}

// Generate a random configuration within the defined ranges
Configuration get_random_configuration(const ParamSpace& space, std::mt19937& rng) {
    Configuration config;
    for (const auto& pair : space) {
        const ParamDef& def = pair.second;
        if (!def.tune) {
            config[pair.first] = def.default_val;
            continue;
        }
        if (def.type == "int") {
            std::uniform_int_distribution<> dist(std::get<int>(def.min_val), std::get<int>(def.max_val));
            config[pair.first] = dist(rng);
        } else {
            std::uniform_real_distribution<> dist(std::get<double>(def.min_val), std::get<double>(def.max_val));
            config[pair.first] = dist(rng);
        }
    }
    return config;
}

// Run the target solver with a specific configuration
Solution run_solver_with_config(const std::string& solver_name, const VRPInstance& instance, const Configuration& config) {
    Solution sol;
    try {
        if (solver_name == "sa") {
            SAParams params;
            params.max_iter = std::get<int>(config.at("max_iter"));
            params.T0 = std::get<double>(config.at("T0"));
            params.Tf = std::get<double>(config.at("Tf"));
            params.alpha = std::get<double>(config.at("alpha"));
            params.beta = std::get<double>(config.at("beta"));
            params.patience = std::get<int>(config.at("patience"));
            params.p = std::get<double>(config.at("p"));
            sol = SA::solve(instance, params);
        } else if (solver_name == "ga") {
            GAParams params;
            params.population_size = std::get<int>(config.at("population_size"));
            params.generations = std::get<int>(config.at("generations"));
            params.crossover_rate = std::get<double>(config.at("crossover_rate"));
            params.mutation_rate = std::get<double>(config.at("mutation_rate"));
            params.p = std::get<double>(config.at("p"));
            sol = GA::solve(instance, params);
        } else if (solver_name == "aco-ts") {
            ACOTSParams params;
            params.num_ants = std::get<int>(config.at("num_ants"));
            params.num_iterations = std::get<int>(config.at("num_iterations"));
            params.alpha = std::get<double>(config.at("alpha"));
            params.beta = std::get<double>(config.at("beta"));
            params.rho = std::get<double>(config.at("evaporation_rate"));
            params.Q = std::get<double>(config.at("Q"));
            params.p = std::get<double>(config.at("p"));
            params.stagnation_limit = std::get<int>(config.at("stagnation_limit"));
            sol = ACO_TS::solve(instance, params);
        } else if (solver_name == "paco") {
            PACOParams params;
            params.m = std::get<int>(config.at("m"));
            params.alpha = std::get<double>(config.at("alpha"));
            params.beta = std::get<double>(config.at("beta"));
            params.rho = std::get<double>(config.at("rho"));
            params.Q = std::get<double>(config.at("Q"));
            params.I = std::get<int>(config.at("I"));
            params.t = std::get<int>(config.at("t"));
            params.p = std::get<int>(config.at("p"));
            sol = PACO::solve(instance, params);
        } else {
            std::cerr << "Error: Unknown solver name '" << solver_name << "' in run_solver_with_config" << std::endl;
            sol.objective_value = std::numeric_limits<double>::max();
        }
    } catch (...) {
        sol.objective_value = std::numeric_limits<double>::max();
    }
    if (sol.routes.empty() && sol.objective_value < std::numeric_limits<double>::max() / 2.0) {
        sol.objective_value = std::numeric_limits<double>::max();
    }
    return sol;
}

// Evaluate a configuration on a set of instances
double evaluate_configuration(const std::string& solver_name, const Configuration& config, const std::vector<std::string>& instance_files, int num_runs_per_instance, double time_penalty_factor = 0.01) {
    double total_avg_objective = 0.0;
    double total_avg_runtime = 0.0;
    int feasible_instances = 0;
    for (const auto& instance_file : instance_files) {
        std::vector<double> objectives;
        std::vector<double> runtimes;
        bool instance_feasible_at_least_once = false;
        VRPInstance instance;
        try {
            instance = InstanceParser::parse(instance_file);
            instance.build_distance_matrix();
        } catch (...) { continue; }
        for (int run = 0; run < num_runs_per_instance; ++run) {
            auto start = std::chrono::high_resolution_clock::now();
            Solution sol = run_solver_with_config(solver_name, instance, config);
            auto end = std::chrono::high_resolution_clock::now();
            double runtime = std::chrono::duration<double>(end - start).count();
            runtimes.push_back(runtime);
            if (sol.objective_value < std::numeric_limits<double>::max()) {
                objectives.push_back(sol.objective_value);
                instance_feasible_at_least_once = true;
            } else {
                objectives.push_back(std::numeric_limits<double>::max());
            }
        }
        if (instance_feasible_at_least_once) {
            double avg_obj = 0;
            int feasible_runs = 0;
            for(double obj : objectives) {
                if (obj < std::numeric_limits<double>::max()) {
                    avg_obj += obj;
                    feasible_runs++;
                }
            }
            avg_obj = (feasible_runs > 0) ? (avg_obj / feasible_runs) : std::numeric_limits<double>::max();
            double avg_rt = std::accumulate(runtimes.begin(), runtimes.end(), 0.0) / runtimes.size();
            total_avg_objective += avg_obj;
            total_avg_runtime += avg_rt;
            feasible_instances++;
        } else {
            total_avg_objective += std::numeric_limits<double>::max();
            total_avg_runtime += std::accumulate(runtimes.begin(), runtimes.end(), 0.0) / runtimes.size();
        }
    }
    if (feasible_instances == 0) {
        return std::numeric_limits<double>::max();
    }
    double final_avg_objective = total_avg_objective / feasible_instances;
    double final_avg_runtime = total_avg_runtime / instance_files.size();
    double performance_score = final_avg_objective;
    if (performance_score >= std::numeric_limits<double>::max()) {
        return std::numeric_limits<double>::max();
    }
    return performance_score;
}

// Save configuration to YAML
void save_configuration(const std::string& filename, const std::string& solver_name, const Configuration& config) {
    YAML::Emitter out;
    out << YAML::BeginMap;
    out << YAML::Key << solver_name;
    out << YAML::Value << YAML::BeginMap;
    for (const auto& pair : config) {
        out << YAML::Key << pair.first;
        out << YAML::Value;
        std::visit([&out](auto&& arg) { out << arg; }, pair.second);
    }
    out << YAML::EndMap;
    out << YAML::EndMap;
    std::ofstream fout(filename);
    if (!fout.is_open()) {
        std::cerr << "Error: Could not open file for saving configuration: " << filename << std::endl;
        return;
    }
    fout << out.c_str();
    fout.close();
    std::cout << "Saved best configuration to: " << filename << std::endl;
}

// Helper: Serialize ParamSpace and history to JSON
std::string paramspace_to_json(const ParamSpace& space) {
    nlohmann::json j;
    for (const auto& [k, v] : space) {
        nlohmann::json p;
        p["type"] = v.type;
        if (v.type == "int") {
            p["min"] = std::get<int>(v.min_val);
            p["max"] = std::get<int>(v.max_val);
            p["step"] = std::get<int>(v.step);
        } else {
            p["min"] = std::get<double>(v.min_val);
            p["max"] = std::get<double>(v.max_val);
            p["step"] = std::get<double>(v.step);
        }
        p["default"] = v.default_val.index() == 0 ? std::get<int>(v.default_val) : std::get<double>(v.default_val);
        p["tune"] = v.tune;
        j[k] = p;
    }
    return j.dump();
}

std::string config_to_json(const Configuration& config) {
    nlohmann::json j;
    for (const auto& [k, v] : config) {
        if (v.index() == 0) j[k] = std::get<int>(v);
        else j[k] = std::get<double>(v);
    }
    return j.dump();
}

// --- Bayesian Optimization Routine ---
// Implements BO steps: initial random sampling, surrogate fitting, acquisition, evaluation loop
Configuration bayesian_optimization(
    const std::string& solver_name,
    const ParamSpace& space,
    const std::vector<std::string>& instance_files,
    int max_evaluations,
    int num_runs_per_instance,
    double time_penalty_factor,
    int verbose_level
) {
    // Seed random generator
    std::random_device rd;
    std::mt19937 rng(rd());
    // History of evaluated configurations
    struct Eval { Configuration config; double score; };
    std::vector<Eval> history;

    // Step 1: Initial random sampling
    int init_samples = std::max(5, max_evaluations / 5);
    init_samples = std::min(init_samples, max_evaluations);
    if (verbose_level >= 1) std::cout << "BO: Initial random sampling (" << init_samples << " configs)" << std::endl;
    for (int i = 0; i < init_samples; ++i) {
        Configuration cfg = get_random_configuration(space, rng);
        double sc = evaluate_configuration(solver_name, cfg, instance_files, num_runs_per_instance, time_penalty_factor);
        history.push_back({cfg, sc});
        if (verbose_level >= 2) std::cout << "  Init sample " << i+1 << ": score=" << sc << std::endl;
    }
    int evals = init_samples;

    // Best seen configuration
    auto best = std::min_element(history.begin(), history.end(), [](auto &a, auto &b){ return a.score < b.score; });
    Configuration best_config = best->config;
    double best_score = best->score;
    if (verbose_level >= 1) std::cout << "BO: Best initial score=" << best_score << std::endl;

    // Step 2: BO iteration via Python surrogate/acquisition
    while (evals < max_evaluations) {
        // Write paramspace and history to temp files
        std::string param_file = "bao_paramspace.json";
        std::string hist_file = "bao_history.json";
        std::ofstream pf(param_file); pf << paramspace_to_json(space); pf.close();
        nlohmann::json jhist = nlohmann::json::array();
        for (auto& h : history) {
            nlohmann::json je;
            for (const auto& [k, v] : h.config) {
                if (v.index() == 0) je[k] = std::get<int>(v);
                else je[k] = std::get<double>(v);
            }
            je["score"] = h.score;
            jhist.push_back(je);
        }
        std::ofstream hf(hist_file); hf << jhist.dump(); hf.close();
        // Call Python script to get next config
        std::string py_cmd = "uv run ../src/bo_runner.py " + param_file + " " + hist_file;
        FILE* pipe = popen(py_cmd.c_str(), "r");
        if (!pipe) { std::cerr << "Failed to run Python BO script!" << std::endl; break; }
        char buffer[4096];
        std::string result;
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) result += buffer;
        pclose(pipe);
        // Parse returned config
        nlohmann::json jnext = nlohmann::json::parse(result);
        Configuration next_cfg;
        for (auto& [k, v] : jnext.items()) {
            if (!space.count(k)) continue;
            if (space.at(k).type == "int") next_cfg[k] = v.get<int>();
            else next_cfg[k] = v.get<double>();
        }
        double sc = evaluate_configuration(solver_name, next_cfg, instance_files, num_runs_per_instance, time_penalty_factor);
        history.push_back({next_cfg, sc});
        evals++;
        if (verbose_level >= 2) std::cout << "BO eval " << evals << ": score=" << sc << std::endl;
        if (sc < best_score) {
            best_score = sc;
            best_config = next_cfg;
            if (verbose_level >= 1) std::cout << "BO: New best score=" << best_score << std::endl;
        }
    }
    return best_config;
}

// --- Main Function ---
int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: ./bao --solver <name> --params <param_file.yaml> --instances <dir> [--max-evals <int>] [--runs-per-instance <int>] [--output <output_file.yaml>] [--verbose <level>] [--samples <int>]" << std::endl;
        return 1;
    }
    std::string solver_name;
    std::string param_file;
    std::string instance_dir;
    int max_evaluations = 1;
    int num_runs_per_instance = 1;
    std::string output_file = "../output/tuning/tuned_params.yaml";
    int verbose_level = 1;
    int sample_size = -1; // number of instances to sample for tuning
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--solver" && i + 1 < argc) {
            solver_name = argv[++i];
        } else if (arg == "--params" && i + 1 < argc) {
            param_file = argv[++i];
        } else if (arg == "--instances" && i + 1 < argc) {
            instance_dir = argv[++i];
        } else if (arg == "--max-evals" && i + 1 < argc) {
            max_evaluations = std::stoi(argv[++i]);
        } else if (arg == "--runs-per-instance" && i + 1 < argc) {
            num_runs_per_instance = std::stoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "--verbose" && i + 1 < argc) {
            verbose_level = std::stoi(argv[++i]);
        } else if (arg == "--samples" && i + 1 < argc) {
            sample_size = std::stoi(argv[++i]);
        }
    }
    if (solver_name.empty() || param_file.empty() || instance_dir.empty()) {
        std::cerr << "Error: --solver, --params, and --instances are required." << std::endl;
        return 1;
    }
    ParamSpace space = load_param_space(param_file);
    std::vector<std::string> instance_files;
    for (const auto& entry : std::filesystem::directory_iterator(instance_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            instance_files.push_back(entry.path().string());
        }
    }
    if (sample_size > 0 && (int)instance_files.size() > sample_size) {
        std::shuffle(instance_files.begin(), instance_files.end(), std::mt19937{std::random_device{}()});
        instance_files.resize(sample_size);
        if (verbose_level >= 1) std::cout << "Sampling " << sample_size << " instances for tuning." << std::endl;
    }
    if (instance_files.empty()) {
        std::cerr << "Error: No instance files found in " << instance_dir << std::endl;
        return 1;
    }

    // Print out configuration details
    std::cout << "Solver: " << solver_name << std::endl;
    // - Number of runs per instance
    std::cout << "Number of runs per instance: " << num_runs_per_instance << std::endl;

    Configuration best_config = bayesian_optimization(
        solver_name,
        space,
        instance_files,
        max_evaluations,
        num_runs_per_instance,
        1.0, // time_penalty_factor
        verbose_level
    );
    std::filesystem::path out_path(output_file);
    if (out_path.has_parent_path()) {
        try { std::filesystem::create_directories(out_path.parent_path()); } catch (...) {}
    }
    save_configuration(output_file, solver_name, best_config);
    std::cout << "\nBest configuration saved to " << output_file << std::endl;
    return 0;
}
