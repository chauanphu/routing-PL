#include "InstanceParser.h"
#include "Solver.h"
#include "SA.h"
#include <iostream>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <chrono>
#include <filesystem>
#include <algorithm>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <experiment_size> [instance_file]" << std::endl;
        std::cout << "  experiment_size: small | medium | large" << std::endl;
        return 1;
    }
    std::string exp_size = argv[1];
    std::string instance_file = (argc >= 3) ? argv[2] : "";

    // Parse param.yaml
    YAML::Node config = YAML::LoadFile("../param.yaml");
    if (!config[exp_size]) {
        std::cerr << "Experiment size '" << exp_size << "' not found in param.yaml" << std::endl;
        return 1;
    }
    auto sa_params_node = config[exp_size]["sa_params"];
    SAParams sa_params;
    sa_params.max_iter = sa_params_node["max_iter"].as<int>();
    sa_params.T0 = sa_params_node["T0"].as<double>();
    sa_params.Tf = sa_params_node["Tf"].as<double>();
    sa_params.alpha = sa_params_node["alpha"].as<double>();
    sa_params.beta = sa_params_node["beta"].as<double>();
    sa_params.patience = sa_params_node["patience"].as<int>();
    sa_params.p = sa_params_node["p"].as<double>();

    std::vector<std::string> instance_files;
    if (!instance_file.empty()) {
        instance_files.push_back(instance_file);
    } else {
        std::string data_dir = config[exp_size]["data_dir"].as<std::string>();
        for (const auto& entry : std::filesystem::directory_iterator("../" + data_dir)) {
            if (entry.is_regular_file()) {
                instance_files.push_back(entry.path().string());
            }
        }
        // Sort: C -> R -> RC, then by number ascending
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
    }
    std::string output_csv = "../" + config[exp_size]["output_csv"].as<std::string>();
    int num_runs = config[exp_size]["num_runs"].as<int>();

    std::ofstream ofs(output_csv);
    ofs << "instance_name,Num Vehicles,Best Distance,AVG Distance,Std Distance,AVG Runtime (s),Std Runtime (s)\n";

    // Print loaded parameters
    std::cout << "Loaded parameters for experiment size: " << exp_size << std::endl;
    std::cout << "  max_iter: " << sa_params.max_iter << std::endl;
    std::cout << "  T0: " << sa_params.T0 << std::endl;
    std::cout << "  Tf: " << sa_params.Tf << std::endl;
    std::cout << "  alpha: " << sa_params.alpha << std::endl;
    std::cout << "  beta: " << sa_params.beta << std::endl;
    std::cout << "  patience: " << sa_params.patience << std::endl;
    std::cout << "  p: " << sa_params.p << std::endl;
    std::cout << "  num_runs: " << num_runs << std::endl;
    std::cout << "  output_csv: " << output_csv << std::endl;
    std::cout << "  data_dir: " << config[exp_size]["data_dir"].as<std::string>() << std::endl;

    for (const auto& file : instance_files) {
        std::cout << "\nProcessing instance: " << file << std::endl;
        std::vector<double> distances, runtimes;
        int best_num_vehicles = 0;
        double best_distance = 1e12;
        for (int run = 0; run < num_runs; ++run) {
            auto start = std::chrono::high_resolution_clock::now();
            VRPInstance instance = InstanceParser::parse(file);
            instance.build_distance_matrix();
            Solution sol = SA::solve(instance, sa_params);
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
        // Compute statistics
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
        std::cout << "Results saved to: " << output_csv << std::endl;
    }
    ofs.close();
    return 0;
}
