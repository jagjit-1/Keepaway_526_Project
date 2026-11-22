// ---------------------------------------------------------------------------
// train_neural_net.cpp
// Benchmarking and training application for Neural Network function approximator
// in 3 vs 2 Keepaway.
//
// Based on:
//   Stone, Kuhlmann, Taylor & Liu (2005),
//   "Keepaway Soccer: From Machine Learning Testbed to Benchmark", Section 4.3.
// ---------------------------------------------------------------------------

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "common.hpp"
#include "learn/neural_net.hpp"
#include "learn/train.hpp"
#include "sim/keepaway_env.hpp"

using namespace keepaway;

int main(int argc, char **argv)
{
    int episodes = 1000;
    int eval_episodes = 200;
    uint64_t seed0 = 1;
    int hidden_units = 20;
    double alpha = 0.01;
    double momentum = 0.0;
    double epsilon = 0.02;
    double gamma = 0.9;
    std::string algo = "sarsa"; // "sarsa" or "qlearn"
    std::string out_dir = "results";
    std::string model_path = "";
    std::string log_traj_path = "";

    // Parse command line arguments
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--episodes" && i + 1 < argc)
        {
            episodes = std::atoi(argv[++i]);
        }
        else if (arg == "--algo" && i + 1 < argc)
        {
            algo = argv[++i];
        }
        else if (arg == "--hidden" && i + 1 < argc)
        {
            hidden_units = std::atoi(argv[++i]);
        }
        else if (arg == "--alpha" && i + 1 < argc)
        {
            alpha = std::atof(argv[++i]);
        }
        else if (arg == "--momentum" && i + 1 < argc)
        {
            momentum = std::atof(argv[++i]);
        }
        else if (arg == "--epsilon" && i + 1 < argc)
        {
            epsilon = std::atof(argv[++i]);
        }
        else if (arg == "--seed" && i + 1 < argc)
        {
            seed0 = std::strtoull(argv[++i], nullptr, 10);
        }
        else if (arg == "--out" && i + 1 < argc)
        {
            out_dir = argv[++i];
        }
        else if (arg == "--save-model" && i + 1 < argc)
        {
            model_path = argv[++i];
        }
        else if (arg == "--log-traj" && i + 1 < argc)
        {
            log_traj_path = argv[++i];
        }
        else if (arg[0] != '-' && i == 1)
        {
            episodes = std::atoi(argv[i]);
        }
    }

    std::filesystem::create_directories(out_dir);
    if (model_path.empty())
    {
        model_path = out_dir + "/nn_model_" + algo + ".txt";
    }

    std::printf("===================================================================\n");
    std::printf("Keepaway 3v2 Neural Network Experiment (Stone et al., 2005)\n");
    std::printf("Architecture: 3 separate %d-%d-1 networks (one per action)\n",
                static_cast<int>(kFullFeatures.size()), hidden_units);
    std::printf("Algorithm: SMDP %s | Alpha: %.4f | Momentum: %.2f | Epsilon: %.3f\n",
                algo.c_str(), alpha, momentum, epsilon);
    std::printf("Training for %d episodes starting from seed %llu\n", episodes,
                static_cast<unsigned long long>(seed0));
    std::printf("===================================================================\n");

    Config cfg;
    Env env(cfg);
    const auto &subset = kFullFeatures;

    Config_NN nn_cfg;
    nn_cfg.num_of_state_variables = static_cast<int>(subset.size());
    nn_cfg.num_of_hidden = hidden_units;
    nn_cfg.num_of_actions = kNumActions;
    nn_cfg.alpha = alpha;
    nn_cfg.momentum = momentum;
    nn_cfg.gradient_clip = 1.0;
    nn_cfg.seed = seed0;
    nn_cfg.architecture = NNArchitecture::SeparatePerAction;

    NeuralNet q(nn_cfg);

    // Run training
    std::vector<Episode> train_eps;
    if (algo == "sarsa")
    {
        train_eps = train_sarsa_loop(q, env, cfg, subset, gamma, episodes, seed0, epsilon);
    }
    else
    {
        train_eps = train_loop(q, env, cfg, subset, gamma, episodes, seed0, epsilon);
    }

    std::vector<Row> rows;
    std::string train_name = "NN_" + algo + "_train";
    rows.push_back({train_name, train_eps});

    // Run post-training evaluation (pure greedy policy, no exploration noise)
    std::printf("\nEvaluating frozen learned policy on %d unseen episodes...\n", eval_episodes);
    auto eval_eps = evaluate_policy(q, env, cfg, subset, eval_episodes, seed0 + episodes + 1000);
    std::string eval_name = "NN_" + algo + "_eval";
    rows.push_back({eval_name, eval_eps});

    // Save model
    q.save(model_path);
    std::printf("Model weights saved to %s\n\n", model_path.c_str());

    // Print summary
    std::printf("Results Summary:\n");
    for (const Row &r : rows)
    {
        print_row(r.name, r.eps);
    }

    // Save CSV
    save_csv(rows, out_dir);
    std::printf("Saved episode statistics to %s/benchmarks.csv\n", out_dir.c_str());

    // Optionally record a replay trajectory
    if (!log_traj_path.empty())
    {
        env.set_trajectory_log(log_traj_path);
        env.reset(seed0 + episodes + 2000);
        while (!env.reset(seed0 + episodes + 2000).done)
            ;
        env.set_trajectory_log("");
        std::printf("Logged replay trajectory to %s\n", log_traj_path.c_str());
    }

    return 0;
}
