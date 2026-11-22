// ---------------------------------------------------------------------------
// q_learning.cpp
// Benchmarking Q-learning with Tile Coding and Neural Network approximators.
// ---------------------------------------------------------------------------

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "common.hpp"
#include "learn/neural_net.hpp"
#include "learn/tile_coding.hpp"
#include "learn/train.hpp"
#include "sim/keepaway_env.hpp"

namespace keepaway
{
    Row train_tilecoding_full(int episodes, const std::string &out_dir)
    {
        Config cfg;
        Env env(cfg);

        const auto &subset = kFullFeatures;
        Config_TC config;
        config.num_of_actions = kNumActions;
        config.num_of_tiles = 10;
        config.num_of_tilings = 10;
        config.num_of_state_variables = static_cast<int>(subset.size());
        config.alpha = 0.1 / (config.num_of_state_variables * config.num_of_tilings);

        TileCoding q(config);
        std::vector<Row> rows;
        rows.push_back({"TileCoding", train_loop(q, env, cfg, subset, 0.9, episodes, 1)});
        save_csv(rows, out_dir);

        return rows[0];
    }

    Row train_neuralnet_full(int episodes, const std::string &out_dir)
    {
        Config cfg;
        Env env(cfg);

        const auto &subset = kFullFeatures;
        Config_NN config;
        config.num_of_actions = kNumActions;
        config.num_of_state_variables = static_cast<int>(subset.size());
        config.num_of_hidden = 20;
        config.alpha = 0.01;
        config.gradient_clip = 1.0;
        config.seed = 1;
        config.architecture = NNArchitecture::SeparatePerAction;

        NeuralNet q(config);
        std::vector<Row> rows;
        rows.push_back({"NeuralNet", train_loop(q, env, cfg, subset, 0.9, episodes, 1, 0.02)});
        save_csv(rows, out_dir);

        return rows[0];
    }
} // namespace keepaway

int main(int argc, char **argv)
{
    std::filesystem::create_directories("results");
    std::string out_dir = "results";
    int episodes = 1000;
    std::string mode = "all"; // "all", "tc", "nn"

    if (argc > 1)
    {
        episodes = std::atoi(argv[1]);
    }
    if (argc > 2)
    {
        mode = argv[2];
    }

    std::vector<Row> rows;

    if (mode == "all" || mode == "tc")
    {
        std::printf("Training TileCoding for %d episodes...\n", episodes);
        rows.push_back(keepaway::train_tilecoding_full(episodes, out_dir));
    }
    if (mode == "all" || mode == "nn")
    {
        std::printf("Training NeuralNet (13-20-1) for %d episodes...\n", episodes);
        rows.push_back(keepaway::train_neuralnet_full(episodes, out_dir));
    }

    std::printf("\n================================ Performance Comparison ================================\n");
    for (const Row &r : rows)
    {
        print_row(r.name, r.eps);
    }

    return 0;
}
