// ---------------------------------------------------------------------------
// apps/train_neural_net.cpp
//
// Standalone executable to train and benchmark the Neural Network Q-function
// approximator for 3-v-2 Keepaway.
//
// Supports:
//   - Algorithms: Sarsa (--algo sarsa) or Q-learning (--algo qlearn)
//   - Architecture: 3 separate 13-20-1 networks (default) or shared hidden layer (--shared)
//   - Hyperparameters: episodes, eval episodes, alpha (learning rate), momentum, hidden units, epsilon, gamma, seed
//   - Output: Checkpoint file, CSV logs (for both training and greedy evaluation)
//
// Example usage:
//   ./build/train_neural_net --features full --seeds 10
//   ./build/train_neural_net --features reduced --seeds 10
//
// Partially generated with assistance from AI (DeepMind Antigravity / Claude)
// and adapted/verified by human authors.
//
// Changed by an LLM: Claude (Anthropic), 2026-11-23, at Jagjit's request, so the
// network runs under the same protocol as the tile-coding and RBF comparison:
// defaults gamma = 0.99, epsilon = 0.02, 20,000 episodes; --features full|reduced;
// --seeds N (seeds 10, 100010, ...); Q-learning through train_loop (the loop used by
// tile coding and RBF); training CSV in the save_csv format (one row block per
// seed, scored from the training curve); weights saved per seed with --weights-dir.
// The greedy evaluation is kept as a separate CSV.
// Prompts (verbatim): see the header of apps/run_q_learning.cpp, 2026-11-23.
// ---------------------------------------------------------------------------

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstring>
#include <filesystem>
#include <iomanip>

#include "sim/keepaway_env.hpp"
#include "learn/neural_net.hpp"
#include "learn/train.hpp"
#include "common.hpp"

namespace fs = std::filesystem;
using namespace keepaway;

void print_usage(const char* prog) {
  std::cout << "Usage: " << prog << " [options]\n\n"
            << "Options:\n"
            << "  --algo <sarsa|qlearn>   RL algorithm (default: qlearn)\n"
            << "  --episodes <N>          Training episodes per seed (default: 20000)\n"
            << "  --eval-episodes <N>     Evaluation episodes (default: 200)\n"
            << "  --alpha <val>           Learning rate (default: 0.01)\n"
            << "  --momentum <val>        Momentum coefficient (default: 0.0)\n"
            << "  --hidden <N>            Hidden units (default: 20)\n"
            << "  --epsilon <val>         Exploration rate (default: 0.02)\n"
            << "  --gamma <val>           Discount factor (default: 0.99)\n"
            << "  --shared                Use 13-20-3 shared hidden layer architecture\n"
            << "  --features <full|reduced> State variables: all 13, or the 4 reduced ones (default: full)\n"
            << "  --seeds <N>             Number of seeds: seed, seed+100000, ... (default: 10)\n"
            << "  --seed <N>              First seed (default: 10)\n"
            << "  --weights-dir <path>    Folder for per-seed weights (default: results/weights/<run>)\n"
            << "  --out-dir <path>        Output directory for CSVs (default: results)\n"
            << "  --help                  Show this message\n";
}

int main(int argc, char** argv) {
  std::string algo_str = "qlearn";
  int episodes = 20000;
  int eval_episodes = 200;
  double alpha = 0.01;
  double momentum = 0.0;
  int hidden = 20;
  double epsilon = 0.02;
  double gamma = 0.99;
  bool shared = false;
  uint64_t seed = 10;
  int seed_count = 10;
  const uint64_t seed_multiplier = 100000;  // same spacing as apps/run_q_learning.cpp
  std::string features_str = "full";
  std::string weights_dir = "";
  std::string out_dir = "results";

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--algo" && i + 1 < argc) {
      algo_str = argv[++i];
    } else if (arg == "--episodes" && i + 1 < argc) {
      episodes = std::stoi(argv[++i]);
    } else if (arg == "--eval-episodes" && i + 1 < argc) {
      eval_episodes = std::stoi(argv[++i]);
    } else if (arg == "--alpha" && i + 1 < argc) {
      alpha = std::stod(argv[++i]);
    } else if (arg == "--momentum" && i + 1 < argc) {
      momentum = std::stod(argv[++i]);
    } else if (arg == "--hidden" && i + 1 < argc) {
      hidden = std::stoi(argv[++i]);
    } else if (arg == "--epsilon" && i + 1 < argc) {
      epsilon = std::stod(argv[++i]);
    } else if (arg == "--gamma" && i + 1 < argc) {
      gamma = std::stod(argv[++i]);
    } else if (arg == "--shared") {
      shared = true;
    } else if (arg == "--seed" && i + 1 < argc) {
      seed = std::stoull(argv[++i]);
    } else if (arg == "--seeds" && i + 1 < argc) {
      seed_count = std::stoi(argv[++i]);
    } else if (arg == "--features" && i + 1 < argc) {
      features_str = argv[++i];
    } else if (arg == "--weights-dir" && i + 1 < argc) {
      weights_dir = argv[++i];
    } else if (arg == "--out-dir" && i + 1 < argc) {
      out_dir = argv[++i];
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      print_usage(argv[0]);
      return 1;
    }
  }

  RLAlgorithm algo = (algo_str == "sarsa") ? RLAlgorithm::Sarsa : RLAlgorithm::QLearning;
  if (features_str != "full" && features_str != "reduced") {
    std::cerr << "--features must be full or reduced\n";
    return 1;
  }
  const std::vector<Feature>& features = (features_str == "full") ? kFullFeatures : kReducedFeatures;
  // Run name matches the tile-coding / RBF comparison files (cmp-tc-final, cmp-rbf-reduced-final, ...)
  const std::string run_name = (features_str == "full") ? "cmp-nn-final" : "cmp-nn-reduced-final";
  if (weights_dir.empty()) {
    weights_dir = "results/weights/" + run_name;
  }

  std::cout << "=========================================================\n";
  std::cout << "  3-v-2 Keepaway: Neural Network Function Approximator   \n";
  std::cout << "=========================================================\n";
  std::cout << "Algorithm:       " << (algo == RLAlgorithm::Sarsa ? "Sarsa (on-policy, train_rl)" : "Q-learning (off-policy, train_loop)") << "\n";
  std::cout << "Architecture:    " << (shared ? "Shared hidden layer" : "3 separate networks, one per action") << "\n";
  std::cout << "Inputs:          " << features.size() << " normalized features (" << features_str << ")\n";
  std::cout << "Hidden Units:    " << hidden << " (sigmoid clamped [-45, 45])\n";
  std::cout << "Outputs:         " << kNumActions << " linear action values\n";
  std::cout << "Training Runs:   " << episodes << " episodes per seed (epsilon=" << epsilon << ")\n";
  std::cout << "Evaluation Runs: " << eval_episodes << " episodes per seed (greedy policy)\n";
  std::cout << "Hyperparameters: alpha=" << alpha << ", momentum=" << momentum << ", gamma=" << gamma << "\n";
  std::cout << "Seeds:           " << seed_count << " (" << seed << ", " << seed + seed_multiplier << ", ...)\n";
  std::cout << "---------------------------------------------------------\n";

  fs::create_directories(out_dir);
  fs::create_directories(weights_dir);

  std::vector<Row> train_rows;
  std::vector<Row> eval_rows;

  for (int s = 0; s < seed_count; ++s) {
    const uint64_t run_seed = seed + s * seed_multiplier;

    Config env_cfg;
    Env env(env_cfg);

    Config_NN nn_cfg;
    nn_cfg.num_inputs = static_cast<int>(features.size());
    nn_cfg.num_hidden = hidden;
    nn_cfg.num_actions = kNumActions;
    nn_cfg.alpha = alpha;
    nn_cfg.momentum = momentum;
    nn_cfg.weight_decay = 0.0;
    nn_cfg.clip_td_error = true;
    nn_cfg.shared_hidden = shared;
    nn_cfg.seed = run_seed;

    NeuralNet nn(nn_cfg);

    std::cout << "Training seed " << run_seed << "...\n";
    auto start_time = std::chrono::steady_clock::now();

    // Q-learning uses train_loop, the same loop as tile coding and RBF.
    std::vector<Episode> train_eps =
        (algo == RLAlgorithm::QLearning)
            ? train_loop(nn, env, env_cfg, features, gamma, episodes, run_seed, epsilon)
            : train_rl(nn, env, env_cfg, features, algo, gamma, epsilon, episodes, run_seed);

    auto end_time = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "Training completed in " << std::fixed << std::setprecision(2) << elapsed << " s ("
              << (episodes / elapsed) << " eps/s)\n";

    const std::string name = "seed-" + std::to_string(run_seed);
    print_row(name, train_eps);
    print_block_averages(train_eps);

    // Greedy evaluation on seeds that no training run uses (extra; the comparison uses the training curve)
    std::vector<Episode> eval_eps = evaluate_policy(nn, env, env_cfg, features, eval_episodes, run_seed + 1000000);
    print_row(name + "-greedy", eval_eps);

    const std::string weights_path = weights_dir + "/" + name + ".txt";
    nn.save(weights_path);
    std::cout << "Saved weights to: " << weights_path << "\n\n";

    train_rows.push_back({name, train_eps});
    eval_rows.push_back({name, eval_eps});
  }

  // Same CSV format as the tile-coding / RBF runs: policy,episode,seconds,end_reason
  save_csv(train_rows, out_dir + "/" + run_name);
  save_csv(eval_rows, out_dir + "/" + run_name + "-greedy-eval");
  std::cout << "Saved training CSV to " << out_dir << "/" << run_name << "_<timestamp>.csv\n";
  std::cout << "Saved greedy evaluation CSV to " << out_dir << "/" << run_name << "-greedy-eval_<timestamp>.csv\n";

  return 0;
}
