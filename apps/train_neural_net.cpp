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
//   ./build/train_neural_net --algo qlearn --episodes 1000 --eval-episodes 200 --alpha 0.01 --hidden 20
//
// Partially generated with assistance from AI (DeepMind Antigravity / Claude)
// and adapted/verified by human authors.
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
            << "  --episodes <N>          Training episodes (default: 1000)\n"
            << "  --eval-episodes <N>     Evaluation episodes (default: 200)\n"
            << "  --alpha <val>           Learning rate (default: 0.01)\n"
            << "  --momentum <val>        Momentum coefficient (default: 0.0)\n"
            << "  --hidden <N>            Hidden units (default: 20)\n"
            << "  --epsilon <val>         Exploration rate (default: 0.02)\n"
            << "  --gamma <val>           Discount factor (default: 0.9)\n"
            << "  --shared                Use 13-20-3 shared hidden layer architecture\n"
            << "  --seed <N>              Random seed (default: 1)\n"
            << "  --checkpoint <path>     Save checkpoint model to path\n"
            << "  --out-dir <path>        Output directory for CSVs (default: results/nn)\n"
            << "  --help                  Show this message\n";
}

int main(int argc, char** argv) {
  std::string algo_str = "qlearn";
  int episodes = 1000;
  int eval_episodes = 200;
  double alpha = 0.01;
  double momentum = 0.0;
  int hidden = 20;
  double epsilon = 0.02;
  double gamma = 0.9;
  bool shared = false;
  uint64_t seed = 1;
  std::string checkpoint_path = "";
  std::string out_dir = "results/nn";

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
    } else if (arg == "--checkpoint" && i + 1 < argc) {
      checkpoint_path = argv[++i];
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

  std::cout << "=========================================================\n";
  std::cout << "  3-v-2 Keepaway: Neural Network Function Approximator   \n";
  std::cout << "=========================================================\n";
  std::cout << "Algorithm:       " << (algo == RLAlgorithm::Sarsa ? "Sarsa (on-policy)" : "Q-learning (off-policy)") << "\n";
  std::cout << "Architecture:    " << (shared ? "13-20-3 Shared Hidden Layer" : "3 Separate 13-20-1 Networks") << "\n";
  std::cout << "Inputs:          " << kNumFeatures << " normalized features\n";
  std::cout << "Hidden Units:    " << hidden << " (sigmoid clamped [-45, 45])\n";
  std::cout << "Outputs:         " << kNumActions << " linear action values\n";
  std::cout << "Training Runs:   " << episodes << " episodes (epsilon=" << epsilon << ")\n";
  std::cout << "Evaluation Runs: " << eval_episodes << " episodes (greedy policy)\n";
  std::cout << "Hyperparameters: alpha=" << alpha << ", momentum=" << momentum << ", gamma=" << gamma << "\n";
  std::cout << "Random Seed:     " << seed << "\n";
  std::cout << "---------------------------------------------------------\n";

  fs::create_directories(out_dir);

  Config env_cfg;
  Env env(env_cfg);

  Config_NN nn_cfg;
  nn_cfg.num_inputs = kNumFeatures;
  nn_cfg.num_hidden = hidden;
  nn_cfg.num_actions = kNumActions;
  nn_cfg.alpha = alpha;
  nn_cfg.momentum = momentum;
  nn_cfg.weight_decay = 0.0;
  nn_cfg.clip_td_error = true;
  nn_cfg.shared_hidden = shared;
  nn_cfg.seed = seed;

  NeuralNet nn(nn_cfg);

  std::cout << "Starting training...\n";
  auto start_time = std::chrono::steady_clock::now();

  std::vector<Episode> train_eps = train_rl(nn, env, env_cfg, kFullFeatures, algo, gamma, epsilon, episodes, seed);

  auto end_time = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration<double>(end_time - start_time).count();
  std::cout << "Training completed in " << std::fixed << std::setprecision(2) << elapsed << " s ("
            << (episodes / elapsed) << " eps/s)\n";

  print_row("NN_Train", train_eps);
  if (train_eps.size() >= 200) {
    size_t blk = (train_eps.size() >= 1000) ? 200 : (train_eps.size() / 5);
    print_block_averages(train_eps, blk);
  }

  // Evaluation phase: run greedy episodes
  std::cout << "\nStarting greedy evaluation (unseen seeds)...\n";
  std::vector<Episode> eval_eps = evaluate_policy(nn, env, env_cfg, kFullFeatures, eval_episodes, seed + 1000000);
  print_row("NN_Greedy_Eval", eval_eps);

  // Save checkpoint if requested
  if (checkpoint_path.empty()) {
    checkpoint_path = out_dir + "/model_" + algo_str + "_" + (shared ? "shared" : "sep") + ".txt";
  }
  nn.save(checkpoint_path);
  std::cout << "\nSaved model checkpoint to: " << checkpoint_path << "\n";

  // Write CSV results
  std::string train_csv = out_dir + "/train_" + algo_str + "_" + (shared ? "shared" : "sep") + ".csv";
  std::ofstream ofs_train(train_csv);
  ofs_train << "policy,episode,seconds,end_reason\n";
  for (size_t ep = 0; ep < train_eps.size(); ++ep) {
    ofs_train << "NN_Train," << ep << "," << train_eps[ep].seconds << "," << reason_name(train_eps[ep].reason) << "\n";
  }
  ofs_train.close();
  std::cout << "Saved training log to: " << train_csv << "\n";

  std::string eval_csv = out_dir + "/eval_" + algo_str + "_" + (shared ? "shared" : "sep") + ".csv";
  std::ofstream ofs_eval(eval_csv);
  ofs_eval << "policy,episode,seconds,end_reason\n";
  for (size_t ep = 0; ep < eval_eps.size(); ++ep) {
    ofs_eval << "NN_Eval," << ep << "," << eval_eps[ep].seconds << "," << reason_name(eval_eps[ep].reason) << "\n";
  }
  ofs_eval.close();
  std::cout << "Saved evaluation log to: " << eval_csv << "\n";

  // Write benchmark summary format
  std::string bench_csv = out_dir + "/benchmarks.csv";
  bool append = fs::exists(bench_csv);
  std::ofstream ofs_bench(bench_csv, std::ios::app);
  if (!append) {
    ofs_bench << "policy,episode,seconds,end_reason\n";
  }
  for (size_t ep = 0; ep < eval_eps.size(); ++ep) {
    ofs_bench << "NeuralNet_" << algo_str << "," << ep << "," << eval_eps[ep].seconds << "," << reason_name(eval_eps[ep].reason) << "\n";
  }
  ofs_bench.close();
  std::cout << "Updated benchmarks to: " << bench_csv << "\n";

  return 0;
}
