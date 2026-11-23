// ---------------------------------------------------------------------------
// neural_net.hpp
// Neural network function approximator for 3-v-2 Keepaway.
//
// Implements the Approximator interface using either:
//   1. Default: Three separate networks (one 13-20-1 network per action)
//   2. Experimental: Shared hidden layer (one 13-20-3 network)
//
// Uses sigmoid hidden units and linear outputs with temporal difference (TD)
// backpropagation updates.
//
// Partially generated with assistance from AI (DeepMind Antigravity / Claude)
// and adapted/verified by human authors.
// ---------------------------------------------------------------------------

#pragma once

#include <string>
#include <vector>
#include <random>
#include <cstdint>
#include "approximator.hpp"

namespace keepaway {

// Configuration for NeuralNet function approximator
struct Config_NN {
  int num_inputs = 13;          // Number of normalized input features (default 13)
  int num_hidden = 20;          // Number of sigmoid hidden units per network
  int num_actions = 3;          // Number of discrete actions (HOLD, PASS_F2, PASS_F3)
  double alpha = 0.01;          // Learning rate
  double momentum = 0.0;        // Polyak momentum coefficient [0, 1)
  double weight_decay = 0.0;    // Optional L2 weight decay (regularization)
  bool clip_td_error = true;    // Whether to clip TD error to [-1.0, 1.0]
  bool shared_hidden = false;   // false = 3 separate 13-20-1 subnets, true = 13-20-3 shared
  uint64_t seed = 42;           // Random seed for Xavier weight initialization
};

// A single 2-layer MLP subnet (Inputs -> Sigmoid Hidden -> 1 Linear Output)
class SubNet {
 public:
  SubNet() = default;
  SubNet(int num_in, int num_hid, double lr, double mom, double decay, bool clip_err, std::mt19937_64& rng);

  // Computes linear output given input x and caches hidden activations
  double forward(const std::vector<double>& x) const;

  // Backpropagates TD error: error = target - predicted_output
  void backward(const std::vector<double>& x, double target);

  // Save / Load weights to/from text stream
  void save(std::ostream& os) const;
  void load(std::istream& is);

  int num_inputs_ = 13;
  int num_hidden_ = 20;
  double alpha_ = 0.01;
  double momentum_ = 0.0;
  double weight_decay_ = 0.0;
  bool clip_td_error_ = true;

  // Weights and biases:
  // W1: [num_hidden x num_inputs], b1: [num_hidden]
  // W2: [num_hidden], b2: scalar
  std::vector<std::vector<double>> w1_;
  std::vector<double> b1_;
  std::vector<double> w2_;
  double b2_ = 0.0;

  // Momentum velocities
  std::vector<std::vector<double>> v_w1_;
  std::vector<double> v_b1_;
  std::vector<double> v_w2_;
  double v_b2_ = 0.0;

  // Cached hidden activations from last forward pass (mutable for const forward)
  mutable std::vector<double> hidden_acts_;
};

// Main NeuralNet approximator implementing keepaway::Approximator
class NeuralNet : public Approximator {
 public:
  NeuralNet(const Config_NN& cfg = Config_NN());

  // Approximator interface:
  double value(const std::vector<double>& x, int action) const override;
  void update(const std::vector<double>& x, int action, double target) override;
  void save(const std::string& path) const override;
  void load(const std::string& path) override;

  const Config_NN& config() const { return cfg_; }

 private:
  Config_NN cfg_;

  // 1. Separate networks: one SubNet per action
  std::vector<SubNet> subnets_;

  // 2. Shared-hidden network parameters (used if cfg_.shared_hidden == true):
  // W1: [num_hidden x num_inputs], b1: [num_hidden]
  // W2: [num_actions x num_hidden], b2: [num_actions]
  std::vector<std::vector<double>> shared_w1_;
  std::vector<double> shared_b1_;
  std::vector<std::vector<double>> shared_w2_;
  std::vector<double> shared_b2_;

  // Momentum velocities for shared network
  std::vector<std::vector<double>> v_shared_w1_;
  std::vector<double> v_shared_b1_;
  std::vector<std::vector<double>> v_shared_w2_;
  std::vector<double> v_shared_b2_;

  mutable std::vector<double> shared_hidden_acts_;

  void init_shared_network(std::mt19937_64& rng);
  double forward_shared(const std::vector<double>& x, int action) const;
  void backward_shared(const std::vector<double>& x, int action, double target);
};

}  // namespace keepaway
