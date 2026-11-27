// ---------------------------------------------------------------------------
// neural_net.cpp
// Neural network function approximator implementation for 3-v-2 Keepaway.
//
// Implements:
//   - SubNet forward / backward with Xavier initialization and clamped sigmoid.
//   - NeuralNet separate-per-action subnets and experimental shared-hidden mode.
//   - Model persistence (save/load).
//
// Partially generated with assistance from AI (DeepMind Antigravity / Claude)
// and adapted/verified by human authors.
//
// Changed by an LLM: Claude (Anthropic), 2026-11-26, at Jagjit's request: in
// SubNet::backward and NeuralNet::backward_shared, the hidden-layer gradient now uses
// the output weights from before the update. Previously the output weights were updated
// first and the new values were used, so each step mixed two different networks.
// ---------------------------------------------------------------------------

#include "neural_net.hpp"
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace keepaway {

namespace {

// Clamped sigmoid activation to prevent exp() underflow or overflow
inline double sigmoid(double z) {
  // Clamp z to [-45, 45]
  double z_clamped = std::clamp(z, -45.0, 45.0);
  return 1.0 / (1.0 + std::exp(-z_clamped));
}

// Derivative of sigmoid: sigma(z) * (1 - sigma(z))
inline double d_sigmoid(double act) {
  return act * (1.0 - act);
}

}  // namespace

// ===========================================================================
// SubNet Implementation (Separate Network for 1 action)
// ===========================================================================

SubNet::SubNet(int num_in, int num_hid, double lr, double mom, double decay, bool clip_err, std::mt19937_64& rng)
    : num_inputs_(num_in),
      num_hidden_(num_hid),
      alpha_(lr),
      momentum_(mom),
      weight_decay_(decay),
      clip_td_error_(clip_err) {
  hidden_acts_.resize(num_hidden_, 0.0);

  // Xavier/Glorot uniform initialization:
  // For layer 1: uniform in [-limit, limit], limit = sqrt(6 / (fan_in + fan_out))
  double limit1 = std::sqrt(6.0 / (num_inputs_ + num_hidden_));
  std::uniform_real_distribution<double> dist1(-limit1, limit1);

  w1_.resize(num_hidden_, std::vector<double>(num_inputs_));
  v_w1_.resize(num_hidden_, std::vector<double>(num_inputs_, 0.0));
  b1_.resize(num_hidden_, 0.0);
  v_b1_.resize(num_hidden_, 0.0);

  for (int j = 0; j < num_hidden_; ++j) {
    b1_[j] = 0.0;
    for (int i = 0; i < num_inputs_; ++i) {
      w1_[j][i] = dist1(rng);
    }
  }

  // For layer 2 (hidden to linear output):
  double limit2 = std::sqrt(6.0 / (num_hidden_ + 1));
  std::uniform_real_distribution<double> dist2(-limit2, limit2);

  w2_.resize(num_hidden_);
  v_w2_.resize(num_hidden_, 0.0);
  b2_ = 0.0;
  v_b2_ = 0.0;

  for (int j = 0; j < num_hidden_; ++j) {
    w2_[j] = dist2(rng);
  }
}

double SubNet::forward(const std::vector<double>& x) const {
  // 1. Hidden layer: z_j = b1_j + sum_i(w1_ji * x_i), h_j = sigmoid(z_j)
  for (int j = 0; j < num_hidden_; ++j) {
    double z = b1_[j];
    for (int i = 0; i < num_inputs_; ++i) {
      z += w1_[j][i] * x[i];
    }
    hidden_acts_[j] = sigmoid(z);
  }

  // 2. Output layer (linear): out = b2 + sum_j(w2_j * h_j)
  double out = b2_;
  for (int j = 0; j < num_hidden_; ++j) {
    out += w2_[j] * hidden_acts_[j];
  }
  return out;
}

void SubNet::backward(const std::vector<double>& x, double target) {
  // Forward pass to ensure cached hidden activations are up to date
  double pred = forward(x);

  // TD error = target - predicted
  double td_error = target - pred;
  if (clip_td_error_) {
    td_error = std::clamp(td_error, -1.0, 1.0);
  }

  // Gradient of Loss = 0.5 * (target - out)^2:
  // dL/dout = -(target - out) = -td_error
  // Parameter update in direction of -dL/dtheta:
  // Delta_theta = alpha * td_error * dout/dtheta

  // Hidden-layer gradients are computed first, with the output weights as they were
  // before this update (dout/dh_j = w2_j, dh_j/dz_j = hidden_acts_[j] * (1 - hidden_acts_[j])).
  std::vector<double> grad_h(num_hidden_);
  for (int j = 0; j < num_hidden_; ++j) {
    grad_h[j] = td_error * w2_[j] * d_sigmoid(hidden_acts_[j]);
  }

  // 1. Output layer updates:
  // dout / db2 = 1
  // dout / dw2_j = hidden_acts_[j]
  double delta_b2 = alpha_ * td_error;
  v_b2_ = momentum_ * v_b2_ + delta_b2;
  b2_ += v_b2_;

  std::vector<double> delta_w2(num_hidden_);
  for (int j = 0; j < num_hidden_; ++j) {
    delta_w2[j] = alpha_ * (td_error * hidden_acts_[j] - weight_decay_ * w2_[j]);
    v_w2_[j] = momentum_ * v_w2_[j] + delta_w2[j];
    w2_[j] += v_w2_[j];
  }

  // 2. Hidden layer updates:
  // dout / dh_j = w2_j
  // dh_j / dz_j = hidden_acts_[j] * (1 - hidden_acts_[j])
  // dz_j / db1_j = 1
  // dz_j / dw1_ji = x_i
  for (int j = 0; j < num_hidden_; ++j) {
    double delta_b1 = alpha_ * grad_h[j];
    v_b1_[j] = momentum_ * v_b1_[j] + delta_b1;
    b1_[j] += v_b1_[j];

    for (int i = 0; i < num_inputs_; ++i) {
      double delta_w1 = alpha_ * (grad_h[j] * x[i] - weight_decay_ * w1_[j][i]);
      v_w1_[j][i] = momentum_ * v_w1_[j][i] + delta_w1;
      w1_[j][i] += v_w1_[j][i];
    }
  }
}

void SubNet::save(std::ostream& os) const {
  os << num_inputs_ << " " << num_hidden_ << "\n";
  // b1
  for (int j = 0; j < num_hidden_; ++j) {
    os << b1_[j] << (j + 1 == num_hidden_ ? "\n" : " ");
  }
  // w1
  for (int j = 0; j < num_hidden_; ++j) {
    for (int i = 0; i < num_inputs_; ++i) {
      os << w1_[j][i] << (i + 1 == num_inputs_ ? "\n" : " ");
    }
  }
  // b2
  os << b2_ << "\n";
  // w2
  for (int j = 0; j < num_hidden_; ++j) {
    os << w2_[j] << (j + 1 == num_hidden_ ? "\n" : " ");
  }
}

void SubNet::load(std::istream& is) {
  is >> num_inputs_ >> num_hidden_;
  hidden_acts_.resize(num_hidden_, 0.0);
  b1_.resize(num_hidden_);
  v_b1_.assign(num_hidden_, 0.0);
  for (int j = 0; j < num_hidden_; ++j) {
    is >> b1_[j];
  }
  w1_.assign(num_hidden_, std::vector<double>(num_inputs_));
  v_w1_.assign(num_hidden_, std::vector<double>(num_inputs_, 0.0));
  for (int j = 0; j < num_hidden_; ++j) {
    for (int i = 0; i < num_inputs_; ++i) {
      is >> w1_[j][i];
    }
  }
  is >> b2_;
  v_b2_ = 0.0;
  w2_.resize(num_hidden_);
  v_w2_.assign(num_hidden_, 0.0);
  for (int j = 0; j < num_hidden_; ++j) {
    is >> w2_[j];
  }
}

// ===========================================================================
// NeuralNet Implementation (Approximator wrapper)
// ===========================================================================

NeuralNet::NeuralNet(const Config_NN& cfg) : cfg_(cfg) {
  std::mt19937_64 rng(cfg_.seed);

  if (!cfg_.shared_hidden) {
    // 3 separate action networks
    subnets_.reserve(cfg_.num_actions);
    for (int a = 0; a < cfg_.num_actions; ++a) {
      // Create separate seed for each subnet to ensure diverse initializations
      std::mt19937_64 subnet_rng(cfg_.seed + 1000 * (a + 1));
      subnets_.emplace_back(cfg_.num_inputs, cfg_.num_hidden, cfg_.alpha,
                            cfg_.momentum, cfg_.weight_decay, cfg_.clip_td_error, subnet_rng);
    }
  } else {
    // Shared hidden layer network
    init_shared_network(rng);
  }
}

void NeuralNet::init_shared_network(std::mt19937_64& rng) {
  shared_hidden_acts_.resize(cfg_.num_hidden, 0.0);

  double limit1 = std::sqrt(6.0 / (cfg_.num_inputs + cfg_.num_hidden));
  std::uniform_real_distribution<double> dist1(-limit1, limit1);

  shared_w1_.resize(cfg_.num_hidden, std::vector<double>(cfg_.num_inputs));
  v_shared_w1_.resize(cfg_.num_hidden, std::vector<double>(cfg_.num_inputs, 0.0));
  shared_b1_.resize(cfg_.num_hidden, 0.0);
  v_shared_b1_.resize(cfg_.num_hidden, 0.0);

  for (int j = 0; j < cfg_.num_hidden; ++j) {
    for (int i = 0; i < cfg_.num_inputs; ++i) {
      shared_w1_[j][i] = dist1(rng);
    }
  }

  double limit2 = std::sqrt(6.0 / (cfg_.num_hidden + cfg_.num_actions));
  std::uniform_real_distribution<double> dist2(-limit2, limit2);

  shared_w2_.resize(cfg_.num_actions, std::vector<double>(cfg_.num_hidden));
  v_shared_w2_.resize(cfg_.num_actions, std::vector<double>(cfg_.num_hidden, 0.0));
  shared_b2_.resize(cfg_.num_actions, 0.0);
  v_shared_b2_.resize(cfg_.num_actions, 0.0);

  for (int a = 0; a < cfg_.num_actions; ++a) {
    for (int j = 0; j < cfg_.num_hidden; ++j) {
      shared_w2_[a][j] = dist2(rng);
    }
  }
}

double NeuralNet::forward_shared(const std::vector<double>& x, int action) const {
  for (int j = 0; j < cfg_.num_hidden; ++j) {
    double z = shared_b1_[j];
    for (int i = 0; i < cfg_.num_inputs; ++i) {
      z += shared_w1_[j][i] * x[i];
    }
    shared_hidden_acts_[j] = sigmoid(z);
  }

  double out = shared_b2_[action];
  for (int j = 0; j < cfg_.num_hidden; ++j) {
    out += shared_w2_[action][j] * shared_hidden_acts_[j];
  }
  return out;
}

void NeuralNet::backward_shared(const std::vector<double>& x, int action, double target) {
  double pred = forward_shared(x, action);
  double td_error = target - pred;
  if (cfg_.clip_td_error) {
    td_error = std::clamp(td_error, -1.0, 1.0);
  }

  // Hidden-layer gradients first, with this action's output weights from before the update
  std::vector<double> grad_h(cfg_.num_hidden);
  for (int j = 0; j < cfg_.num_hidden; ++j) {
    grad_h[j] = td_error * shared_w2_[action][j] * d_sigmoid(shared_hidden_acts_[j]);
  }

  // Update output weights for selected action
  double delta_b2 = cfg_.alpha * td_error;
  v_shared_b2_[action] = cfg_.momentum * v_shared_b2_[action] + delta_b2;
  shared_b2_[action] += v_shared_b2_[action];

  for (int j = 0; j < cfg_.num_hidden; ++j) {
    double delta_w2 = cfg_.alpha * (td_error * shared_hidden_acts_[j] - cfg_.weight_decay * shared_w2_[action][j]);
    v_shared_w2_[action][j] = cfg_.momentum * v_shared_w2_[action][j] + delta_w2;
    shared_w2_[action][j] += v_shared_w2_[action][j];
  }

  // Update shared hidden layer
  for (int j = 0; j < cfg_.num_hidden; ++j) {
    double delta_b1 = cfg_.alpha * grad_h[j];
    v_shared_b1_[j] = cfg_.momentum * v_shared_b1_[j] + delta_b1;
    shared_b1_[j] += v_shared_b1_[j];

    for (int i = 0; i < cfg_.num_inputs; ++i) {
      double delta_w1 = cfg_.alpha * (grad_h[j] * x[i] - cfg_.weight_decay * shared_w1_[j][i]);
      v_shared_w1_[j][i] = cfg_.momentum * v_shared_w1_[j][i] + delta_w1;
      shared_w1_[j][i] += v_shared_w1_[j][i];
    }
  }
}

double NeuralNet::value(const std::vector<double>& x, int action) const {
  if (action < 0 || action >= cfg_.num_actions) {
    throw std::out_of_range("Invalid action index in NeuralNet::value");
  }
  if (!cfg_.shared_hidden) {
    return subnets_[action].forward(x);
  } else {
    return forward_shared(x, action);
  }
}

void NeuralNet::update(const std::vector<double>& x, int action, double target) {
  if (action < 0 || action >= cfg_.num_actions) {
    throw std::out_of_range("Invalid action index in NeuralNet::update");
  }
  if (!cfg_.shared_hidden) {
    subnets_[action].backward(x, target);
  } else {
    backward_shared(x, action, target);
  }
}

void NeuralNet::save(const std::string& path) const {
  std::ofstream ofs(path);
  if (!ofs) {
    throw std::runtime_error("Failed to open path for saving NeuralNet: " + path);
  }

  ofs << "# NEURAL_NET_CHECKPOINT\n";
  ofs << "num_inputs " << cfg_.num_inputs << "\n";
  ofs << "num_hidden " << cfg_.num_hidden << "\n";
  ofs << "num_actions " << cfg_.num_actions << "\n";
  ofs << "shared_hidden " << (cfg_.shared_hidden ? 1 : 0) << "\n";
  ofs << "alpha " << cfg_.alpha << "\n";
  ofs << "momentum " << cfg_.momentum << "\n";
  ofs << "weight_decay " << cfg_.weight_decay << "\n";
  ofs << "clip_td_error " << (cfg_.clip_td_error ? 1 : 0) << "\n";

  if (!cfg_.shared_hidden) {
    for (int a = 0; a < cfg_.num_actions; ++a) {
      ofs << "# SUBNET " << a << "\n";
      subnets_[a].save(ofs);
    }
  } else {
    ofs << "# SHARED_NETWORK\n";
    for (int j = 0; j < cfg_.num_hidden; ++j) {
      ofs << shared_b1_[j] << (j + 1 == cfg_.num_hidden ? "\n" : " ");
    }
    for (int j = 0; j < cfg_.num_hidden; ++j) {
      for (int i = 0; i < cfg_.num_inputs; ++i) {
        ofs << shared_w1_[j][i] << (i + 1 == cfg_.num_inputs ? "\n" : " ");
      }
    }
    for (int a = 0; a < cfg_.num_actions; ++a) {
      ofs << shared_b2_[a] << (a + 1 == cfg_.num_actions ? "\n" : " ");
    }
    for (int a = 0; a < cfg_.num_actions; ++a) {
      for (int j = 0; j < cfg_.num_hidden; ++j) {
        ofs << shared_w2_[a][j] << (j + 1 == cfg_.num_hidden ? "\n" : " ");
      }
    }
  }
}

void NeuralNet::load(const std::string& path) {
  std::ifstream ifs(path);
  if (!ifs) {
    throw std::runtime_error("Failed to open path for loading NeuralNet: " + path);
  }

  std::string tag, key;
  ifs >> tag; // # NEURAL_NET_CHECKPOINT
  int shared_int = 0, clip_int = 1;

  ifs >> key >> cfg_.num_inputs;
  ifs >> key >> cfg_.num_hidden;
  ifs >> key >> cfg_.num_actions;
  ifs >> key >> shared_int;
  cfg_.shared_hidden = (shared_int != 0);
  ifs >> key >> cfg_.alpha;
  ifs >> key >> cfg_.momentum;
  ifs >> key >> cfg_.weight_decay;
  ifs >> key >> clip_int;
  cfg_.clip_td_error = (clip_int != 0);

  if (!cfg_.shared_hidden) {
    subnets_.resize(cfg_.num_actions);
    for (int a = 0; a < cfg_.num_actions; ++a) {
      std::string marker, sub_tag;
      ifs >> marker >> sub_tag >> tag; // # SUBNET a
      subnets_[a].alpha_ = cfg_.alpha;
      subnets_[a].momentum_ = cfg_.momentum;
      subnets_[a].weight_decay_ = cfg_.weight_decay;
      subnets_[a].clip_td_error_ = cfg_.clip_td_error;
      subnets_[a].load(ifs);
    }
  } else {
    std::string marker, sub_tag;
    ifs >> marker >> sub_tag; // # SHARED_NETWORK
    shared_hidden_acts_.resize(cfg_.num_hidden, 0.0);
    shared_b1_.resize(cfg_.num_hidden);
    v_shared_b1_.assign(cfg_.num_hidden, 0.0);
    for (int j = 0; j < cfg_.num_hidden; ++j) {
      ifs >> shared_b1_[j];
    }
    shared_w1_.assign(cfg_.num_hidden, std::vector<double>(cfg_.num_inputs));
    v_shared_w1_.assign(cfg_.num_hidden, std::vector<double>(cfg_.num_inputs, 0.0));
    for (int j = 0; j < cfg_.num_hidden; ++j) {
      for (int i = 0; i < cfg_.num_inputs; ++i) {
        ifs >> shared_w1_[j][i];
      }
    }
    shared_b2_.resize(cfg_.num_actions);
    v_shared_b2_.assign(cfg_.num_actions, 0.0);
    for (int a = 0; a < cfg_.num_actions; ++a) {
      ifs >> shared_b2_[a];
    }
    shared_w2_.assign(cfg_.num_actions, std::vector<double>(cfg_.num_hidden));
    v_shared_w2_.assign(cfg_.num_actions, std::vector<double>(cfg_.num_hidden, 0.0));
    for (int a = 0; a < cfg_.num_actions; ++a) {
      for (int j = 0; j < cfg_.num_hidden; ++j) {
        ifs >> shared_w2_[a][j];
      }
    }
  }
}

}  // namespace keepaway
