// run_benchmarks.cpp  (simulator v1)
// Runs the paper's three benchmark policies (all three keepers follow the same policy),
// plus an API check that drives the External interface with random actions. The API
// check should give roughly the same possession time as the internal Random policy.
//
// Usage (from the Project folder):
//   ./build/run_benchmarks [episodes=1000] [first_seed=1] [--out DIR] [--log DIR]
//   Writes DIR/benchmarks.csv (policy, episode, seconds, end_reason); DIR defaults to results.
//   --log DIR also writes one replayable episode per policy to DIR/traj_<policy>.csv

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "sim/keepaway_env.hpp"

using namespace keepaway;

namespace {

struct Episode {
  double seconds;
  EndReason reason;
};

const char* reason_name(EndReason r) {
  switch (r) {
    case EndReason::Steal: return "steal";
    case EndReason::OutOfBounds: return "out";
    case EndReason::Truncated: return "truncated";
    default: return "none";
  }
}

// All three keepers use the same built-in controller; reset() plays the whole episode.
std::vector<Episode> run_internal(Controller c, int episodes, uint64_t seed0,
                                  const std::string& log_path) {
  Config cfg;
  cfg.keepers = {c, c, c};
  Env env(cfg);
  std::vector<Episode> out;
  for (int e = 0; e < episodes; ++e) {
    if (e == 0) env.set_trajectory_log(log_path);
    env.reset(seed0 + e);
    if (e == 0) env.set_trajectory_log("");
    out.push_back({env.episode_seconds(), env.end_reason()});
  }
  return out;
}

// All three keepers External; actions chosen at random by the caller (tests the API path).
std::vector<Episode> run_external_random(int episodes, uint64_t seed0) {
  Config cfg;
  cfg.keepers = {Controller::External, Controller::External, Controller::External};
  Env env(cfg);
  std::mt19937_64 rng(seed0 * 7919 + 17);
  std::vector<Episode> out;
  for (int e = 0; e < episodes; ++e) {
    StepResult r = env.reset(seed0 + e);
    while (!r.done) r = env.step(random_policy(rng));
    out.push_back({env.episode_seconds(), env.end_reason()});
  }
  return out;
}

void print_row(const std::string& name, const std::vector<Episode>& eps) {
  std::vector<double> s;
  int steals = 0, outs = 0, trunc = 0;
  for (const Episode& e : eps) {
    s.push_back(e.seconds);
    if (e.reason == EndReason::Steal) ++steals;
    if (e.reason == EndReason::OutOfBounds) ++outs;
    if (e.reason == EndReason::Truncated) ++trunc;
  }
  double mean = 0.0;
  for (double v : s) mean += v;
  mean /= s.size();
  double var = 0.0;
  for (double v : s) var += (v - mean) * (v - mean);
  double sd = s.size() > 1 ? std::sqrt(var / (s.size() - 1)) : 0.0;
  std::sort(s.begin(), s.end());
  double median = s[s.size() / 2];
  std::printf("%-18s mean %7.2f s  sd %6.2f  median %6.2f   steal %4d  out %4d  truncated %d\n",
              name.c_str(), mean, sd, median, steals, outs, trunc);
}

}  // namespace

int main(int argc, char** argv) {
  int episodes = 1000;
  uint64_t seed0 = 1;
  std::string log_dir;
  std::string out_dir = "results";
  int positional = 0;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--log" && i + 1 < argc) {
      log_dir = argv[++i];
    } else if (a == "--out" && i + 1 < argc) {
      out_dir = argv[++i];
    } else if (positional == 0) {
      episodes = std::atoi(argv[i]);
      ++positional;
    } else {
      seed0 = std::strtoull(argv[i], nullptr, 10);
      ++positional;
    }
  }
  std::filesystem::create_directories(out_dir);
  if (!log_dir.empty()) std::filesystem::create_directories(log_dir);
  auto log_path = [&](const std::string& name) {
    return log_dir.empty() ? std::string() : log_dir + "/traj_" + name + ".csv";
  };

  struct Row {
    std::string name;
    std::vector<Episode> eps;
  };
  std::vector<Row> rows;
  rows.push_back({"AlwaysHold", run_internal(Controller::AlwaysHold, episodes, seed0, log_path("AlwaysHold"))});
  rows.push_back({"Random", run_internal(Controller::Random, episodes, seed0, log_path("Random"))});
  rows.push_back({"HandCoded", run_internal(Controller::HandCoded, episodes, seed0, log_path("HandCoded"))});
  rows.push_back({"External+random", run_external_random(episodes, seed0)});

  std::printf("%d episodes per policy, seeds %llu..%llu\n", episodes,
              static_cast<unsigned long long>(seed0),
              static_cast<unsigned long long>(seed0 + episodes - 1));
  for (const Row& r : rows) print_row(r.name, r.eps);

  std::ofstream csv(out_dir + "/benchmarks.csv");
  csv << "policy,episode,seconds,end_reason\n";
  for (const Row& r : rows)
    for (size_t e = 0; e < r.eps.size(); ++e)
      csv << r.name << ',' << e << ',' << r.eps[e].seconds << ',' << reason_name(r.eps[e].reason)
          << '\n';
  return 0;
}
