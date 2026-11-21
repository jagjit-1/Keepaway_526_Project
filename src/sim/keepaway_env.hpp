// keepaway_env.hpp  (simulator v1)
// API between the keepaway simulator and any learning algorithm.
//
// Reference: Stone, Sutton & Singh (2001), "Reinforcement Learning for 3 vs. 2 Keepaway".
// Values marked "our choice" are NOT from the paper; the paper relied on the RoboCup
// soccer server for them.

#pragma once
#include <array>
#include <cstdint>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace keepaway
{

  // ---------------------------------------------------------------------------
  // Actions
  // ---------------------------------------------------------------------------
  // Relative to the state ordering: F1 = keeper with the ball, F2 = closer teammate,
  // F3 = the other teammate. The learner never sees absolute player ids for actions.
  enum Action : int
  {
    HOLD = 0,
    PASS_F2 = 1,
    PASS_F3 = 2
  };
  constexpr int kNumActions = 3;
  inline const std::vector<Action> possibleActions = {HOLD, PASS_F2, PASS_F3};

  // ---------------------------------------------------------------------------
  // State: the paper's 13 variables (Section 5), in this fixed order
  // ---------------------------------------------------------------------------
  // F1 = keeper with ball, F2 = other keeper closest to ball, F3 = remaining keeper,
  // D1 = taker closest to ball, D2 = other taker, C = centre of the field.
  enum Feature : int
  {
    D_F1_C = 0,     // dist(F1, C)
    D_F1_F2,        // dist(F1, F2)
    D_F1_F3,        // dist(F1, F3)
    D_F1_D1,        // dist(F1, D1)
    D_F1_D2,        // dist(F1, D2)
    D_F2_C,         // dist(F2, C)
    D_F3_C,         // dist(F3, C)
    D_D1_C,         // dist(D1, C)
    D_D2_C,         // dist(D2, C)
    MIN_D_F2_TAKER, // min(dist(F2, D1), dist(F2, D2))
    MIN_D_F3_TAKER, // min(dist(F3, D1), dist(F3, D2))
    MIN_ANG_F2,     // min(ang(F2, F1, D1), ang(F2, F1, D2)), degrees, vertex at F1
    MIN_ANG_F3,     // min(ang(F3, F1, D1), ang(F3, F1, D2)), degrees, vertex at F1
    kNumFeatures    // = 13
  };
  using FullState = std::array<double, kNumFeatures>;

  // Feature subsets. Pass one of these to select_features().
  inline const std::vector<Feature> kFullFeatures = {
      D_F1_C, D_F1_F2, D_F1_F3, D_F1_D1, D_F1_D2, D_F2_C, D_F3_C,
      D_D1_C, D_D2_C, MIN_D_F2_TAKER, MIN_D_F3_TAKER, MIN_ANG_F2, MIN_ANG_F3};
  // Reduced set (our choice): pressure on the ball-holder + openness of each passing lane.
  inline const std::vector<Feature> kReducedFeatures = {D_F1_D1, D_F1_D2, MIN_ANG_F2, MIN_ANG_F3};

  // ---------------------------------------------------------------------------
  // Configuration
  // ---------------------------------------------------------------------------
  // External = decisions are returned to the caller (the learner).
  // The others are decided inside the simulator.
  enum class Controller
  {
    External,
    HandCoded,
    AlwaysHold,
    Random
  };

  struct Config
  {
    // From the paper
    double field_size = 20.0;            // 20 m x 20 m square
    double step_seconds = 0.1;           // 100 ms per step
    double handcoded_hold_radius = 10.0; // hand-coded policy: hold if no taker within 10 m

    // Our choice
    double player_speed = 0.4;              // m per step, keepers and takers alike
    double move_noise = 0.05;               // std dev of movement noise, fraction of player_speed
    double kick_speed = 1.5;                // ball speed right after a pass, m per step
    double ball_decay = 0.94;               // ball speed multiplier per step
    double kickable_radius = 1.0;           // m
    int steal_steps = 5;                    // taker within kickable radius this many steps in a row -> takers win
    double kick_angle_noise_deg = 5.0;      // std dev of pass direction noise
    double kick_speed_noise = 0.05;         // std dev of pass speed noise, fraction of kick_speed
    double handcoded_pass_angle_deg = 30.0; // stand-in for the paper's trained pass evaluator
    double corner_inset = 2.0;              // reset positions are this far inside each corner
    int64_t max_steps = 20000;              // safety cap (2000 s) so a perfect policy cannot run forever

    // Who decides for each keeper. Start: one learner + two hand-coded teammates.
    std::array<Controller, 3> keepers = {Controller::External, Controller::HandCoded,
                                         Controller::HandCoded};
  };

  // ---------------------------------------------------------------------------
  // Decision points and step results
  // ---------------------------------------------------------------------------
  // A decision point exists only when an External keeper has the ball within kickable range.
  struct Decision
  {
    int keeper_id = -1; // 0..2: which keeper must decide (route to that keeper's learner)
    FullState state{};  // features from that keeper's point of view (it is F1)
    int64_t t = 0;      // global step count since the episode started
  };

  struct StepResult
  {
    bool done = false;      // episode over: takers won the ball, ball out of bounds, or cap hit
    bool truncated = false; // true only if it ended because t reached max_steps
    Decision next;          // valid only when done == false
    int64_t t = 0;          // global step count now (the episode length if done)
  };

  enum class EndReason
  {
    None,
    Steal,
    OutOfBounds,
    Truncated
  };

  struct Vec2
  {
    double x = 0.0, y = 0.0;
  };

  // ---------------------------------------------------------------------------
  // Environment
  // ---------------------------------------------------------------------------
  class Env
  {
  public:
    explicit Env(const Config &cfg);

    // Starts a new episode with the paper's reset (takers in one corner, keepers in the
    // other three at random, ball at a keeper) and simulates until the first External
    // decision. Returns done == true if the episode ends before any External decision
    // (always the case when no keeper is External, e.g. when running benchmarks).
    StepResult reset(uint64_t seed);

    // Applies the action for the pending decision, then simulates every other agent
    // (takers chase the ball; non-deciding keepers follow the paper's Fig. 2 structure)
    // until the next External decision or the end of the episode.
    StepResult step(Action a);

    // Possession time of the current episode in seconds: t * step_seconds.
    double episode_seconds() const;
    int64_t steps() const { return t_; }
    EndReason end_reason() const { return end_reason_; }
    const Config &config() const { return cfg_; }

    // Writes per-step positions of all 5 players and the ball to a CSV for viewer.py.
    // Pass an empty string to stop logging. Keep it off during training runs.
    void set_trajectory_log(const std::string &path);

  private:
    Config cfg_;
    std::array<Vec2, 3> keepers_{};
    std::array<Vec2, 2> takers_{};
    Vec2 ball_{}, ball_vel_{};
    int64_t t_ = 0;
    int64_t episode_ = -1;
    int steal_count_ = 0;
    bool done_ = true;
    EndReason end_reason_ = EndReason::None;

    int holder_ = -1;              // keeper with the ball this step, -1 if nobody
    std::array<int, 3> order_{};   // holder's view {F1, F2, F3} as absolute keeper ids
    Action action_ = HOLD;         // holder's action for the coming step
    bool awaiting_action_ = false; // true between returning a decision and step()
    bool holding_ = false;         // the holder is holding during the current step
    int kicker_ = -1;              // last keeper to pass
    int kick_age_ = 1000;          // steps since that pass

    std::mt19937_64 rng_;
    std::ofstream log_;

    StepResult decide_or_run();
    StepResult finished() const;
    void tick();
    void kick(int receiver);
    int find_holder() const;
    std::array<int, 3> keeper_order(int holder) const;
    FullState compute_state(const std::array<int, 3> &order) const;
    Action internal_action(Controller c, const FullState &s);
    Vec2 intercept_point(Vec2 p) const;
    int steps_to_ball(Vec2 p) const;
    int fastest_keeper_to_ball() const;
    Vec2 get_open_spot(int k, const std::vector<Vec2> &claimed) const;
    Vec2 nearest_taker(Vec2 p) const;
    void move_toward(Vec2 &p, Vec2 target);
    void resolve_contact(Vec2 prev);
    bool ball_out() const;
    void log_row();
  };

  // ---------------------------------------------------------------------------
  // Feature helpers (shared, so both approximators see identical inputs)
  // ---------------------------------------------------------------------------
  std::vector<double> select_features(const FullState &s, const std::vector<Feature> &subset);

  struct Range
  {
    double lo, hi;
  };
  // Distances lie in [0, field diagonal ~28.3 m]; distances to C in [0, ~14.1 m];
  // angles in [0, 180] degrees.
  Range feature_range(Feature f, const Config &cfg);

  // Scales each selected feature to [0, 1] using feature_range().
  std::vector<double> normalize(const std::vector<double> &x, const std::vector<Feature> &subset,
                                const Config &cfg);

  // ---------------------------------------------------------------------------
  // Benchmark policies (paper Section 3); also used internally by Controller values
  // ---------------------------------------------------------------------------
  Action random_policy(std::mt19937_64 &rng);
  Action always_hold_policy();
  // Hold if no taker within handcoded_hold_radius; otherwise pass to the teammate with the
  // wider passing-lane angle if it exceeds handcoded_pass_angle_deg; otherwise hold.
  Action hand_coded_policy(const FullState &s, const Config &cfg);

} // namespace keepaway

// ---------------------------------------------------------------------------
// How a learning loop uses this API (Q-learning, one External keeper).
// `approx` is any keepaway::Approximator from src/learn/approximator.hpp.
// ---------------------------------------------------------------------------
//   StepResult r = env.reset(seed);
//   while (!r.done) {
//     x  = normalize(select_features(r.next.state, subset), subset, cfg);
//     a  = epsilon_greedy(approx, x);
//     t0 = r.next.t;
//     r  = env.step(a);
//     k  = r.t - t0;   // steps elapsed between decisions (decisions are irregular)
//     // Paper, Section 6: reward is 0 except -1 at the end, discounted by gamma^k.
//     target = r.done ? -pow(gamma, k - 1)
//                     : pow(gamma, k) * max over a' of approx.value(x_next, a');
//     approx.update(x, a, target);
//   }
//   log(episode, env.episode_seconds());
