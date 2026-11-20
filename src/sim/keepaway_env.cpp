// keepaway_env.cpp  (simulator v1)
// Implementation of the keepaway simulator declared in keepaway_env.hpp.
//
// One step = 100 ms. Each step:
//   1. the keeper with the ball (if any) carries out its action: hold or pass;
//   2. everyone picks a movement target (takers chase the ball, keepers follow Fig. 2);
//   3. players move, then the ball moves;
//   4. we check for the ball leaving the field, players stopping a moving ball,
//      and the takers winning the ball.

#include "keepaway_env.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace keepaway {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kStopSpeed = 0.02;  // ball slower than this (m per step) counts as stopped

Vec2 operator+(Vec2 a, Vec2 b) { return {a.x + b.x, a.y + b.y}; }
Vec2 operator-(Vec2 a, Vec2 b) { return {a.x - b.x, a.y - b.y}; }
Vec2 operator*(Vec2 a, double s) { return {a.x * s, a.y * s}; }
double norm(Vec2 a) { return std::hypot(a.x, a.y); }
double dist(Vec2 a, Vec2 b) { return norm(a - b); }
Vec2 unit(Vec2 a) {
  double n = norm(a);
  return n > 1e-9 ? a * (1.0 / n) : Vec2{0.0, 0.0};
}
Vec2 rotate(Vec2 v, double deg) {
  double r = deg * kPi / 180.0, c = std::cos(r), s = std::sin(r);
  return {v.x * c - v.y * s, v.x * s + v.y * c};
}

// Angle at `vertex` between the rays vertex->a and vertex->b, in degrees [0, 180].
double angle_deg(Vec2 a, Vec2 vertex, Vec2 b) {
  Vec2 u = a - vertex, v = b - vertex;
  double nu = norm(u), nv = norm(v);
  if (nu < 1e-9 || nv < 1e-9) return 0.0;
  double c = std::clamp((u.x * v.x + u.y * v.y) / (nu * nv), -1.0, 1.0);
  return std::acos(c) * 180.0 / kPi;
}

// Closest point to q on the segment p0->p1; *s_out is how far along it is, in [0, 1].
Vec2 closest_on_segment(Vec2 p0, Vec2 p1, Vec2 q, double* s_out) {
  Vec2 d = p1 - p0;
  double len2 = d.x * d.x + d.y * d.y;
  double s = 0.0;
  if (len2 > 1e-12) s = std::clamp(((q.x - p0.x) * d.x + (q.y - p0.y) * d.y) / len2, 0.0, 1.0);
  *s_out = s;
  return p0 + d * s;
}

}  // namespace

// ===========================================================================
// Episode control
// ===========================================================================

Env::Env(const Config& cfg) : cfg_(cfg) {}

double Env::episode_seconds() const { return static_cast<double>(t_) * cfg_.step_seconds; }

StepResult Env::reset(uint64_t seed) {
  rng_.seed(seed);
  ++episode_;
  t_ = 0;
  steal_count_ = 0;
  done_ = false;
  end_reason_ = EndReason::None;
  holder_ = -1;
  awaiting_action_ = false;
  holding_ = false;
  kicker_ = -1;
  kick_age_ = 1000;

  // Paper: both takers start in one corner, each keeper in one of the other three.
  const double L = cfg_.field_size, m = cfg_.corner_inset;
  const std::array<Vec2, 4> corners = {Vec2{m, m}, Vec2{L - m, m}, Vec2{L - m, L - m},
                                       Vec2{m, L - m}};
  int taker_corner = std::uniform_int_distribution<int>(0, 3)(rng_);
  std::vector<int> keeper_corners;
  for (int c = 0; c < 4; ++c)
    if (c != taker_corner) keeper_corners.push_back(c);
  std::shuffle(keeper_corners.begin(), keeper_corners.end(), rng_);

  std::normal_distribution<double> jitter(0.0, 0.3);
  for (int k = 0; k < 3; ++k)
    keepers_[k] = corners[keeper_corners[k]] + Vec2{jitter(rng_), jitter(rng_)};
  // The two takers share a corner; nudge the second one toward the centre.
  Vec2 to_centre = unit(Vec2{L / 2, L / 2} - corners[taker_corner]);
  takers_[0] = corners[taker_corner] + Vec2{jitter(rng_), jitter(rng_)};
  takers_[1] = corners[taker_corner] + to_centre * 1.0 + Vec2{jitter(rng_), jitter(rng_)};

  // Keepers start with the ball (our choice: at a random keeper).
  int start = std::uniform_int_distribution<int>(0, 2)(rng_);
  ball_ = keepers_[start];
  ball_vel_ = {0.0, 0.0};

  log_row();
  return decide_or_run();
}

StepResult Env::step(Action a) {
  if (!awaiting_action_) throw std::logic_error("Env::step() called with no pending decision");
  awaiting_action_ = false;
  action_ = a;
  tick();
  if (done_) return finished();
  return decide_or_run();
}

// Finds who has the ball. If an External keeper has it, hands the decision to the caller.
// Otherwise internal controllers decide and the simulation keeps going.
StepResult Env::decide_or_run() {
  while (true) {
    holder_ = find_holder();
    if (holder_ >= 0) {
      order_ = keeper_order(holder_);
      FullState s = compute_state(order_);
      Controller c = cfg_.keepers[holder_];
      if (c == Controller::External) {
        awaiting_action_ = true;
        StepResult r;
        r.done = false;
        r.truncated = false;
        r.t = t_;
        r.next.keeper_id = holder_;
        r.next.state = s;
        r.next.t = t_;
        return r;
      }
      action_ = internal_action(c, s);
    }
    tick();
    if (done_) return finished();
  }
}

StepResult Env::finished() const {
  StepResult r;
  r.done = true;
  r.truncated = (end_reason_ == EndReason::Truncated);
  r.t = t_;
  return r;
}

Action Env::internal_action(Controller c, const FullState& s) {
  switch (c) {
    case Controller::HandCoded: return hand_coded_policy(s, cfg_);
    case Controller::AlwaysHold: return always_hold_policy();
    case Controller::Random: return random_policy(rng_);
    case Controller::External: break;
  }
  return HOLD;
}

// ===========================================================================
// One simulation step (100 ms)
// ===========================================================================

void Env::tick() {
  // 1. The ball-holder carries out its action for this step.
  holding_ = false;
  if (holder_ >= 0) {
    if (action_ == HOLD) {
      holding_ = true;
    } else {
      kick(order_[action_ == PASS_F2 ? 1 : 2]);
    }
  }

  // 2. Movement targets, all computed from the state at the start of the step.
  //    Takers (paper): both always go to the ball.
  std::array<Vec2, 2> taker_target;
  for (int d = 0; d < 2; ++d) taker_target[d] = intercept_point(takers_[d]);

  //    Keepers (paper, Fig. 2): if a teammate has the ball or can reach it faster -> get open;
  //    otherwise go to the ball. The holder stays put while holding.
  int chaser = holding_ ? -1 : fastest_keeper_to_ball();
  std::array<Vec2, 3> keeper_target;
  std::vector<Vec2> claimed;  // spots already chosen by teammates this step
  for (int k = 0; k < 3; ++k) {
    if (holding_ && k == holder_) {
      keeper_target[k] = keepers_[k];
    } else if (k == chaser) {
      keeper_target[k] = intercept_point(keepers_[k]);
    } else {
      keeper_target[k] = get_open_spot(k, claimed);
      claimed.push_back(keeper_target[k]);
    }
  }

  // 3. Move players.
  for (int k = 0; k < 3; ++k)
    if (!(holding_ && k == holder_)) move_toward(keepers_[k], keeper_target[k]);
  for (int d = 0; d < 2; ++d) move_toward(takers_[d], taker_target[d]);

  // 4. Move the ball.
  Vec2 prev = ball_;
  if (holding_) {
    // Paper: HoldBall keeps the ball as far from the opponents as possible.
    // We keep it half a kickable radius behind the holder, away from the nearest taker.
    // The shielded position is kept inside the field, so holding never puts the ball out.
    const double L = cfg_.field_size, margin = 0.1;
    Vec2 h = keepers_[holder_];
    Vec2 away = unit(h - nearest_taker(h));
    ball_ = h + away * (0.5 * cfg_.kickable_radius);
    ball_.x = std::clamp(ball_.x, margin, L - margin);
    ball_.y = std::clamp(ball_.y, margin, L - margin);
    ball_vel_ = {0.0, 0.0};
  } else {
    ball_ = ball_ + ball_vel_;
    ball_vel_ = ball_vel_ * cfg_.ball_decay;
    if (norm(ball_vel_) < kStopSpeed) ball_vel_ = {0.0, 0.0};
  }
  ++t_;
  ++kick_age_;

  // 5. Ball out of the square: the keepers lose it.
  if (ball_out()) {
    done_ = true;
    end_reason_ = EndReason::OutOfBounds;
    log_row();
    return;
  }

  // 6. A moving ball is stopped by the first player whose kickable radius it passes through.
  if (!holding_ && dist(prev, ball_) > 1e-9) resolve_contact(prev);

  // 7. Takers win once one of them stays within kickable radius for steal_steps in a row.
  bool taker_near = false;
  for (const Vec2& d : takers_)
    if (dist(d, ball_) <= cfg_.kickable_radius) taker_near = true;
  steal_count_ = taker_near ? steal_count_ + 1 : 0;

  if (steal_count_ >= cfg_.steal_steps) {
    done_ = true;
    end_reason_ = EndReason::Steal;
  } else if (t_ >= cfg_.max_steps) {
    done_ = true;
    end_reason_ = EndReason::Truncated;
  }
  log_row();
}

void Env::kick(int receiver) {
  // Paper: PassBall(f) kicks the ball directly toward forward f. Direction and speed are noisy.
  std::normal_distribution<double> angle_noise(0.0, cfg_.kick_angle_noise_deg);
  std::normal_distribution<double> speed_noise(0.0, cfg_.kick_speed_noise);
  Vec2 dir = unit(keepers_[receiver] - ball_);
  if (norm(dir) < 1e-9) dir = {1.0, 0.0};
  dir = rotate(dir, angle_noise(rng_));
  double speed = cfg_.kick_speed * std::max(0.1, 1.0 + speed_noise(rng_));
  ball_vel_ = dir * speed;
  kicker_ = holder_;
  kick_age_ = 0;
}

// Stops a ball that moved from `prev` to ball_ at the first player it passed within reach of.
void Env::resolve_contact(Vec2 prev) {
  double best_s = 2.0;
  Vec2 best_point{};
  bool found = false;
  auto consider = [&](Vec2 player) {
    double s = 0.0;
    Vec2 c = closest_on_segment(prev, ball_, player, &s);
    if (dist(c, player) <= cfg_.kickable_radius && s < best_s) {
      best_s = s;
      best_point = c;
      found = true;
    }
  };
  for (int k = 0; k < 3; ++k) {
    if (k == kicker_ && kick_age_ < 3) continue;  // the passer cannot stop its own pass
    consider(keepers_[k]);
  }
  for (const Vec2& d : takers_) consider(d);
  if (found) {
    ball_ = best_point;
    ball_vel_ = {0.0, 0.0};
  }
}

bool Env::ball_out() const {
  const double L = cfg_.field_size;
  return ball_.x < 0.0 || ball_.x > L || ball_.y < 0.0 || ball_.y > L;
}

// ===========================================================================
// Who has the ball, and the state from the holder's point of view
// ===========================================================================

int Env::find_holder() const {
  int best = -1;
  double best_d = cfg_.kickable_radius;
  for (int k = 0; k < 3; ++k) {
    if (k == kicker_ && kick_age_ < 3 && norm(ball_vel_) > 0.0) continue;
    double d = dist(keepers_[k], ball_);
    if (d <= best_d) {
      best_d = d;
      best = k;
    }
  }
  return best;
}

// {F1, F2, F3}: the holder, then the other two keepers sorted by distance to the ball.
std::array<int, 3> Env::keeper_order(int holder) const {
  std::array<int, 3> o{holder, -1, -1};
  int i = 1;
  for (int k = 0; k < 3; ++k)
    if (k != holder) o[i++] = k;
  if (dist(keepers_[o[2]], ball_) < dist(keepers_[o[1]], ball_)) std::swap(o[1], o[2]);
  return o;
}

FullState Env::compute_state(const std::array<int, 3>& order) const {
  const double L = cfg_.field_size;
  const Vec2 C{L / 2, L / 2};
  const Vec2 F1 = keepers_[order[0]], F2 = keepers_[order[1]], F3 = keepers_[order[2]];
  Vec2 D1 = takers_[0], D2 = takers_[1];
  if (dist(D2, ball_) < dist(D1, ball_)) std::swap(D1, D2);

  FullState s{};
  s[D_F1_C] = dist(F1, C);
  s[D_F1_F2] = dist(F1, F2);
  s[D_F1_F3] = dist(F1, F3);
  s[D_F1_D1] = dist(F1, D1);
  s[D_F1_D2] = dist(F1, D2);
  s[D_F2_C] = dist(F2, C);
  s[D_F3_C] = dist(F3, C);
  s[D_D1_C] = dist(D1, C);
  s[D_D2_C] = dist(D2, C);
  s[MIN_D_F2_TAKER] = std::min(dist(F2, D1), dist(F2, D2));
  s[MIN_D_F3_TAKER] = std::min(dist(F3, D1), dist(F3, D2));
  s[MIN_ANG_F2] = std::min(angle_deg(F2, F1, D1), angle_deg(F2, F1, D2));
  s[MIN_ANG_F3] = std::min(angle_deg(F3, F1, D1), angle_deg(F3, F1, D2));
  return s;
}

// ===========================================================================
// Skills: GoToBall, GetOpen, movement
// ===========================================================================

// Paper: GoToBall intercepts a moving ball or goes straight to a stationary one.
// We roll the ball forward (ignoring noise) and return the first point the player can reach.
Vec2 Env::intercept_point(Vec2 p) const {
  Vec2 b = ball_, v = ball_vel_;
  const double reach = 0.5 * cfg_.kickable_radius;
  for (int n = 0; n <= 300; ++n) {
    if (dist(p, b) <= n * cfg_.player_speed + reach) return b;
    if (norm(v) < kStopSpeed) return b;
    b = b + v;
    v = v * cfg_.ball_decay;
  }
  return b;
}

int Env::steps_to_ball(Vec2 p) const {
  Vec2 b = ball_, v = ball_vel_;
  const double reach = 0.5 * cfg_.kickable_radius;
  for (int n = 0; n <= 300; ++n) {
    double gap = dist(p, b) - n * cfg_.player_speed - reach;
    if (gap <= 0.0) return n;
    if (norm(v) < kStopSpeed) return n + static_cast<int>(std::ceil(gap / cfg_.player_speed));
    b = b + v;
    v = v * cfg_.ball_decay;
  }
  return 1000000;
}

int Env::fastest_keeper_to_ball() const {
  int best = -1, best_n = 1 << 30;
  for (int k = 0; k < 3; ++k) {
    if (k == kicker_ && kick_age_ < 3) continue;  // the passer does not chase its own pass
    int n = steps_to_ball(keepers_[k]);
    if (n < best_n) {
      best_n = n;
      best = k;
    }
  }
  return best;
}

// Stand-in for the paper's GetOpen (which used SPAR): score a 5x5 grid of spots by
// distance from the takers, openness of the passing lane from the ball, crowding by
// teammates and travel distance, and head for the best one. The grid spans 15%-85% of
// the field (3 m to 17 m) so keepers do not wait right on the edge. All weights are our choice.
Vec2 Env::get_open_spot(int k, const std::vector<Vec2>& claimed) const {
  const double L = cfg_.field_size;
  Vec2 best = keepers_[k];
  double best_score = -1e18;
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      Vec2 p{L * (0.15 + 0.175 * i), L * (0.15 + 0.175 * j)};
      double safety = std::min(dist(p, takers_[0]), dist(p, takers_[1]));
      double lane = std::min(angle_deg(p, ball_, takers_[0]), angle_deg(p, ball_, takers_[1]));
      double crowd = 0.0;
      for (int o = 0; o < 3; ++o)
        if (o != k && dist(p, keepers_[o]) < 5.0) crowd += 5.0 - dist(p, keepers_[o]);
      for (const Vec2& c : claimed)
        if (dist(p, c) < 5.0) crowd += 5.0 - dist(p, c);
      double score = std::min(safety, 10.0) + lane / 10.0 - 2.0 * crowd -
                     0.3 * dist(p, keepers_[k]);
      if (score > best_score) {
        best_score = score;
        best = p;
      }
    }
  }
  return best;
}

Vec2 Env::nearest_taker(Vec2 p) const {
  return dist(p, takers_[0]) <= dist(p, takers_[1]) ? takers_[0] : takers_[1];
}

void Env::move_toward(Vec2& p, Vec2 target) {
  Vec2 d = target - p;
  double n = norm(d);
  double len = std::min(cfg_.player_speed, n);
  if (len > 1e-9) {
    p = p + d * (len / n);
    std::normal_distribution<double> noise(0.0, cfg_.move_noise * cfg_.player_speed);
    p = p + Vec2{noise(rng_), noise(rng_)};
  }
  const double L = cfg_.field_size;
  p.x = std::clamp(p.x, 0.0, L);
  p.y = std::clamp(p.y, 0.0, L);
}

// ===========================================================================
// Trajectory log for viewer.py
// ===========================================================================

void Env::set_trajectory_log(const std::string& path) {
  if (log_.is_open()) log_.close();
  if (path.empty()) return;
  log_.open(path);
  if (!log_) throw std::runtime_error("cannot open trajectory log: " + path);
  log_ << "episode,t,k0x,k0y,k1x,k1y,k2x,k2y,d0x,d0y,d1x,d1y,bx,by,holder,steal\n";
}

void Env::log_row() {
  if (!log_.is_open()) return;
  log_ << episode_ << ',' << t_;
  for (const Vec2& k : keepers_) log_ << ',' << k.x << ',' << k.y;
  for (const Vec2& d : takers_) log_ << ',' << d.x << ',' << d.y;
  log_ << ',' << ball_.x << ',' << ball_.y << ',' << holder_ << ',' << steal_count_ << '\n';
}

// ===========================================================================
// Feature helpers
// ===========================================================================

std::vector<double> select_features(const FullState& s, const std::vector<Feature>& subset) {
  std::vector<double> x;
  x.reserve(subset.size());
  for (Feature f : subset) x.push_back(s[f]);
  return x;
}

Range feature_range(Feature f, const Config& cfg) {
  const double diag = cfg.field_size * std::sqrt(2.0);
  switch (f) {
    case MIN_ANG_F2:
    case MIN_ANG_F3: return {0.0, 180.0};
    case D_F1_C:
    case D_F2_C:
    case D_F3_C:
    case D_D1_C:
    case D_D2_C: return {0.0, diag / 2.0};
    default: return {0.0, diag};
  }
}

std::vector<double> normalize(const std::vector<double>& x, const std::vector<Feature>& subset,
                              const Config& cfg) {
  std::vector<double> out(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    Range r = feature_range(subset[i], cfg);
    out[i] = std::clamp((x[i] - r.lo) / (r.hi - r.lo), 0.0, 1.0);
  }
  return out;
}

// ===========================================================================
// Benchmark policies
// ===========================================================================

Action random_policy(std::mt19937_64& rng) {
  return static_cast<Action>(std::uniform_int_distribution<int>(0, kNumActions - 1)(rng));
}

Action always_hold_policy() { return HOLD; }

Action hand_coded_policy(const FullState& s, const Config& cfg) {
  // Paper: if no taker is within 10 m, hold.
  if (std::min(s[D_F1_D1], s[D_F1_D2]) > cfg.handcoded_hold_radius) return HOLD;
  // Paper: otherwise pass if a teammate is better placed and the pass is likely to succeed.
  // Our stand-in: pass along the more open lane, if it is open enough.
  Action best = s[MIN_ANG_F2] >= s[MIN_ANG_F3] ? PASS_F2 : PASS_F3;
  double best_angle = std::max(s[MIN_ANG_F2], s[MIN_ANG_F3]);
  return best_angle >= cfg.handcoded_pass_angle_deg ? best : HOLD;
}

}  // namespace keepaway
