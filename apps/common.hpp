#pragma once

#include <vector>
#include "sim/keepaway_env.hpp"

using namespace keepaway;

struct Episode
{
    double seconds;
    EndReason reason;
};

struct Row
{
    std::string name;
    std::vector<Episode> eps;
};

inline const char *reason_name(EndReason r)
{
    switch (r)
    {
    case EndReason::Steal:
        return "steal";
    case EndReason::OutOfBounds:
        return "out";
    case EndReason::Truncated:
        return "truncated";
    default:
        return "none";
    }
}

inline void print_row(const std::string &name, const std::vector<Episode> &eps)
{
    std::vector<double> s;
    int steals = 0, outs = 0, trunc = 0;
    for (const Episode &e : eps)
    {
        s.push_back(e.seconds);
        if (e.reason == EndReason::Steal)
            ++steals;
        if (e.reason == EndReason::OutOfBounds)
            ++outs;
        if (e.reason == EndReason::Truncated)
            ++trunc;
    }
    double mean = 0.0;
    for (double v : s)
        mean += v;
    mean /= s.size();
    double var = 0.0;
    for (double v : s)
        var += (v - mean) * (v - mean);
    double sd = s.size() > 1 ? std::sqrt(var / (s.size() - 1)) : 0.0;
    std::sort(s.begin(), s.end());
    double median = s[s.size() / 2];
    std::printf("%-18s mean %7.2f s  sd %6.2f  median %6.2f   steal %4d  out %4d  truncated %d\n",
                name.c_str(), mean, sd, median, steals, outs, trunc);
}

inline void save_csv(const std::vector<Row> &rows, const std::string &out_dir)
{
    std::ofstream csv(out_dir + "/benchmarks.csv");
    csv << "policy,episode,seconds,end_reason\n";
    for (const Row &r : rows)
        for (size_t e = 0; e < r.eps.size(); ++e)
            csv << r.name << ',' << e << ',' << r.eps[e].seconds << ',' << reason_name(r.eps[e].reason)
                << '\n';
}