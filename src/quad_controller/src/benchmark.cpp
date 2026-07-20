#include "quad_controller/policy_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Compares two serialised engines built from the same network: how fast each
// one runs, and how far their outputs drift apart.
//
// Both are loaded through IPolicyEngine, so the timing and comparison code has
// no idea which precision it is holding. That is the point of the interface.
//
//   ./benchmark <engine_a> <engine_b> [iterations]
//
// Latency here is measured with nothing else on the GPU. Numbers collected
// while the simulator is rendering are considerably worse, and that gap is
// itself worth reporting.
// ---------------------------------------------------------------------------

namespace {

constexpr int kWarmup = 200;

struct Stats {
    double mean;
    double p50;
    double p99;
    double max;
};

Stats summarise(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    const auto n = samples.size();
    double sum = 0.0;
    for (double v : samples) sum += v;
    return Stats{sum / static_cast<double>(n),
                 samples[n / 2],
                 samples[static_cast<std::size_t>(static_cast<double>(n) * 0.99)],
                 samples.back()};
}

// Observations shaped like the real thing: small body velocities, joint offsets
// near zero, and a terrain scan inside the clip range the policy was trained on.
std::vector<quad::Observation> makeObservations(std::size_t count, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> vel(-0.6f, 0.6f);
    std::uniform_real_distribution<float> joint(-0.3f, 0.3f);
    std::uniform_real_distribution<float> scan(-0.4f, 0.4f);

    std::vector<quad::Observation> out(count);
    for (auto& obs : out) {
        for (std::size_t i = 0; i < 6; ++i) obs[i] = vel(rng);
        obs[6] = 0.0f;
        obs[7] = 0.0f;
        obs[8] = -1.0f;
        obs[9] = 0.5f;
        obs[10] = 0.0f;
        obs[11] = 0.0f;
        for (std::size_t i = 12; i < quad::BASE_OBS_DIM; ++i) obs[i] = joint(rng);
        for (std::size_t i = quad::BASE_OBS_DIM; i < quad::OBS_DIM; ++i) obs[i] = scan(rng);
    }
    return out;
}

std::vector<double> timeEngine(quad::IPolicyEngine& engine,
                               const std::vector<quad::Observation>& inputs,
                               std::vector<quad::JointArray>& outputs) {
    using Clock = std::chrono::steady_clock;

    for (int i = 0; i < kWarmup; ++i) {
        (void)engine.infer(inputs[static_cast<std::size_t>(i) % inputs.size()]);
    }

    std::vector<double> micros;
    micros.reserve(inputs.size());
    outputs.clear();
    outputs.reserve(inputs.size());

    for (const auto& obs : inputs) {
        const auto start = Clock::now();
        const quad::JointArray action = engine.infer(obs);
        const auto end = Clock::now();

        micros.push_back(std::chrono::duration<double, std::micro>(end - start).count());
        outputs.push_back(action);
    }
    return micros;
}

void printStats(const std::string& label, const Stats& s) {
    std::cout << std::left << std::setw(28) << label << std::right << std::fixed
              << std::setprecision(2)
              << "  mean " << std::setw(8) << s.mean
              << "  p50 " << std::setw(8) << s.p50
              << "  p99 " << std::setw(8) << s.p99
              << "  max " << std::setw(8) << s.max << "  us\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <engine_a> <engine_b> [iterations]\n";
        return 1;
    }

    const std::string path_a = argv[1];
    const std::string path_b = argv[2];
    const std::size_t iterations =
        (argc > 3) ? static_cast<std::size_t>(std::atoi(argv[3])) : 5000;

    try {
        const std::vector<quad::Observation> inputs = makeObservations(iterations, 42);

        quad::TensorRtPolicy engine_a(path_a);
        quad::TensorRtPolicy engine_b(path_b);

        std::cout << "A: " << path_a << "\n   " << engine_a.describe() << "\n";
        std::cout << "B: " << path_b << "\n   " << engine_b.describe() << "\n";
        std::cout << "iterations: " << iterations << " (plus " << kWarmup
                  << " warmup)\n\n";

        std::vector<quad::JointArray> out_a;
        std::vector<quad::JointArray> out_b;

        const Stats stats_a = summarise(timeEngine(engine_a, inputs, out_a));
        const Stats stats_b = summarise(timeEngine(engine_b, inputs, out_b));

        printStats("A latency", stats_a);
        printStats("B latency", stats_b);
        std::cout << "\nspeedup (mean A/B): " << std::setprecision(3)
                  << stats_a.mean / stats_b.mean << "x\n";

        // How far apart do the two engines land on identical inputs?
        double max_abs = 0.0;
        double sum_abs = 0.0;
        double max_rel = 0.0;
        std::size_t count = 0;

        for (std::size_t i = 0; i < out_a.size(); ++i) {
            for (std::size_t j = 0; j < quad::ACT_DIM; ++j) {
                const double diff = std::abs(static_cast<double>(out_a[i][j]) -
                                             static_cast<double>(out_b[i][j]));
                const double scale = std::max(1e-6, std::abs(static_cast<double>(out_a[i][j])));
                max_abs = std::max(max_abs, diff);
                max_rel = std::max(max_rel, diff / scale);
                sum_abs += diff;
                ++count;
            }
        }

        std::cout << std::setprecision(6)
                  << "\nagreement over " << count << " action values:\n"
                  << "  mean abs difference: " << sum_abs / static_cast<double>(count) << "\n"
                  << "  max abs difference:  " << max_abs << "\n"
                  << "  max rel difference:  " << max_rel << "\n";

        // Actions are scaled by 0.5 into joint radians, so translate the error
        // into the units that actually reach the robot.
        std::cout << "  max joint error:     " << max_abs * quad::ACTION_SCALE
                  << " rad ("
                  << max_abs * quad::ACTION_SCALE * 180.0 / M_PI << " deg)\n";

    } catch (const std::exception& e) {
        std::cerr << "benchmark failed: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
