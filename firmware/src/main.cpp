// Author : Ruben Rehal | Date : November 2025
#include <chrono>
#include <cstdio>
#include <thread>

#include "orion/config.h"
#include "orion/imu_driver.h"
#include "orion/model_runner.h"
#include "orion/optical_driver.h"
#include "orion/power.h"
#include "orion/sensor_sanity.h"
#include "orion/ultrasonic_driver.h"
#include "orion/watchdog.h"

namespace {
uint32_t MicrosSinceStart() {
    using namespace std::chrono;
    static const auto start = steady_clock::now();
    return static_cast<uint32_t>(
        duration_cast<microseconds>(steady_clock::now() - start).count());
}
}

int main() {
    orion::MockImuDriver imu;
    orion::MockUltrasonicDriver sonar;
    orion::MockOpticalDriver optic;
    orion::ModelRunner runner;
    orion::Watchdog watchdog;
    orion::PowerMonitor power;

    watchdog.Configure(static_cast<uint32_t>(1e6 / orion::kControlLoopHz * 2));

    for (size_t iteration = 0; iteration < 400; ++iteration) {
        const uint32_t loop_start = MicrosSinceStart();

        const orion::ImuFrame imu_frame = imu.Read();
        const orion::UltrasonicFrame sonar_frame = sonar.Read();
        const orion::OpticalFrame optic_frame = optic.Read();

        const orion::InferenceOutput output = runner.Step(imu_frame, sonar_frame, optic_frame);
        const auto health = runner.LastHealth();
        if (!health.imu_ok || !health.ultrasonic_ok || !health.optical_ok || !health.timestamp_sync) {
            std::puts("[ORION] Sensor anomaly detected, skipping control output.");
            continue;
        }
        power.Update(3.25f, loop_start);

        if (power.IsBrownout() || output.confidence < orion::kConfidenceFallback) {
            std::puts("[ORION] Entering fallback mode.");
        } else {
            std::printf("[ORION] score=%.3f confidence=%.3f\n", output.anomaly_score, output.confidence);
        }

        watchdog.Kick(loop_start);
        if (watchdog.Expired(MicrosSinceStart())) {
            std::puts("[ORION] Watchdog expired, system reset required.");
            break;
        }

        const uint32_t loop_time_us = MicrosSinceStart() - loop_start;
        if (loop_time_us < static_cast<uint32_t>(orion::kLoopPeriodMs * 1000.0f)) {
            std::this_thread::sleep_for(
                std::chrono::microseconds(static_cast<int64_t>(orion::kLoopPeriodMs * 1000.0f) - loop_time_us));
        }
    }
    return 0;
}

