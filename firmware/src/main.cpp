// Author : Ruben Rehal | Date : November 2025
//
// main.cpp — Entry point for the Orion firmware. Runs a deterministic 200 Hz
// control loop that reads three sensor modalities (IMU, ultrasonic, optical),
// fuses them into a feature vector, runs a quantised neural-network inference,
// and applies safety checks (sensor sanity, watchdog, brownout) before acting
// on the result. The loop demonstrates the full edge-AI pipeline from raw
// sensor data to confidence-gated anomaly output.

#include <chrono>   // steady_clock, duration_cast for high-resolution timing
#include <cstdio>   // std::puts, std::printf for lightweight console output
#include <thread>   // std::this_thread::sleep_for for loop-budget sleep

#include "orion/config.h"              // kControlLoopHz, kLoopPeriodMs, kConfidenceFallback
#include "orion/imu_driver.h"          // MockImuDriver
#include "orion/model_runner.h"        // ModelRunner — fuses sensors + runs inference
#include "orion/optical_driver.h"      // MockOpticalDriver
#include "orion/power.h"              // PowerMonitor — brownout detection
#include "orion/sensor_sanity.h"      // SensorHealth struct (used via ModelRunner)
#include "orion/ultrasonic_driver.h"  // MockUltrasonicDriver
#include "orion/watchdog.h"           // Watchdog — timing-fault detection

namespace {

// Returns the number of microseconds elapsed since the very first call.
// Uses std::chrono::steady_clock which is monotonic (never jumps backwards).
// The static `start` is initialised once and persists for the process lifetime.
uint32_t MicrosSinceStart() {
    using namespace std::chrono;
    // Capture the start time on the first call (static initialisation)
    static const auto start = steady_clock::now();
    // Compute elapsed microseconds and truncate to 32 bits (wraps after ~71 min)
    return static_cast<uint32_t>(
        duration_cast<microseconds>(steady_clock::now() - start).count());
}

}  // anonymous namespace

// ── Main control loop ────────────────────────────────────────────────────────
// Instantiates all hardware drivers and safety monitors, then enters a fixed-
// rate 200 Hz loop for 400 iterations (2 seconds of simulated operation).
int main() {
    // Instantiate mock sensor drivers (swap for real hardware drivers on target)
    orion::MockImuDriver imu;             // Simulated 6-axis IMU
    orion::MockUltrasonicDriver sonar;    // Simulated sonar range-finder
    orion::MockOpticalDriver optic;       // Simulated optical/light sensor

    // ModelRunner wires together fusion + sanity + inference in a single Step()
    orion::ModelRunner runner;

    // Software watchdog — triggers if the loop overruns its timing budget
    orion::Watchdog watchdog;

    // Supply-rail voltage monitor — flags brownout below 2.8 V
    orion::PowerMonitor power;

    // Configure the watchdog timeout to 2× the loop period (10 000 µs).
    // This tolerates one missed deadline but catches two consecutive misses.
    watchdog.Configure(static_cast<uint32_t>(1e6 / orion::kControlLoopHz * 2));

    // ── 200 Hz deterministic control loop ────────────────────────────────────
    for (size_t iteration = 0; iteration < 400; ++iteration) {
        // Record the timestamp at the top of the loop for budget accounting
        const uint32_t loop_start = MicrosSinceStart();

        // ── 1. Sensor acquisition ────────────────────────────────────────────
        // Read one sample from each sensor. In production these would be
        // DMA-driven interrupt callbacks; here they are synchronous mock reads.
        const orion::ImuFrame imu_frame = imu.Read();
        const orion::UltrasonicFrame sonar_frame = sonar.Read();
        const orion::OpticalFrame optic_frame = optic.Read();

        // ── 2. Fuse + sanity-check + infer ───────────────────────────────────
        // ModelRunner::Step() filters the raw frames, packs the 8-D feature
        // vector, validates plausibility, and runs the int8 MLP.
        const orion::InferenceOutput output = runner.Step(imu_frame, sonar_frame, optic_frame);

        // ── 3. Sensor health gate ────────────────────────────────────────────
        // If any sensor is out of spec or timestamps are stale, skip the entire
        // control output. This is the first line of defence (ISO-26262-style).
        const auto health = runner.LastHealth();
        if (!health.imu_ok || !health.ultrasonic_ok || !health.optical_ok || !health.timestamp_sync) {
            std::puts("[ORION] Sensor anomaly detected, skipping control output.");
            continue;  // Jump to the next iteration without actuating
        }

        // ── 4. Power-rail check ──────────────────────────────────────────────
        // Feed the current supply voltage (3.25 V mock) and timestamp to the
        // power monitor so it can update its brownout state.
        power.Update(3.25f, loop_start);

        // ── 5. Confidence-gated output ───────────────────────────────────────
        // If the supply is in brownout OR the model's self-reported confidence
        // is below the fallback threshold (0.2), enter safe mode.
        if (power.IsBrownout() || output.confidence < orion::kConfidenceFallback) {
            std::puts("[ORION] Entering fallback mode.");
        } else {
            // Normal operation: print the anomaly score and confidence
            std::printf("[ORION] score=%.3f confidence=%.3f\n", output.anomaly_score, output.confidence);
        }

        // ── 6. Watchdog service ──────────────────────────────────────────────
        // Kick the watchdog to prove this iteration completed on time
        watchdog.Kick(loop_start);

        // Check whether the watchdog has expired (loop took too long)
        if (watchdog.Expired(MicrosSinceStart())) {
            std::puts("[ORION] Watchdog expired, system reset required.");
            break;  // Exit the loop — on real hardware this would trigger a reset
        }

        // ── 7. Budget sleep ──────────────────────────────────────────────────
        // Compute how many microseconds remain in this 5 ms (5 000 µs) period
        // and sleep for the difference to maintain a steady 200 Hz rate.
        const uint32_t loop_time_us = MicrosSinceStart() - loop_start;
        if (loop_time_us < static_cast<uint32_t>(orion::kLoopPeriodMs * 1000.0f)) {
            std::this_thread::sleep_for(
                std::chrono::microseconds(static_cast<int64_t>(orion::kLoopPeriodMs * 1000.0f) - loop_time_us));
        }
    }

    return 0;
}
