// Author : Ruben Rehal | Date : November 2025
//
// sensor_sanity.cpp — Plausibility checker inspired by ISO-26262 diagnostic
// requirements. Validates every sensor frame against known physical limits
// (accelerometer range, gyro range, sonar distance, optical intensity) and
// checks inter-frame timestamp gaps to detect stale or stuck sensors. If any
// check fails, the main loop skips control output as a safety measure.

#include "orion/sensor_sanity.h"

#include <cmath>  // std::fabs for absolute-value range checks

namespace orion {

// Evaluates all three sensor frames and returns a SensorHealth struct with
// one boolean per modality plus a timing-synchronisation flag. The main loop
// AND's all four booleans to decide whether to trust the inference output.
SensorHealth SensorSanityChecker::Evaluate(const ImuFrame& imu,
                                           const UltrasonicFrame& sonar,
                                           const OpticalFrame& optic) {
    SensorHealth health{};

    // ── IMU range check ──────────────────────────────────────────────────────
    // Accelerometer: X and Y must be < 8 g (typical MEMS full-scale range).
    // Z allows up to 20 g because gravity (1 g) plus high vertical acceleration
    // can legitimately exceed the ±8 g range used for lateral axes.
    const bool imu_range =
        std::fabs(imu.accel[0]) < 8.0f && std::fabs(imu.accel[1]) < 8.0f && std::fabs(imu.accel[2]) < 20.0f;

    // Gyroscope: all three axes must be < 20 rad/s (≈1146 °/s) — well within
    // common MEMS gyro full-scale limits.
    const bool gyro_range =
        std::fabs(imu.gyro[0]) < 20.0f && std::fabs(imu.gyro[1]) < 20.0f && std::fabs(imu.gyro[2]) < 20.0f;

    // IMU is OK only if both accel and gyro are within range
    health.imu_ok = imu_range && gyro_range;

    // ── Ultrasonic range check ───────────────────────────────────────────────
    // Distance must be within the physical operating range of a typical ultrasonic
    // sensor: 5 cm (near-field dead zone) to 5 m (max reliable range).
    // Confidence must exceed 0.2 (below this the echo is too weak to trust).
    health.ultrasonic_ok = sonar.distance_m > 0.05f && sonar.distance_m < 5.0f && sonar.confidence > 0.2f;

    // ── Optical range check ──────────────────────────────────────────────────
    // Normalised intensity must be in [0, 2]. Values outside this indicate a
    // saturated or disconnected sensor.
    health.optical_ok = optic.intensity >= 0.0f && optic.intensity <= 2.0f;

    // ── Timestamp synchronisation check ──────────────────────────────────────
    // Verifies that no sensor has "gone silent". IMU must update within 20 ms
    // (allows missing up to 4 samples at 200 Hz). Sonar and optical are allowed
    // up to 40 ms (they may run at lower rates).
    if (initialized_) {
        // Compute microsecond deltas from the previous frames
        const uint32_t imu_delta = imu.timestamp_us - last_imu_.timestamp_us;
        const uint32_t sonar_delta = sonar.timestamp_us - last_sonar_.timestamp_us;
        const uint32_t optic_delta = optic.timestamp_us - last_optic_.timestamp_us;

        // All three must be within their respective deadlines
        health.timestamp_sync = imu_delta < 20000 && sonar_delta < 40000 && optic_delta < 40000;
    } else {
        // First call: no previous frame to compare against, skip timestamp check
        initialized_ = true;
    }

    // Store the current frames for next iteration's delta computation
    last_imu_ = imu;
    last_sonar_ = sonar;
    last_optic_ = optic;

    return health;
}

}
