// Author : Ruben Rehal | Date : November 2025
//
// fusion_pipeline.cpp — Sensor fusion implementation. Takes raw IMU, ultrasonic,
// and optical frames, runs each through its dedicated EMA low-pass filter, and
// packs the filtered values into an 8-element FusedFrame feature vector that
// serves as the input to the neural network. This is the bridge between
// heterogeneous physical sensors and a fixed-dimension ML input.

#include "orion/fusion_pipeline.h"

#include <algorithm>  // (available for future clamp/minmax use)
#include <cmath>      // std::sqrt for Euclidean magnitude

namespace orion {

// Initialises the IMU filter with α_accel=0.25 (heavy smoothing for noisy
// MEMS accelerometer) and α_gyro=0.35 (lighter smoothing for faster gyro
// tracking). Ultrasonic and optical filters use their own defaults.
FusionPipeline::FusionPipeline()
    : imu_filter_(0.25f, 0.35f) {}

// Processes one set of raw sensor frames and returns the fused 8-D feature
// vector that the inference engine consumes. This is called once per loop
// iteration by ModelRunner::Step().
FusedFrame FusionPipeline::Process(const ImuFrame& imu_raw,
                                   const UltrasonicFrame& ultrasonics_raw,
                                   const OpticalFrame& optical_raw) {
    // ── Step 1: Filter each modality independently ───────────────────────────
    const ImuFrame imu = imu_filter_.Filter(imu_raw);                   // EMA-filtered IMU
    const UltrasonicFrame sonar = ultrasonic_filter_.Filter(ultrasonics_raw);  // EMA-filtered sonar
    const OpticalFrame optic = optical_filter_.Smooth(optical_raw);     // EMA-filtered optical

    // ── Step 2: Pack the 8-element feature vector ────────────────────────────
    FusedFrame fused{};

    // [0] Accelerometer magnitude — sqrt(ax² + ay² + az²).
    //     Rotation-invariant measure of total acceleration (≈1 g when stationary).
    fused.features[0] = Magnitude(imu.accel);

    // [1] Gyroscope magnitude — sqrt(gx² + gy² + gz²).
    //     Captures total angular velocity regardless of rotation axis.
    fused.features[1] = Magnitude(imu.gyro);

    // [2] Raw Z-axis acceleration — gravity-aligned axis.
    //     Detects vertical shocks and tilt changes independently of features[0].
    fused.features[2] = imu.accel[2];

    // [3] Sonar distance in metres — primary range measurement.
    fused.features[3] = sonar.distance_m;

    // [4] Sonar confidence [0,1] — the sensor's own reliability estimate.
    //     Gives the model a self-diagnostic input for the range measurement.
    fused.features[4] = sonar.confidence;

    // [5] Optical intensity (normalised) — ambient light / reflectance level.
    fused.features[5] = optic.intensity;

    // [6] Optical delta — frame-to-frame intensity change.
    //     Detects motion, occlusion, or rapid lighting transitions.
    fused.features[6] = optic.delta;

    // [7] Inter-frame dt in seconds — time since the previous Process() call.
    //     Lets the model learn that stale data (large dt) should reduce confidence.
    fused.features[7] = static_cast<float>(imu.timestamp_us - last_timestamp_us_) * 1e-6f;

    // Record the timestamp for next iteration's dt calculation
    fused.timestamp_us = imu.timestamp_us;
    last_timestamp_us_ = imu.timestamp_us;

    return fused;
}

// Computes the Euclidean (L2) norm of a 3-element vector.
// Used to collapse 3-axis accel/gyro into a single scalar that is
// invariant to the sensor's orientation.
float FusionPipeline::Magnitude(const std::array<float, 3>& v) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

}
