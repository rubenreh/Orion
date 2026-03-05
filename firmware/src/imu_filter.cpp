// Author : Ruben Rehal | Date : November 2025
//
// imu_filter.cpp — Exponential Moving Average (EMA) filter for IMU data.
// Smooths noisy accelerometer and gyroscope readings independently using the
// formula: filtered = α·raw + (1−α)·previous. A lower α means heavier
// smoothing (more noise rejection, more latency). Temperature is averaged
// with a fixed α=0.5 since it changes slowly.

#include "orion/imu_filter.h"

namespace orion {

// Stores the per-modality smoothing factors provided by the caller
// (FusionPipeline passes α_accel=0.25, α_gyro=0.35).
ImuFilter::ImuFilter(float alpha_accel, float alpha_gyro)
    : alpha_accel_(alpha_accel), alpha_gyro_(alpha_gyro) {}

// Filters one raw IMU frame and returns the smoothed result.
// On the very first call the raw frame is copied directly into state_ so
// that the filter starts from a real measurement rather than from zero.
ImuFrame ImuFilter::Filter(const ImuFrame& raw) {
    if (!initialized_) {
        // Cold-start: adopt the first sample as the initial state
        state_ = raw;
        initialized_ = true;
        return state_;
    }

    // Apply EMA independently to each of the three accelerometer axes
    for (size_t i = 0; i < 3; ++i) {
        // new_filtered = α * raw + (1-α) * old_filtered
        state_.accel[i] = alpha_accel_ * raw.accel[i] + (1.0f - alpha_accel_) * state_.accel[i];
        // Same formula for gyroscope but with a different (higher) alpha
        state_.gyro[i] = alpha_gyro_ * raw.gyro[i] + (1.0f - alpha_gyro_) * state_.gyro[i];
    }

    // Temperature changes slowly, so use a simple 50/50 average (α=0.5)
    state_.temperature_c = 0.5f * raw.temperature_c + 0.5f * state_.temperature_c;

    // Always carry the latest timestamp so downstream code sees fresh timing
    state_.timestamp_us = raw.timestamp_us;

    return state_;
}

}
