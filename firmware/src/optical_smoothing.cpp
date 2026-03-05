// Author : Ruben Rehal | Date : November 2025
//
// optical_smoothing.cpp — EMA low-pass filter for optical sensor readings.
// Uses α=0.4 (moderate smoothing) because optical sensors are generally cleaner
// than sonar but still benefit from jitter removal. Both intensity and delta
// are smoothed with the same alpha for simplicity.

#include "orion/optical_smoothing.h"

namespace orion {

// Default constructor — state_ is zero-initialised by the struct default.
OpticalSmoothing::OpticalSmoothing() = default;

// Smooths one raw optical frame and returns the filtered result.
// On the first call, the raw frame is adopted as-is (cold-start).
OpticalFrame OpticalSmoothing::Smooth(const OpticalFrame& raw) {
    if (!initialized_) {
        // Cold-start: use the first sample directly as the initial state
        state_ = raw;
        initialized_ = true;
    } else {
        // Moderate EMA: 40% new data, 60% retained from previous filtered value
        const float alpha = 0.4f;

        // Smooth intensity: filtered = 0.4·new + 0.6·old
        state_.intensity = alpha * raw.intensity + (1.0f - alpha) * state_.intensity;

        // Smooth delta with the same alpha
        state_.delta = alpha * raw.delta + (1.0f - alpha) * state_.delta;

        // Carry the latest timestamp forward
        state_.timestamp_us = raw.timestamp_us;
    }
    return state_;
}

}
