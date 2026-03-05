// Author : Ruben Rehal | Date : November 2025
//
// ultrasonic_merge.cpp — EMA low-pass filter for ultrasonic (sonar) readings.
// Sonar sensors are prone to multipath reflections and spurious echo returns,
// so this filter uses a heavy smoothing factor (α=0.2) to reject outliers.
// Confidence is averaged 50/50 because it already represents a statistical
// measure from the sensor's own processing.

#include "orion/ultrasonic_merge.h"

namespace orion {

// Default constructor — state_ is zero-initialised by the struct default.
UltrasonicMerge::UltrasonicMerge() = default;

// Filters one raw sonar frame and returns the smoothed result.
// On the first call, the raw frame is adopted as-is (cold-start).
UltrasonicFrame UltrasonicMerge::Filter(const UltrasonicFrame& raw) {
    if (!initialized_) {
        // Cold-start: use the first sample directly as the initial state
        state_ = raw;
        initialized_ = true;
    } else {
        // EMA smoothing factor — 0.2 means 80% of the previous value is retained,
        // providing heavy smoothing to reject sonar multipath spikes.
        const float alpha = 0.2f;

        // Smooth distance: filtered = 0.2·new + 0.8·old
        state_.distance_m = alpha * raw.distance_m + (1.0f - alpha) * state_.distance_m;

        // Average confidence 50/50 between old and new readings
        state_.confidence = 0.5f * (raw.confidence + state_.confidence);

        // Carry the latest timestamp forward
        state_.timestamp_us = raw.timestamp_us;
    }
    return state_;
}

}
