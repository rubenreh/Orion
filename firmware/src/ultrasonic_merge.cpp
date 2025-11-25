// Author : Ruben Rehal | Date : November 2025
#include "orion/ultrasonic_merge.h"

namespace orion {
UltrasonicMerge::UltrasonicMerge() = default;

UltrasonicFrame UltrasonicMerge::Filter(const UltrasonicFrame& raw) {
    if (!initialized_) {
        state_ = raw;
        initialized_ = true;
    } else {
        const float alpha = 0.2f;
        state_.distance_m = alpha * raw.distance_m + (1.0f - alpha) * state_.distance_m;
        state_.confidence = 0.5f * (raw.confidence + state_.confidence);
        state_.timestamp_us = raw.timestamp_us;
    }
    return state_;
}
}
