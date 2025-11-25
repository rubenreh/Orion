// Author : Ruben Rehal | Date : November 2025
#include "orion/optical_smoothing.h"

namespace orion {
OpticalSmoothing::OpticalSmoothing() = default;

OpticalFrame OpticalSmoothing::Smooth(const OpticalFrame& raw) {
    if (!initialized_) {
        state_ = raw;
        initialized_ = true;
    } else {
        const float alpha = 0.4f;
        state_.intensity = alpha * raw.intensity + (1.0f - alpha) * state_.intensity;
        state_.delta = alpha * raw.delta + (1.0f - alpha) * state_.delta;
        state_.timestamp_us = raw.timestamp_us;
    }
    return state_;
}
}
