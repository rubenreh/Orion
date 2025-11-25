// Author : Ruben Rehal | Date : November 2025
#include "orion/imu_filter.h"

namespace orion {
ImuFilter::ImuFilter(float alpha_accel, float alpha_gyro)
    : alpha_accel_(alpha_accel), alpha_gyro_(alpha_gyro) {}

ImuFrame ImuFilter::Filter(const ImuFrame& raw) {
    if (!initialized_) {
        state_ = raw;
        initialized_ = true;
        return state_;
    }
    for (size_t i = 0; i < 3; ++i) {
        state_.accel[i] = alpha_accel_ * raw.accel[i] + (1.0f - alpha_accel_) * state_.accel[i];
        state_.gyro[i] = alpha_gyro_ * raw.gyro[i] + (1.0f - alpha_gyro_) * state_.gyro[i];
    }
    state_.temperature_c = 0.5f * raw.temperature_c + 0.5f * state_.temperature_c;
    state_.timestamp_us = raw.timestamp_us;
    return state_;
}
}
