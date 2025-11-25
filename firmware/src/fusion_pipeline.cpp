// Author : Ruben Rehal | Date : November 2025
#include "orion/fusion_pipeline.h"

#include <algorithm>
#include <cmath>

namespace orion {
FusionPipeline::FusionPipeline()
    : imu_filter_(0.25f, 0.35f) {}

FusedFrame FusionPipeline::Process(const ImuFrame& imu_raw,
                                   const UltrasonicFrame& ultrasonics_raw,
                                   const OpticalFrame& optical_raw) {
    const ImuFrame imu = imu_filter_.Filter(imu_raw);
    const UltrasonicFrame sonar = ultrasonic_filter_.Filter(ultrasonics_raw);
    const OpticalFrame optic = optical_filter_.Smooth(optical_raw);

    FusedFrame fused{};
    fused.features[0] = Magnitude(imu.accel);
    fused.features[1] = Magnitude(imu.gyro);
    fused.features[2] = imu.accel[2];
    fused.features[3] = sonar.distance_m;
    fused.features[4] = sonar.confidence;
    fused.features[5] = optic.intensity;
    fused.features[6] = optic.delta;
    fused.features[7] = static_cast<float>(imu.timestamp_us - last_timestamp_us_) * 1e-6f;
    fused.timestamp_us = imu.timestamp_us;
    last_timestamp_us_ = imu.timestamp_us;
    return fused;
}

float FusionPipeline::Magnitude(const std::array<float, 3>& v) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}
}

