// Author : Ruben Rehal | Date : November 2025
#pragma once

#include <array>
#include <cstdint>

namespace orion {
struct ImuFrame {
    std::array<float, 3> accel;
    std::array<float, 3> gyro;
    float temperature_c;
    uint32_t timestamp_us;
};

struct UltrasonicFrame {
    float distance_m;
    float confidence;
    uint32_t timestamp_us;
};

struct OpticalFrame {
    float intensity;
    float delta;
    uint32_t timestamp_us;
};

struct FusedFrame {
    std::array<float, 8> features;
    uint32_t timestamp_us;
};

struct InferenceOutput {
    float anomaly_score;
    float confidence;
};
}
