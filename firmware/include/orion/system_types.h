// Author : Ruben Rehal | Date : November 2025
//
// system_types.h — Core data structures shared across every layer of the Orion
// stack. These POD (plain-old-data) structs define the sensor frame formats,
// the fused feature vector that feeds the neural network, and the inference
// output that drives control decisions.
#pragma once

#include <array>    // std::array for fixed-size sensor and feature buffers
#include <cstdint>  // uint32_t for microsecond timestamps

namespace orion {

// Raw reading from a 6-axis IMU (Inertial Measurement Unit).
// Captures linear acceleration, angular velocity, die temperature, and the
// microsecond timestamp at which the sample was latched.
struct ImuFrame {
    std::array<float, 3> accel;    // Linear acceleration [x, y, z] in g
    std::array<float, 3> gyro;     // Angular velocity   [x, y, z] in rad/s
    float temperature_c;           // Sensor die temperature in degrees Celsius
    uint32_t timestamp_us;         // Monotonic timestamp in microseconds
};

// Raw reading from an ultrasonic range-finder (sonar).
// Provides a distance estimate together with the sensor's own confidence
// metric (e.g., derived from echo signal-to-noise ratio).
struct UltrasonicFrame {
    float distance_m;              // Measured distance to target in metres
    float confidence;              // Sensor self-reported confidence [0, 1]
    uint32_t timestamp_us;         // Monotonic timestamp in microseconds
};

// Raw reading from an optical (light/reflectance) sensor.
// Provides an intensity measurement and the frame-to-frame delta, which can
// indicate motion or occlusion events.
struct OpticalFrame {
    float intensity;               // Measured light intensity (normalised)
    float delta;                   // Change from previous frame
    uint32_t timestamp_us;         // Monotonic timestamp in microseconds
};

// The fused 8-element feature vector produced by FusionPipeline and consumed
// by the InferenceEngine. Each element maps to a specific physical quantity
// (see fusion_pipeline.cpp for the mapping).
struct FusedFrame {
    std::array<float, 8> features; // 8-D feature vector for the neural network
    uint32_t timestamp_us;         // Timestamp of the fused sample
};

// Output produced by the InferenceEngine after running the neural-network
// model on a FusedFrame. The main loop uses these two values to decide
// whether to act, warn, or fall back to a safe state.
struct InferenceOutput {
    float anomaly_score;           // Predicted anomaly level [0, 1]
    float confidence;              // Model's self-reported confidence [0, 1]
};

}
