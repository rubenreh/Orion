// Author : Ruben Rehal | Date : November 2025
//
// model_data.h — Declares the neural-network architecture constants and the
// Model struct that bundles layer configs. The actual weight arrays and the
// factory function GetCalibratedModel() live in model_data.cpp. Separating
// declaration from definition keeps compile times short and lets the linker
// place the weight arrays in flash.
#pragma once

#include <array>      // (used by model_data.cpp for weight arrays)
#include <cstddef>    // size_t for dimension constants
#include <cstdint>    // int8_t, int32_t for weight types

#include "layers.h"   // DenseLayerConfig

namespace orion {
namespace model {

// MLP architecture dimensions: 8 inputs → 12 hidden → 6 hidden → 2 outputs.
constexpr size_t kInputDim = 8;     // Fused feature vector length
constexpr size_t kHidden1 = 12;     // First hidden layer width
constexpr size_t kHidden2 = 6;      // Second hidden layer width
constexpr size_t kOutputDim = 2;    // Output: [anomaly_score, confidence]

// Bundles the three DenseLayerConfig structs that together describe the full
// model. Passed to InferenceEngine::LoadModel().
struct Model {
    DenseLayerConfig layer1;  // 8 → 12  (ReLU)
    DenseLayerConfig layer2;  // 12 → 6  (ReLU)
    DenseLayerConfig layer3;  // 6 → 2   (Tanh)
};

// Returns a reference to the singleton calibrated model with pre-quantised
// int8 weights, int32 biases, and per-layer scales ready for inference.
const Model& GetCalibratedModel();

}  // namespace model
}  // namespace orion
