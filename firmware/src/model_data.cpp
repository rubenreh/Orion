// Author : Ruben Rehal | Date : November 2025
//
// model_data.cpp — Pre-quantised int8 weight arrays, int32 biases, and per-layer
// quantisation scales for the Orion 3-layer MLP (8→12→6→2). These values were
// produced by the Python training pipeline (ml/training_pipeline.py) using
// post-training quantisation: float weights are divided by a scale factor,
// rounded, and clamped to [-128, 127]. The scales are chosen so that division
// by the weight scale (0.03125 = 2^-5) can be implemented as a single-cycle
// bit-shift on the MCU.
//
// Total model size: 180 weights (180 B) + 20 biases (80 B int32) ≈ 256 B.

#include "orion/model_data.h"

#include <array>  // std::array for constexpr weight storage

#include "orion/activations.h"  // (included for consistency with the model build)

namespace orion {
namespace model {
namespace {

// ── Quantisation scales ──────────────────────────────────────────────────────
// kInputScale: each int8 step in the input represents 0.02 float units.
//   Representable range: [-2.56, +2.54] — covers all fused feature magnitudes.
constexpr float kInputScale = 0.02f;

// kHidden1Scale: output scale for the first hidden layer (12 neurons).
constexpr float kHidden1Scale = 0.04f;

// kHidden2Scale: output scale for the second hidden layer (6 neurons).
constexpr float kHidden2Scale = 0.05f;

// ── Layer 1 weights: 8 inputs × 12 outputs = 96 int8 values ─────────────────
// Stored in row-major order: row o contains the weights for output neuron o.
constexpr std::array<int8_t, kInputDim * kHidden1> kLayer1Weights = {
    -3,  4,  2, -2,  5,  6, -5,  3,   // Output neuron 0 weights (8 inputs)
     1, -4,  3,  6, -2, -6,  5,  2,   // Output neuron 1
     3,  1, -1,  4, -5,  4,  2, -3,   // Output neuron 2
     0,  6,  5, -4,  3, -2, -1,  4,   // Output neuron 3
    -4,  2, -3,  5,  6, -1,  1,  0,   // Output neuron 4
     5, -6,  4,  2, -3,  3, -2,  1,   // Output neuron 5
    -2,  1,  6, -5,  4, -3,  2,  5,   // Output neuron 6
     4, -5,  1,  3,  2,  6, -4, -2,   // Output neuron 7
    -5,  3, -4,  2,  1, -2,  6,  5,   // Output neuron 8
     2, -3,  5, -6,  4,  1, -1,  2,   // Output neuron 9
     3,  5, -2,  1, -4,  2,  6, -5,   // Output neuron 10
    -1,  2,  4, -3,  5, -6,  3,  2};  // Output neuron 11

// Layer 1 biases: one int32 per output neuron (12 total).
// Stored as int32 to match the dot-product accumulator type.
constexpr std::array<int32_t, kHidden1> kLayer1Bias = {
    12, -18, 5, 7, -6, 10, -9, 4, 3, -2, 6, -4};

// ── Layer 2 weights: 12 inputs × 6 outputs = 72 int8 values ─────────────────
constexpr std::array<int8_t, kHidden1 * kHidden2> kLayer2Weights = {
     4, -3,  2, -1,  5,  3,   // Output neuron 0 weights (12 inputs)
    -5,  4, -2,  3, -1,  2,   // Output neuron 1
     3,  5, -4,  2, -3,  1,   // Output neuron 2
    -2,  1,  5, -4,  3, -5,   // Output neuron 3
     5, -2,  3,  4, -1,  2,   // Output neuron 4
    -4,  3, -5,  1,  2, -3,   // Output neuron 5
     2,  4, -1,  5, -3,  2,   // Output neuron 6
    -3,  2,  4, -2,  5, -1,   // Output neuron 7
     4, -5,  2, -3,  1,  3,   // Output neuron 8
    -1,  3, -2,  4, -4,  5,   // Output neuron 9
     2, -1,  5,  3, -2,  1,   // Output neuron 10
    -5,  4, -3,  2,  3, -2};  // Output neuron 11

// Layer 2 biases: one int32 per output neuron (6 total)
constexpr std::array<int32_t, kHidden2> kLayer2Bias = {6, -4, 3, -2, 5, -3};

// ── Layer 3 weights: 6 inputs × 2 outputs = 12 int8 values ──────────────────
constexpr std::array<int8_t, kHidden2 * kOutputDim> kLayer3Weights = {
     4, -3,   // Output neuron 0 (anomaly_score)
    -5,  2,
     3, -4,
    -2,  5,
     5, -1,
    -3,  4};  // Output neuron 1 (confidence)

// Layer 3 biases: one int32 per output neuron (2 total)
constexpr std::array<int32_t, kOutputDim> kLayer3Bias = {2, -2};

// Constructs the full Model struct by wiring each layer's config to its
// weight/bias arrays and assigning quantisation scales.
Model BuildModel() {
    // Layer 1 config: 8→12, input_scale=0.02, weight_scale=0.03125, output_scale=0.04
    static DenseLayerConfig l1{
        kInputDim,                   // input_dim  = 8
        kHidden1,                    // output_dim = 12
        kLayer1Weights.data(),       // Pointer to the int8 weight array
        kLayer1Bias.data(),          // Pointer to the int32 bias array
        kInputScale,                 // input_scale  = 0.02
        0.03125f,                    // weight_scale = 1/32 (power-of-2 for fast bit-shift)
        kHidden1Scale};              // output_scale = 0.04

    // Layer 2 config: 12→6
    static DenseLayerConfig l2{
        kHidden1,                    // input_dim  = 12
        kHidden2,                    // output_dim = 6
        kLayer2Weights.data(),       // Pointer to the int8 weight array
        kLayer2Bias.data(),          // Pointer to the int32 bias array
        kHidden1Scale,               // input_scale  = 0.04 (matches layer 1 output)
        0.03125f,                    // weight_scale = 1/32
        kHidden2Scale};              // output_scale = 0.05

    // Layer 3 config: 6→2 (output layer)
    static DenseLayerConfig l3{
        kHidden2,                    // input_dim  = 6
        kOutputDim,                  // output_dim = 2
        kLayer3Weights.data(),       // Pointer to the int8 weight array
        kLayer3Bias.data(),          // Pointer to the int32 bias array
        kHidden2Scale,               // input_scale  = 0.05 (matches layer 2 output)
        0.03125f,                    // weight_scale = 1/32
        0.02f};                      // output_scale = 0.02

    // Bundle all three configs into a Model struct
    static Model model{l1, l2, l3};
    return model;
}

}  // anonymous namespace

// Returns a const reference to the singleton calibrated model. The model is
// built once on the first call (lazy initialisation) and reused thereafter.
const Model& GetCalibratedModel() {
    static Model model = BuildModel();
    return model;
}

}  // namespace model
}  // namespace orion
