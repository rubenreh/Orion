// Author : Ruben Rehal | Date : November 2025
//
// inference_engine.cpp — Top-level neural-network runtime. Loads a quantised
// Model, converts a FusedFrame's 8 float features into int8, and executes the
// three dense layers in sequence using a ping-pong buffer pattern that minimises
// peak memory usage. After the final layer, the raw tanh logits are post-
// processed into an anomaly_score and confidence in the range [0, 1].
// This is the Orion equivalent of a TensorRT execution context.

#include "orion/inference_engine.h"

#include <algorithm>  // std::clamp for output post-processing
#include <cmath>      // (included for general numeric support)

#include "orion/activations.h"  // activations::ReluFromLut, activations::TanhApprox

namespace orion {

// Pre-allocates the two ping-pong arena buffers to the widest hidden layer
// so that no heap allocation occurs during Run(). arena_a_ is sized for the
// first hidden layer (12), arena_b_ for the second (6).
InferenceEngine::InferenceEngine() {
    arena_a_.reserve(model::kHidden1);  // Reserve space for 12 int8 values
    arena_b_.reserve(model::kHidden2);  // Reserve space for 6 int8 values
}

// Stores a pointer to the model and builds the three DenseLayer objects, each
// with the correct activation function:
//   Layer 1 (8→12):  ReLU via lookup table (fast, branch-free)
//   Layer 2 (12→6):  ReLU via lookup table
//   Layer 3 (6→2):   Tanh approximation (output needs [-1, +1] range)
void InferenceEngine::LoadModel(const model::Model& model) {
    model_ = &model;                     // Store a non-owning pointer to the model
    layers_.clear();                     // Discard any previously loaded layers
    layers_.emplace_back(model.layer1, activations::ReluFromLut);   // Hidden layer 1
    layers_.emplace_back(model.layer2, activations::ReluFromLut);   // Hidden layer 2
    layers_.emplace_back(model.layer3, activations::TanhApprox);    // Output layer
}

// Converts the 8 float features in a FusedFrame to int8 using the global
// input quantisation scale (0.02). Each float is divided by 0.02, rounded,
// and clamped to [-128, 127].
std::vector<int8_t> InferenceEngine::QuantizeInput(const FusedFrame& fused) const {
    std::vector<int8_t> input(model::kInputDim, 0);  // Allocate 8-element int8 vector
    for (size_t i = 0; i < model::kInputDim; ++i) {
        // Quantise each feature: e.g., accel magnitude 1.0 / 0.02 = 50 → int8(50)
        input[i] = quant::Quantize(fused.features[i], 0.02f);
    }
    return input;
}

// Runs full forward inference on a fused sensor frame and returns the
// post-processed anomaly_score and confidence, both clamped to [0, 1].
//
// Data flow:
//   FusedFrame → QuantizeInput → Layer1 → Layer2 → Layer3 → post-process
//
// The ping-pong pattern alternates buffer_a and buffer_b as input/output
// between layers, so only two buffers are ever alive at once.
InferenceOutput InferenceEngine::Run(const FusedFrame& fused) {
    // Lazy model loading if Run() is called before explicit LoadModel()
    if (!model_) {
        LoadModel(model::GetCalibratedModel());
    }

    // ── Step 1: Quantise the 8 float features to int8 ───────────────────────
    std::vector<int8_t> buffer_a = QuantizeInput(fused);

    // ── Step 2: Layer 1 (8→12, ReLU) ────────────────────────────────────────
    std::vector<int8_t> buffer_b(model::kHidden1);  // Allocate output for 12 neurons
    layers_[0].Invoke(buffer_a, buffer_b);            // Execute layer 1

    // Swap: layer 1's output (buffer_b) becomes layer 2's input (buffer_a)
    buffer_a.assign(buffer_b.begin(), buffer_b.end());
    // Resize buffer_b for layer 2's output (6 neurons)
    buffer_b.assign(model::kHidden2, 0);

    // ── Step 3: Layer 2 (12→6, ReLU) ────────────────────────────────────────
    layers_[1].Invoke(buffer_a, buffer_b);            // Execute layer 2

    // Swap again: layer 2's output becomes layer 3's input
    buffer_a.assign(buffer_b.begin(), buffer_b.end());
    // Resize buffer_b for the output layer (2 neurons)
    buffer_b.assign(model::kOutputDim, 0);

    // ── Step 4: Layer 3 (6→2, Tanh) ─────────────────────────────────────────
    layers_[2].Invoke(buffer_a, buffer_b);            // Execute output layer

    // ── Step 5: Post-process raw logits into [0, 1] outputs ─────────────────
    // Retrieve the float activations from the output layer (before requantisation)
    const auto& logits = layers_.back().LastActivations();

    // Map tanh output (range ≈ [-1, +1]) to [0, 1]:
    //   result = 0.5 + logit × 0.1
    // The 0.1 scaling compresses the output, making predictions less extreme.
    const float anomaly = std::clamp(0.5f + logits[0] * 0.1f, 0.0f, 1.0f);
    const float confidence = std::clamp(0.5f + logits[1] * 0.1f, 0.0f, 1.0f);

    return InferenceOutput{anomaly, confidence};
}

}
