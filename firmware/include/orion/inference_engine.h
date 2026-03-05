// Author : Ruben Rehal | Date : November 2025
//
// inference_engine.h — Top-level neural-network runtime that loads a quantised
// Model, converts a FusedFrame into int8 input, and executes the three dense
// layers in sequence to produce an InferenceOutput (anomaly_score, confidence).
// This is the Orion equivalent of a TensorRT execution context.
#pragma once

#include <array>    // (included transitively)
#include <vector>   // std::vector for dynamic int8 tensor buffers

#include "layers.h"       // DenseLayer — the individual layer executor
#include "model_data.h"   // model::Model, dimension constants
#include "system_types.h" // FusedFrame, InferenceOutput

namespace orion {

// Manages model loading, input quantisation, layer-by-layer execution, and
// output post-processing. Uses a ping-pong buffer pattern (arena_a / arena_b)
// to minimise peak memory usage during inference.
class InferenceEngine {
  public:
    // Pre-allocates arena buffers to avoid heap allocation during inference.
    InferenceEngine();

    // Stores a reference to the model and instantiates the three DenseLayer
    // objects with the appropriate activation functions (ReLU, ReLU, Tanh).
    void LoadModel(const model::Model& model);

    // Runs full forward inference on a fused feature vector:
    //   1. Quantise float features to int8
    //   2. Execute layer 1 (8→12, ReLU)
    //   3. Execute layer 2 (12→6, ReLU)
    //   4. Execute layer 3 (6→2, Tanh)
    //   5. Post-process logits into anomaly_score and confidence [0, 1]
    InferenceOutput Run(const FusedFrame& fused);

  private:
    std::vector<DenseLayer> layers_;        // The three dense layers
    std::vector<int8_t> arena_a_;           // Ping-pong buffer A
    std::vector<int8_t> arena_b_;           // Ping-pong buffer B
    const model::Model* model_ = nullptr;   // Pointer to the loaded model (not owned)

    // Converts the 8 float features to int8 using the global input scale (0.02).
    std::vector<int8_t> QuantizeInput(const FusedFrame& fused) const;
};

}
