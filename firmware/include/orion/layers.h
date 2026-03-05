// Author : Ruben Rehal | Date : November 2025
//
// layers.h — Dense (fully-connected) layer abstraction for the int8 inference
// engine. Each DenseLayer holds a pointer to its quantised weights/biases and
// an activation function, and can execute one forward pass (Invoke) that
// performs: int8 dot-product → dequantise → activate → requantise.
#pragma once

#include <cstddef>      // size_t for dimensions
#include <cstdint>      // int8_t, int32_t for quantised types
#include <functional>   // std::function for pluggable activation functions
#include <vector>       // std::vector for variable-size activation buffer

namespace orion {

// Static configuration for a single dense layer, including pointers to
// weight/bias arrays that live in model_data.cpp and the per-layer quantisation
// scales needed for dequantisation and requantisation.
struct DenseLayerConfig {
    size_t input_dim;            // Number of input neurons
    size_t output_dim;           // Number of output neurons
    const int8_t* weights;       // Row-major weight matrix [output_dim × input_dim]
    const int32_t* bias;         // Bias vector [output_dim], int32 to match accumulator
    float input_scale;           // Quantisation scale of the layer's input activations
    float weight_scale;          // Quantisation scale of the weight matrix
    float output_scale;          // Quantisation scale for the layer's output activations
};

// Executes one fully-connected layer in quantised int8 arithmetic.
// The Invoke method is the hot path called once per layer per inference.
class DenseLayer {
  public:
    // Constructs the layer from its config and a pluggable activation function
    // (e.g., ReluFromLut for hidden layers, TanhApprox for the output layer).
    DenseLayer(const DenseLayerConfig& cfg, std::function<float(float)> activation);

    // Runs the forward pass: for each output neuron, computes the int8 dot
    // product with the input, adds bias, dequantises to float, applies the
    // activation, and requantises to int8 for the next layer.
    void Invoke(const std::vector<int8_t>& input, std::vector<int8_t>& output) const;

    // Returns the float activations from the most recent Invoke(), used by the
    // inference engine to read the final output layer's logits before
    // post-processing into anomaly_score and confidence.
    std::vector<float> LastActivations() const { return last_activations_; }

  private:
    DenseLayerConfig cfg_;                       // Layer geometry, weight pointers, scales
    std::function<float(float)> activation_;     // Pluggable activation function
    mutable std::vector<float> last_activations_;  // Cached float outputs from last Invoke()
};

}
