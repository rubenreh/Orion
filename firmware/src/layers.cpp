// Author : Ruben Rehal | Date : November 2025
//
// layers.cpp — Dense (fully-connected) layer forward-pass implementation.
// This is the hot path of the inference engine: for each output neuron it
// computes an int8 dot product, dequantises to float, applies the activation
// function, and requantises back to int8 for the next layer's input. The
// design mirrors TensorRT's per-tensor quantisation scheme.

#include "orion/layers.h"

#include <algorithm>  // std::clamp for int8 saturation
#include <cmath>      // std::round for requantisation rounding

#include "orion/quant_ops.h"  // quant::Dot, quant::Dequantize

namespace orion {

// Stores the layer configuration and activation function. The activation is
// passed as a std::function so hidden layers can use ReluFromLut while the
// output layer uses TanhApprox — no virtual dispatch overhead.
DenseLayer::DenseLayer(const DenseLayerConfig& cfg, std::function<float(float)> activation)
    : cfg_(cfg), activation_(std::move(activation)) {}

// Executes the full forward pass for one dense layer. This is called once per
// layer per inference (3 times total per loop iteration).
//
// Arithmetic flow for each output neuron o:
//   1. acc = bias[o]                                (int32)
//   2. acc += Σ input[i] * weight[o][i]             (int8 × int8 → int32)
//   3. deq = acc * (input_scale * weight_scale)     (int32 → float)
//   4. activated = activation(deq)                  (float → float)
//   5. output[o] = clamp(round(activated / output_scale), -128, 127)  (float → int8)
void DenseLayer::Invoke(const std::vector<int8_t>& input, std::vector<int8_t>& output) const {
    // Ensure the output buffer is correctly sized for this layer
    if (output.size() != cfg_.output_dim) {
        output.assign(cfg_.output_dim, 0);
    }

    // Clear the float activation cache for this invocation
    last_activations_.assign(cfg_.output_dim, 0.0f);

    // Iterate over every output neuron
    for (size_t o = 0; o < cfg_.output_dim; ++o) {
        // Start the accumulator with the int32 bias for this neuron (or 0 if no bias)
        const int32_t* bias_ptr = cfg_.bias;
        int32_t acc = bias_ptr ? bias_ptr[o] : 0;

        // Pointer to the weight row for output neuron o (row-major layout)
        const int8_t* weight_row = cfg_.weights + o * cfg_.input_dim;

        // Compute int8 × int8 dot product, accumulated into int32.
        // This is the performance-critical inner loop.
        acc += quant::Dot(input.data(), weight_row, cfg_.input_dim);

        // Dequantise: convert the int32 accumulator back to float by
        // multiplying by the product of input and weight scales.
        const float deq = quant::Dequantize(acc, cfg_.input_scale * cfg_.weight_scale);

        // Apply the activation function (ReLU LUT for hidden layers, Tanh for output)
        const float activated = activation_(deq);

        // Cache the float activation for post-processing (output layer logits)
        last_activations_[o] = activated;

        // Requantise: convert the float activation back to int8 for the next
        // layer's input. Divide by output_scale, round, and clamp to [-128, 127].
        const float requant = activated / cfg_.output_scale;
        const int32_t rounded = static_cast<int32_t>(std::round(requant));
        output[o] = static_cast<int8_t>(std::clamp(rounded, -128, 127));
    }
}

}
