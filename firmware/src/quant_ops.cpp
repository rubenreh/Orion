// Author : Ruben Rehal | Date : November 2025
//
// quant_ops.cpp — Low-level quantisation primitives that form the arithmetic
// backbone of the int8 inference engine. These functions convert between float
// and int8, compute integer-only dot products (the hot inner loop), and batch-
// requantise activation buffers. On Cortex-M7, the Dot() loop maps to SMLAD
// (signed dual multiply-accumulate) instructions for maximum throughput.

#include "orion/quant_ops.h"

#include <algorithm>  // std::clamp for saturation to [-128, 127]
#include <cmath>      // std::round for nearest-integer rounding

namespace orion {
namespace quant {

// Converts a single float value to int8 using affine quantisation.
//   q = clamp(round(value / scale), -128, 127)
// Each int8 step represents `scale` units of the original float.
// With scale=0.02, the representable range is [-2.56, +2.54].
int8_t Quantize(float value, float scale) {
    // Divide by scale to map the float into integer bins
    const int32_t scaled = static_cast<int32_t>(std::round(value / scale));
    // Saturate to the int8 range to prevent wrap-around
    return static_cast<int8_t>(std::clamp(scaled, -128, 127));
}

// Converts an int32 accumulator back to float after a dot product.
// The scale_product is (input_scale × weight_scale), which undoes the
// combined scaling introduced by the quantised dot product.
float Dequantize(int32_t value, float scale_product) {
    return static_cast<float>(value) * scale_product;
}

// Integer-only dot product: the performance-critical inner loop.
// Multiplies int8 inputs by int8 weights and accumulates into int32.
// Using int32 prevents overflow: worst case |127 × 127 × length| fits
// comfortably in int32 for any reasonable layer width.
int32_t Dot(const int8_t* input, const int8_t* weights, size_t length) {
    int32_t acc = 0;  // 32-bit accumulator to hold the running sum
    for (size_t i = 0; i < length; ++i) {
        // Widen both operands to int32 before multiplying to avoid int8 overflow,
        // then accumulate the product.
        acc += static_cast<int32_t>(input[i]) * static_cast<int32_t>(weights[i]);
    }
    return acc;
}

// Batch-quantises a float buffer into an int8 buffer using a single scale.
// Useful for converting an entire layer's activations in one pass.
void RequantizeBuffer(const float* input,
                      size_t length,
                      float scale,
                      int8_t* output) {
    for (size_t i = 0; i < length; ++i) {
        // Reuse the scalar Quantize function for each element
        output[i] = Quantize(input[i], scale);
    }
}

}  // namespace quant
}  // namespace orion
