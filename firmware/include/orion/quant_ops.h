// Author : Ruben Rehal | Date : November 2025
//
// quant_ops.h — Low-level quantization primitives used by the inference engine.
// These functions convert between float and int8 representations, compute
// integer-only dot products, and requantize activation buffers. They form the
// arithmetic core of the TensorRT-style int8 inference pipeline, enabling
// efficient neural-network execution on MCUs without a hardware FPU.
#pragma once

#include <cstddef>   // size_t for buffer lengths
#include <cstdint>   // int8_t, int32_t for quantised types

namespace orion {
namespace quant {

// Converts a float value to int8 by dividing by `scale`, rounding to nearest,
// and clamping to [-128, 127]. Each int8 step represents `scale` units.
int8_t Quantize(float value, float scale);

// Converts an int32 accumulator back to float by multiplying by the combined
// scale product (input_scale * weight_scale). Used after a dot product.
float Dequantize(int32_t value, float scale_product);

// Integer-only dot product: accumulates int8×int8 products into an int32
// accumulator. This is the performance-critical inner loop — on Cortex-M7 it
// maps to the SMLAD (dual signed multiply-accumulate) instruction.
int32_t Dot(const int8_t* input, const int8_t* weights, size_t length);

// Batch-quantises a float buffer into an int8 buffer using a single scale.
// Useful for re-encoding activations between layers.
void RequantizeBuffer(const float* input,
                      size_t length,
                      float scale,
                      int8_t* output);

}  // namespace quant
}  // namespace orion
