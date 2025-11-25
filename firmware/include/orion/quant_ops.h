// Author : Ruben Rehal | Date : November 2025
#pragma once

#include <cstddef>
#include <cstdint>

namespace orion {
namespace quant {
int8_t Quantize(float value, float scale);
float Dequantize(int32_t value, float scale_product);
int32_t Dot(const int8_t* input, const int8_t* weights, size_t length);
void RequantizeBuffer(const float* input,
                      size_t length,
                      float scale,
                      int8_t* output);
}
}

