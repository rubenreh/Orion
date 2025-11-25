// Author : Ruben Rehal | Date : November 2025
#include "orion/quant_ops.h"

#include <algorithm>
#include <cmath>

namespace orion {
namespace quant {
int8_t Quantize(float value, float scale) {
    const int32_t scaled = static_cast<int32_t>(std::round(value / scale));
    return static_cast<int8_t>(std::clamp(scaled, -128, 127));
}

float Dequantize(int32_t value, float scale_product) {
    return static_cast<float>(value) * scale_product;
}

int32_t Dot(const int8_t* input, const int8_t* weights, size_t length) {
    int32_t acc = 0;
    for (size_t i = 0; i < length; ++i) {
        acc += static_cast<int32_t>(input[i]) * static_cast<int32_t>(weights[i]);
    }
    return acc;
}

void RequantizeBuffer(const float* input,
                      size_t length,
                      float scale,
                      int8_t* output) {
    for (size_t i = 0; i < length; ++i) {
        output[i] = Quantize(input[i], scale);
    }
}
}
}

