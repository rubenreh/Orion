// Author : Ruben Rehal | Date : November 2025
#include "orion/activations.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace orion {
namespace {
constexpr size_t kReluLutSize = 256;
std::array<float, kReluLutSize> g_relu_lut{};
bool g_lut_ready = false;
}

namespace activations {
float Relu(float x) {
    return std::max(0.0f, x);
}

float TanhApprox(float x) {
    const float exp_pos = std::exp(std::min(10.0f, x));
    const float exp_neg = std::exp(std::min(10.0f, -x));
    return (exp_pos - exp_neg) / (exp_pos + exp_neg);
}

void ReluLutInit() {
    if (g_lut_ready) return;
    for (size_t i = 0; i < kReluLutSize; ++i) {
        const float x = static_cast<float>(i) / 32.0f - 4.0f;  // map [-4,4]
        g_relu_lut[i] = Relu(x);
    }
    g_lut_ready = true;
}

float ReluFromLut(float x) {
    if (!g_lut_ready) ReluLutInit();
    const float clamped = std::clamp(x, -4.0f, 4.0f);
    const size_t idx = static_cast<size_t>((clamped + 4.0f) * 32.0f);
    return g_relu_lut[std::min(idx, kReluLutSize - 1)];
}
}
}

