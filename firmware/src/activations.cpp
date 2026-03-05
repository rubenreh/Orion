// Author : Ruben Rehal | Date : November 2025
//
// activations.cpp — Activation function implementations for the neural-network
// layers. Provides a direct ReLU, a 256-entry lookup-table (LUT) ReLU for the
// hidden layers (avoids branching in the hot loop), and a tanh approximation
// for the output layer. The LUT trades 1 KB of RAM for deterministic,
// cache-friendly, branch-free evaluation on MCUs where conditional branches
// can stall the pipeline.

#include "orion/activations.h"

#include <algorithm>  // std::clamp for input range clamping
#include <array>      // std::array for the 256-entry LUT
#include <cmath>      // std::exp for tanh, std::max for ReLU

namespace orion {
namespace {

// Size of the ReLU lookup table: 256 entries covering [-4.0, +4.0]
// at 32 bins per unit, giving 0.03125 resolution.
constexpr size_t kReluLutSize = 256;

// Pre-computed ReLU values indexed by quantised input
std::array<float, kReluLutSize> g_relu_lut{};

// Guard flag: the LUT is built lazily on first use
bool g_lut_ready = false;

}  // anonymous namespace

namespace activations {

// Standard ReLU: returns max(0, x). Simple but involves a branch instruction
// that can cause pipeline stalls on in-order Cortex-M cores.
float Relu(float x) {
    return std::max(0.0f, x);
}

// Tanh approximation: (e^x − e^−x) / (e^x + e^−x).
// Both exponents are clamped to min(10, ±x) to prevent float overflow
// (e^10 ≈ 22 026, safely within float range). Used only in the output layer
// where just 2 neurons are evaluated, so the exp() cost is acceptable.
float TanhApprox(float x) {
    // Clamp exponents to ±10 to avoid overflow for extreme inputs
    const float exp_pos = std::exp(std::min(10.0f, x));    // e^(+x), capped
    const float exp_neg = std::exp(std::min(10.0f, -x));   // e^(-x), capped
    // Standard tanh formula: (e^x - e^-x) / (e^x + e^-x)
    return (exp_pos - exp_neg) / (exp_pos + exp_neg);
}

// Populates the 256-entry ReLU LUT. Each entry maps an index i to:
//   x = i/32 − 4   (covers the range [-4.0, +4.0])
//   LUT[i] = max(0, x)
// Called once lazily; subsequent calls are a no-op.
void ReluLutInit() {
    if (g_lut_ready) return;  // Already initialised — nothing to do

    for (size_t i = 0; i < kReluLutSize; ++i) {
        // Map index to float: 0→−4.0, 128→0.0, 255→+3.97
        const float x = static_cast<float>(i) / 32.0f - 4.0f;
        // Store the ReLU result
        g_relu_lut[i] = Relu(x);
    }
    g_lut_ready = true;  // Mark the table as ready
}

// LUT-backed ReLU. Clamps x to [-4, +4], computes an integer index, and
// looks up the pre-computed result. This is branch-free in the hot path
// (the clamping uses min/max intrinsics) and fully deterministic.
float ReluFromLut(float x) {
    // Lazily build the LUT on first invocation
    if (!g_lut_ready) ReluLutInit();

    // Clamp input to the LUT's representable range
    const float clamped = std::clamp(x, -4.0f, 4.0f);

    // Convert the clamped float to an index: shift by +4 then scale by 32 bins/unit
    const size_t idx = static_cast<size_t>((clamped + 4.0f) * 32.0f);

    // Return the pre-computed value, guarding against off-by-one at the boundary
    return g_relu_lut[std::min(idx, kReluLutSize - 1)];
}

}  // namespace activations
}  // namespace orion
