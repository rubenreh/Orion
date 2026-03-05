// Author : Ruben Rehal | Date : November 2025
//
// activations.h — Activation function declarations for the neural-network
// layers. Provides both a direct ReLU, a lookup-table (LUT) accelerated ReLU
// for the hidden layers, and a tanh approximation for the output layer.
// The LUT approach trades 1 KB of RAM for branch-free, cache-friendly
// evaluation on MCUs where conditional branches can stall the pipeline.
#pragma once

#include <cstddef>  // size_t for LUT indexing

namespace orion {
namespace activations {

// Standard Rectified Linear Unit: max(0, x). Simple but involves a branch.
float Relu(float x);

// Tanh approximation using (e^x − e^−x)/(e^x + e^−x) with clamped exponents
// to prevent overflow. Used only in the output layer (2 neurons).
float TanhApprox(float x);

// Populates the 256-entry ReLU lookup table (maps [-4, +4] at 32 bins/unit).
// Called lazily on the first invocation of ReluFromLut().
void ReluLutInit();

// LUT-backed ReLU. Clamps x to [-4, +4], indexes into the pre-computed table,
// and returns the stored value. Avoids branching in the hot inference loop.
float ReluFromLut(float x);

}  // namespace activations
}  // namespace orion
