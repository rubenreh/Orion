// Author : Ruben Rehal | Date : November 2025
#pragma once

#include <cstddef>

namespace orion {
namespace activations {
float Relu(float x);
float TanhApprox(float x);
void ReluLutInit();
float ReluFromLut(float x);
}
}

