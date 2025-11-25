// Author : Ruben Rehal | Date : November 2025
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "layers.h"

namespace orion {
namespace model {
constexpr size_t kInputDim = 8;
constexpr size_t kHidden1 = 12;
constexpr size_t kHidden2 = 6;
constexpr size_t kOutputDim = 2;

struct Model {
    DenseLayerConfig layer1;
    DenseLayerConfig layer2;
    DenseLayerConfig layer3;
};

const Model& GetCalibratedModel();
}
}

