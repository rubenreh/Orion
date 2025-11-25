// Author : Ruben Rehal | Date : November 2025
#include "orion/model_data.h"

#include <array>

#include "orion/activations.h"

namespace orion {
namespace model {
namespace {
constexpr float kInputScale = 0.02f;
constexpr float kHidden1Scale = 0.04f;
constexpr float kHidden2Scale = 0.05f;

constexpr std::array<int8_t, kInputDim * kHidden1> kLayer1Weights = {
    -3,  4,  2, -2,  5,  6, -5,  3,
     1, -4,  3,  6, -2, -6,  5,  2,
     3,  1, -1,  4, -5,  4,  2, -3,
     0,  6,  5, -4,  3, -2, -1,  4,
    -4,  2, -3,  5,  6, -1,  1,  0,
     5, -6,  4,  2, -3,  3, -2,  1,
    -2,  1,  6, -5,  4, -3,  2,  5,
     4, -5,  1,  3,  2,  6, -4, -2,
    -5,  3, -4,  2,  1, -2,  6,  5,
     2, -3,  5, -6,  4,  1, -1,  2,
     3,  5, -2,  1, -4,  2,  6, -5,
    -1,  2,  4, -3,  5, -6,  3,  2};

constexpr std::array<int32_t, kHidden1> kLayer1Bias = {
    12, -18, 5, 7, -6, 10, -9, 4, 3, -2, 6, -4};

constexpr std::array<int8_t, kHidden1 * kHidden2> kLayer2Weights = {
     4, -3,  2, -1,  5,  3,
    -5,  4, -2,  3, -1,  2,
     3,  5, -4,  2, -3,  1,
    -2,  1,  5, -4,  3, -5,
     5, -2,  3,  4, -1,  2,
    -4,  3, -5,  1,  2, -3,
     2,  4, -1,  5, -3,  2,
    -3,  2,  4, -2,  5, -1,
     4, -5,  2, -3,  1,  3,
    -1,  3, -2,  4, -4,  5,
     2, -1,  5,  3, -2,  1,
    -5,  4, -3,  2,  3, -2};

constexpr std::array<int32_t, kHidden2> kLayer2Bias = {6, -4, 3, -2, 5, -3};

constexpr std::array<int8_t, kHidden2 * kOutputDim> kLayer3Weights = {
     4, -3,
    -5,  2,
     3, -4,
    -2,  5,
     5, -1,
    -3,  4};

constexpr std::array<int32_t, kOutputDim> kLayer3Bias = {2, -2};

Model BuildModel() {
    static DenseLayerConfig l1{
        kInputDim,
        kHidden1,
        kLayer1Weights.data(),
        kLayer1Bias.data(),
        kInputScale,
        0.03125f,
        kHidden1Scale};
    static DenseLayerConfig l2{
        kHidden1,
        kHidden2,
        kLayer2Weights.data(),
        kLayer2Bias.data(),
        kHidden1Scale,
        0.03125f,
        kHidden2Scale};
    static DenseLayerConfig l3{
        kHidden2,
        kOutputDim,
        kLayer3Weights.data(),
        kLayer3Bias.data(),
        kHidden2Scale,
        0.03125f,
        0.02f};
    static Model model{l1, l2, l3};
    return model;
}
}

const Model& GetCalibratedModel() {
    static Model model = BuildModel();
    return model;
}
}
}

