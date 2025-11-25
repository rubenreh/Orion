// Author : Ruben Rehal | Date : November 2025
#include "orion/layers.h"

#include <algorithm>
#include <cmath>

#include "orion/quant_ops.h"

namespace orion {
DenseLayer::DenseLayer(const DenseLayerConfig& cfg, std::function<float(float)> activation)
    : cfg_(cfg), activation_(std::move(activation)) {}

void DenseLayer::Invoke(const std::vector<int8_t>& input, std::vector<int8_t>& output) const {
    if (output.size() != cfg_.output_dim) {
        output.assign(cfg_.output_dim, 0);
    }
    last_activations_.assign(cfg_.output_dim, 0.0f);
    for (size_t o = 0; o < cfg_.output_dim; ++o) {
        const int32_t* bias_ptr = cfg_.bias;
        int32_t acc = bias_ptr ? bias_ptr[o] : 0;
        const int8_t* weight_row = cfg_.weights + o * cfg_.input_dim;
        acc += quant::Dot(input.data(), weight_row, cfg_.input_dim);
        const float deq = quant::Dequantize(acc, cfg_.input_scale * cfg_.weight_scale);
        const float activated = activation_(deq);
        last_activations_[o] = activated;
        const float requant = activated / cfg_.output_scale;
        const int32_t rounded = static_cast<int32_t>(std::round(requant));
        output[o] = static_cast<int8_t>(std::clamp(rounded, -128, 127));
    }
}
}

