// Author : Ruben Rehal | Date : November 2025
#include "orion/inference_engine.h"

#include <algorithm>
#include <cmath>

#include "orion/activations.h"

namespace orion {
InferenceEngine::InferenceEngine() {
    arena_a_.reserve(model::kHidden1);
    arena_b_.reserve(model::kHidden2);
}

void InferenceEngine::LoadModel(const model::Model& model) {
    model_ = &model;
    layers_.clear();
    layers_.emplace_back(model.layer1, activations::ReluFromLut);
    layers_.emplace_back(model.layer2, activations::ReluFromLut);
    layers_.emplace_back(model.layer3, activations::TanhApprox);
}

std::vector<int8_t> InferenceEngine::QuantizeInput(const FusedFrame& fused) const {
    std::vector<int8_t> input(model::kInputDim, 0);
    for (size_t i = 0; i < model::kInputDim; ++i) {
        input[i] = quant::Quantize(fused.features[i], 0.02f);
    }
    return input;
}

InferenceOutput InferenceEngine::Run(const FusedFrame& fused) {
    if (!model_) {
        LoadModel(model::GetCalibratedModel());
    }

    std::vector<int8_t> buffer_a = QuantizeInput(fused);
    std::vector<int8_t> buffer_b(model::kHidden1);

    layers_[0].Invoke(buffer_a, buffer_b);
    buffer_a.assign(buffer_b.begin(), buffer_b.end());
    buffer_b.assign(model::kHidden2, 0);

    layers_[1].Invoke(buffer_a, buffer_b);
    buffer_a.assign(buffer_b.begin(), buffer_b.end());
    buffer_b.assign(model::kOutputDim, 0);

    layers_[2].Invoke(buffer_a, buffer_b);

    const auto& logits = layers_.back().LastActivations();
    const float anomaly = std::clamp(0.5f + logits[0] * 0.1f, 0.0f, 1.0f);
    const float confidence = std::clamp(0.5f + logits[1] * 0.1f, 0.0f, 1.0f);
    return InferenceOutput{anomaly, confidence};
}
}

