// Author : Ruben Rehal | Date : November 2025
#pragma once

#include <array>
#include <vector>

#include "layers.h"
#include "model_data.h"
#include "system_types.h"

namespace orion {
class InferenceEngine {
  public:
    InferenceEngine();
    void LoadModel(const model::Model& model);
    InferenceOutput Run(const FusedFrame& fused);

  private:
    std::vector<DenseLayer> layers_;
    std::vector<int8_t> arena_a_;
    std::vector<int8_t> arena_b_;
    const model::Model* model_ = nullptr;

    std::vector<int8_t> QuantizeInput(const FusedFrame& fused) const;
};
}

