// Author : Ruben Rehal | Date : November 2025
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace orion {
struct DenseLayerConfig {
    size_t input_dim;
    size_t output_dim;
    const int8_t* weights;
    const int32_t* bias;
    float input_scale;
    float weight_scale;
    float output_scale;
};

class DenseLayer {
  public:
    DenseLayer(const DenseLayerConfig& cfg, std::function<float(float)> activation);
    void Invoke(const std::vector<int8_t>& input, std::vector<int8_t>& output) const;
    std::vector<float> LastActivations() const { return last_activations_; }

  private:
    DenseLayerConfig cfg_;
    std::function<float(float)> activation_;
    mutable std::vector<float> last_activations_;
};
}

