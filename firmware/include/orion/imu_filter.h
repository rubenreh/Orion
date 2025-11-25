// Author : Ruben Rehal | Date : November 2025
#pragma once

#include "system_types.h"

namespace orion {
class ImuFilter {
  public:
    ImuFilter(float alpha_accel, float alpha_gyro);
    ImuFrame Filter(const ImuFrame& raw);

  private:
    float alpha_accel_;
    float alpha_gyro_;
    ImuFrame state_{};
    bool initialized_ = false;
};
}
