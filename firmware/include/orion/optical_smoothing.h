// Author : Ruben Rehal | Date : November 2025
#pragma once

#include "system_types.h"

namespace orion {
class OpticalSmoothing {
  public:
    OpticalSmoothing();
    OpticalFrame Smooth(const OpticalFrame& raw);

  private:
    OpticalFrame state_{};
    bool initialized_ = false;
};
}
