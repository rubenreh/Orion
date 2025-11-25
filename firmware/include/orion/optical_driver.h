// Author : Ruben Rehal | Date : November 2025
#pragma once

#include "sensor_interfaces.h"

namespace orion {
class MockOpticalDriver : public OpticalDriver {
  public:
    OpticalFrame Read() override;

  private:
    float flicker_ = 0.0f;
};
}
