// Author : Ruben Rehal | Date : November 2025
#pragma once

#include "sensor_interfaces.h"

namespace orion {
class MockUltrasonicDriver : public UltrasonicDriver {
  public:
    UltrasonicFrame Read() override;

  private:
    float phase_ = 0.0f;
};
}
