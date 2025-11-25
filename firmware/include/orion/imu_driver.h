// Author : Ruben Rehal | Date : November 2025
#pragma once

#include "sensor_interfaces.h"

namespace orion {
class MockImuDriver : public ImuDriver {
  public:
    MockImuDriver();
    ImuFrame Read() override;

  private:
    ImuFrame frame_;
};
}
