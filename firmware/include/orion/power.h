// Author : Ruben Rehal | Date : November 2025
#pragma once

#include <cstdint>

namespace orion {
class PowerMonitor {
  public:
    void Update(float voltage_v, uint32_t timestamp_us);
    bool IsBrownout() const { return brownout_; }
    float LastVoltage() const { return last_voltage_; }

  private:
    float last_voltage_ = 3.3f;
    uint32_t last_timestamp_us_ = 0;
    bool brownout_ = false;
};
}

