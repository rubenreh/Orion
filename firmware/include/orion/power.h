// Author : Ruben Rehal | Date : November 2025
//
// power.h — Supply-rail voltage monitor with brownout detection. On real
// hardware this would sample the MCU's ADC; in simulation a voltage is passed
// in directly. When the supply drops below kVoltageCritical (2.8 V) the system
// flags a brownout and the main loop enters a safe fallback mode to prevent
// data corruption or erratic actuator behaviour.
#pragma once

#include <cstdint>  // uint32_t for timestamps

namespace orion {

// Monitors the supply voltage and raises a brownout flag when it drops below
// the critical threshold defined in config.h.
class PowerMonitor {
  public:
    // Stores the latest voltage reading and sets the brownout flag if the
    // voltage is below kVoltageCritical.
    void Update(float voltage_v, uint32_t timestamp_us);

    // Returns true if the most recent voltage was below the critical threshold.
    bool IsBrownout() const { return brownout_; }

    // Returns the most recently recorded supply voltage in volts.
    float LastVoltage() const { return last_voltage_; }

  private:
    float last_voltage_ = 3.3f;       // Default nominal voltage (3.3 V rail)
    uint32_t last_timestamp_us_ = 0;  // Timestamp of the last Update() call
    bool brownout_ = false;            // True when voltage < kVoltageCritical
};

}
