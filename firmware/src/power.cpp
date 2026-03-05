// Author : Ruben Rehal | Date : November 2025
//
// power.cpp — Supply-rail voltage monitor with brownout detection. On real
// hardware this would sample the MCU's internal ADC on the VDD rail; in
// simulation the voltage is passed in directly. When the supply drops below
// kVoltageCritical (2.8 V), the brownout flag is raised and the main loop
// enters a safe fallback state to prevent data corruption or erratic actuation.

#include "orion/power.h"

#include "orion/config.h"  // kVoltageCritical (2.8 V)

namespace orion {

// Stores the latest voltage reading, records the timestamp, and sets the
// brownout flag if the voltage has dropped below the critical threshold.
// Called once per loop iteration by main.cpp with the current supply voltage.
void PowerMonitor::Update(float voltage_v, uint32_t timestamp_us) {
    last_voltage_ = voltage_v;             // Cache the reading for LastVoltage()
    last_timestamp_us_ = timestamp_us;     // Record when the sample was taken
    // Compare against the critical threshold (2.8 V) from config.h
    brownout_ = voltage_v < kVoltageCritical;
}

}
