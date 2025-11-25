// Author : Ruben Rehal | Date : November 2025
#include "orion/power.h"

#include "orion/config.h"

namespace orion {
void PowerMonitor::Update(float voltage_v, uint32_t timestamp_us) {
    last_voltage_ = voltage_v;
    last_timestamp_us_ = timestamp_us;
    brownout_ = voltage_v < kVoltageCritical;
}
}

