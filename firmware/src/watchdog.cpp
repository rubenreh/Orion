// Author : Ruben Rehal | Date : November 2025
#include "orion/watchdog.h"

namespace orion {
void Watchdog::Configure(uint32_t timeout_us) {
    timeout_us_ = timeout_us;
    armed_ = timeout_us_ > 0;
    last_kick_us_ = 0;
}

void Watchdog::Kick(uint32_t timestamp_us) {
    if (!armed_) return;
    last_kick_us_ = timestamp_us;
}

bool Watchdog::Expired(uint32_t timestamp_us) const {
    if (!armed_) return false;
    if (last_kick_us_ == 0) return false;
    return (timestamp_us - last_kick_us_) > timeout_us_;
}
}

