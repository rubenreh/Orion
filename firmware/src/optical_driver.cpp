// Author : Ruben Rehal | Date : November 2025
#include "orion/optical_driver.h"

#include <cmath>

namespace orion {
OpticalFrame MockOpticalDriver::Read() {
    flicker_ += 0.1f;
    if (flicker_ > 6.28f) flicker_ -= 6.28f;
    OpticalFrame frame;
    frame.intensity = 0.5f + 0.3f * std::sin(flicker_);
    frame.delta = 0.05f * std::cos(flicker_);
    frame.timestamp_us = static_cast<uint32_t>(flicker_ * 1e5f);
    return frame;
}
}
