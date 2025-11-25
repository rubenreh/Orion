// Author : Ruben Rehal | Date : November 2025
#include "orion/ultrasonic_driver.h"

#include <cmath>

namespace orion {
UltrasonicFrame MockUltrasonicDriver::Read() {
    phase_ += 0.05f;
    if (phase_ > 6.28f) phase_ -= 6.28f;
    UltrasonicFrame frame;
    frame.distance_m = 0.5f + 0.15f * std::sin(phase_);
    frame.confidence = 0.9f;
    frame.timestamp_us = static_cast<uint32_t>(phase_ * 1e5f);
    return frame;
}
}
