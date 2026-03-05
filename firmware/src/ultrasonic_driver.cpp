// Author : Ruben Rehal | Date : November 2025
//
// ultrasonic_driver.cpp — Mock ultrasonic range-finder implementation. Produces
// a smooth sinusoidal distance signal around 0.5 m with fixed high confidence
// (0.9). The phase counter advances by 0.05 rad each call, completing one full
// cycle every ~126 calls (~0.63 s at 200 Hz).

#include "orion/ultrasonic_driver.h"

#include <cmath>  // std::sin for sinusoidal distance generation

namespace orion {

// Generates one new sonar sample. Called once per loop iteration (200 Hz).
// Simulates a target that oscillates between ~0.35 m and ~0.65 m.
UltrasonicFrame MockUltrasonicDriver::Read() {
    // Advance the internal phase angle by 0.05 radians
    phase_ += 0.05f;

    // Wrap the phase at 2π (≈6.28) to prevent unbounded growth
    if (phase_ > 6.28f) phase_ -= 6.28f;

    UltrasonicFrame frame;

    // Distance oscillates sinusoidally: 0.5 m centre ± 0.15 m amplitude
    frame.distance_m = 0.5f + 0.15f * std::sin(phase_);

    // Fixed high confidence — simulates a clean echo environment
    frame.confidence = 0.9f;

    // Derive a timestamp from the phase (scales phase to ~100 000 µs per radian)
    frame.timestamp_us = static_cast<uint32_t>(phase_ * 1e5f);

    return frame;
}

}
