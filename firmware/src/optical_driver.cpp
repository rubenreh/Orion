// Author : Ruben Rehal | Date : November 2025
//
// optical_driver.cpp — Mock optical (light/reflectance) sensor implementation.
// Generates sinusoidal intensity and cosine-derived delta signals to simulate
// a flickering ambient light source. Used for desktop pipeline validation
// without real SPI/ADC optical hardware.

#include "orion/optical_driver.h"

#include <cmath>  // std::sin, std::cos for waveform generation

namespace orion {

// Generates one new optical sample. Called once per loop iteration (200 Hz).
// Simulates gentle intensity flickering with a frame-to-frame delta.
OpticalFrame MockOpticalDriver::Read() {
    // Advance the flicker phase by 0.1 radians (faster than sonar phase)
    flicker_ += 0.1f;

    // Wrap at 2π to prevent unbounded growth
    if (flicker_ > 6.28f) flicker_ -= 6.28f;

    OpticalFrame frame;

    // Intensity oscillates sinusoidally: 0.5 centre ± 0.3 amplitude (range ~0.2–0.8)
    frame.intensity = 0.5f + 0.3f * std::sin(flicker_);

    // Delta approximates the derivative of intensity: cos is the derivative of sin,
    // scaled by 0.05 to represent a small frame-to-frame change
    frame.delta = 0.05f * std::cos(flicker_);

    // Derive a timestamp from the phase (scales phase to ~100 000 µs per radian)
    frame.timestamp_us = static_cast<uint32_t>(flicker_ * 1e5f);

    return frame;
}

}
