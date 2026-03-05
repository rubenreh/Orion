// Author : Ruben Rehal | Date : November 2025
//
// optical_driver.h — Mock optical (light/reflectance) sensor driver for desktop
// testing. Generates smooth sinusoidal intensity and delta signals so the full
// pipeline can be validated without real SPI/ADC optical hardware.
#pragma once

#include "sensor_interfaces.h"  // OpticalDriver base class, OpticalFrame

namespace orion {

// Simulated optical sensor. Intensity oscillates around 0.5 with ±0.3
// amplitude; delta is derived from the cosine of the same phase, approximating
// the derivative of intensity.
class MockOpticalDriver : public OpticalDriver {
  public:
    OpticalFrame Read() override;      // Returns a new simulated optical sample

  private:
    float flicker_ = 0.0f;            // Running phase angle (radians) for sine/cosine wave
};

}
