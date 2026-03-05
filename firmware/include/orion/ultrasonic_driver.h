// Author : Ruben Rehal | Date : November 2025
//
// ultrasonic_driver.h — Mock ultrasonic range-finder driver for desktop testing.
// Produces a smooth sinusoidal distance signal around 0.5 m with fixed high
// confidence, allowing the fusion and inference pipeline to be validated
// without physical sonar hardware.
#pragma once

#include "sensor_interfaces.h"  // UltrasonicDriver base class, UltrasonicFrame

namespace orion {

// Simulated sonar sensor. Distance oscillates between ~0.35 m and ~0.65 m
// with an internal phase counter that advances each Read() call.
class MockUltrasonicDriver : public UltrasonicDriver {
  public:
    UltrasonicFrame Read() override;   // Returns a new simulated sonar sample

  private:
    float phase_ = 0.0f;              // Running phase angle (radians) for sine wave
};

}
