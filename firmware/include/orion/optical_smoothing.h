// Author : Ruben Rehal | Date : November 2025
//
// optical_smoothing.h — EMA low-pass filter for optical sensor readings.
// Uses α=0.4 (moderate smoothing) because optical sensors are generally less
// noisy than sonar while still benefiting from some jitter removal.
#pragma once

#include "system_types.h"  // OpticalFrame

namespace orion {

// Smooths raw optical frames with an EMA filter (α=0.4 for both intensity
// and delta). The first frame is adopted as-is (cold-start).
class OpticalSmoothing {
  public:
    OpticalSmoothing();

    // Accepts a raw optical frame and returns the smoothed version.
    OpticalFrame Smooth(const OpticalFrame& raw);

  private:
    OpticalFrame state_{};         // Running filtered state
    bool initialized_ = false;     // False until the first sample arrives
};

}
