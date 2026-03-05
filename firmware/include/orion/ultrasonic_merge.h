// Author : Ruben Rehal | Date : November 2025
//
// ultrasonic_merge.h — EMA low-pass filter for ultrasonic (sonar) readings.
// Sonar is especially prone to multipath reflections and spurious echoes, so it
// uses a heavier smoothing factor (α=0.2) than the IMU or optical channels.
#pragma once

#include "system_types.h"  // UltrasonicFrame

namespace orion {

// Smooths raw sonar frames with an EMA filter. Distance uses α=0.2 (heavy
// smoothing) and confidence is averaged 50/50 between old and new readings.
class UltrasonicMerge {
  public:
    UltrasonicMerge();

    // Accepts a raw sonar frame and returns the smoothed version.
    // The first frame is adopted as-is (cold-start).
    UltrasonicFrame Filter(const UltrasonicFrame& raw);

  private:
    UltrasonicFrame state_{};        // Running filtered state
    bool initialized_ = false;       // False until the first sample arrives
};

}
