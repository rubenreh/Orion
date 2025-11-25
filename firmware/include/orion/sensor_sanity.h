// Author : Ruben Rehal | Date : November 2025
#pragma once

#include "system_types.h"

namespace orion {
struct SensorHealth {
    bool imu_ok = true;
    bool ultrasonic_ok = true;
    bool optical_ok = true;
    bool timestamp_sync = true;
};

class SensorSanityChecker {
  public:
    SensorHealth Evaluate(const ImuFrame& imu,
                          const UltrasonicFrame& sonar,
                          const OpticalFrame& optic);

  private:
    ImuFrame last_imu_{};
    UltrasonicFrame last_sonar_{};
    OpticalFrame last_optic_{};
    bool initialized_ = false;
};
}

