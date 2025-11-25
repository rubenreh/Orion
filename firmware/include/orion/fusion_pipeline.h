// Author : Ruben Rehal | Date : November 2025
#pragma once

#include "imu_filter.h"
#include "optical_smoothing.h"
#include "system_types.h"
#include "ultrasonic_merge.h"

namespace orion {
class FusionPipeline {
  public:
    FusionPipeline();
    FusedFrame Process(const ImuFrame& imu_raw,
                       const UltrasonicFrame& ultrasonics_raw,
                       const OpticalFrame& optical_raw);

  private:
    ImuFilter imu_filter_;
    UltrasonicMerge ultrasonic_filter_;
    OpticalSmoothing optical_filter_;
    uint32_t last_timestamp_us_ = 0;

    static float Magnitude(const std::array<float, 3>& v);
};
}

