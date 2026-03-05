// Author : Ruben Rehal | Date : November 2025
//
// fusion_pipeline.h — Sensor fusion module that takes raw IMU, ultrasonic, and
// optical readings, filters each modality independently, and packs the results
// into an 8-element FusedFrame feature vector consumed by the inference engine.
// This is the bridge between heterogeneous sensor data and the neural network.
#pragma once

#include "imu_filter.h"         // ImuFilter — EMA filter for IMU data
#include "optical_smoothing.h"  // OpticalSmoothing — EMA filter for optical data
#include "system_types.h"       // ImuFrame, UltrasonicFrame, OpticalFrame, FusedFrame
#include "ultrasonic_merge.h"   // UltrasonicMerge — EMA filter for sonar data

namespace orion {

// Orchestrates per-sensor filtering and produces the unified 8-D feature vector:
//   [0] accel magnitude   [1] gyro magnitude     [2] accel Z
//   [3] sonar distance    [4] sonar confidence    [5] optical intensity
//   [6] optical delta     [7] inter-frame dt (seconds)
class FusionPipeline {
  public:
    FusionPipeline();

    // Filters the three raw sensor frames and packs the result into a FusedFrame.
    FusedFrame Process(const ImuFrame& imu_raw,
                       const UltrasonicFrame& ultrasonics_raw,
                       const OpticalFrame& optical_raw);

  private:
    ImuFilter imu_filter_;                 // EMA filter for accelerometer and gyroscope
    UltrasonicMerge ultrasonic_filter_;    // EMA filter for sonar distance/confidence
    OpticalSmoothing optical_filter_;      // EMA filter for optical intensity/delta
    uint32_t last_timestamp_us_ = 0;       // Previous IMU timestamp, used to compute dt

    // Computes the Euclidean magnitude (L2 norm) of a 3-element vector.
    // Used to collapse 3-axis accel/gyro into a single rotation-invariant scalar.
    static float Magnitude(const std::array<float, 3>& v);
};

}
