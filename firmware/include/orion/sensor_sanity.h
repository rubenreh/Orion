// Author : Ruben Rehal | Date : November 2025
//
// sensor_sanity.h — Plausibility checker that validates every sensor frame
// against known physical limits and timing constraints. Inspired by ISO-26262
// diagnostic requirements, it ensures the system never acts on physically
// impossible or stale data. If any check fails, the main loop skips the
// control output and enters a safe degraded state.
#pragma once

#include "system_types.h"  // ImuFrame, UltrasonicFrame, OpticalFrame

namespace orion {

// Aggregated health flags — one boolean per sensor modality plus a timing flag.
// The main loop AND's all four to decide whether to trust the inference output.
struct SensorHealth {
    bool imu_ok = true;            // True if accel and gyro are within physical range
    bool ultrasonic_ok = true;     // True if distance and confidence are within spec
    bool optical_ok = true;        // True if intensity is within normalised range
    bool timestamp_sync = true;    // True if no sensor has gone stale
};

// Evaluates per-frame plausibility and inter-frame timing.
// Retains the previous frames internally to compute timestamp deltas.
class SensorSanityChecker {
  public:
    // Checks every field of all three frames against hard-coded physical limits
    // and verifies that timestamps haven't drifted beyond allowable gaps.
    SensorHealth Evaluate(const ImuFrame& imu,
                          const UltrasonicFrame& sonar,
                          const OpticalFrame& optic);

  private:
    ImuFrame last_imu_{};              // Previous IMU frame for timestamp delta
    UltrasonicFrame last_sonar_{};     // Previous sonar frame for timestamp delta
    OpticalFrame last_optic_{};        // Previous optical frame for timestamp delta
    bool initialized_ = false;         // Skips timestamp check on the very first call
};

}
