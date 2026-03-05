// Author : Ruben Rehal | Date : November 2025
//
// imu_filter.h — Exponential Moving Average (EMA) low-pass filter for IMU data.
// Smooths noisy accelerometer and gyroscope readings before they enter the
// fusion pipeline, using independent alpha coefficients so that each modality
// can trade off noise rejection vs. responsiveness independently.
#pragma once

#include "system_types.h"  // ImuFrame

namespace orion {

// Applies per-axis EMA filtering:  filtered = α·new + (1-α)·old.
// A lower α gives heavier smoothing (more lag, less noise).
// Accelerometer gets α=0.25 (noisier), gyro gets α=0.35 (needs faster tracking).
class ImuFilter {
  public:
    // Constructs the filter with separate smoothing factors for accel and gyro.
    ImuFilter(float alpha_accel, float alpha_gyro);

    // Accepts a raw IMU frame and returns the filtered version.
    // On the very first call the raw frame is adopted as-is (cold-start).
    ImuFrame Filter(const ImuFrame& raw);

  private:
    float alpha_accel_;      // EMA coefficient for accelerometer axes
    float alpha_gyro_;       // EMA coefficient for gyroscope axes
    ImuFrame state_{};       // Running filtered state carried between calls
    bool initialized_ = false;  // False until the first sample is received
};

}
