// Author : Ruben Rehal | Date : November 2025
//
// imu_driver.cpp — Mock IMU driver implementation. Produces deterministic
// sinusoidal sensor data that simulates a stationary device with minor
// vibrations. Timestamps auto-advance at 200 Hz so the rest of the stack
// (filters, sanity checker, fusion) sees realistic timing without real hardware.

#include "orion/imu_driver.h"

#include <cmath>  // std::sin for sinusoidal wobble generation

namespace orion {

// Initialises the IMU frame to a stationary reading: 1 g on the Z-axis
// (gravity), zero angular velocity, room temperature, and time zero.
MockImuDriver::MockImuDriver() {
    frame_.accel = {0.0f, 0.0f, 1.0f};   // [x, y, z] — gravity along Z
    frame_.gyro = {0.0f, 0.0f, 0.0f};    // No rotation
    frame_.temperature_c = 25.0f;          // Room temperature in °C
    frame_.timestamp_us = 0;               // Start at time zero
}

// Generates one new IMU sample. Called once per loop iteration (200 Hz).
// The sinusoidal wobble simulates minor vibrations on a stable platform.
ImuFrame MockImuDriver::Read() {
    // Advance the timestamp by 5 000 µs (= 1/200 Hz), simulating the sample rate
    frame_.timestamp_us += 5000;

    // Compute a slowly-varying wobble factor from the elapsed time (seconds).
    // sin(t) has a period of 2π ≈ 6.28 s, so the wobble cycles every ~6.3 s.
    const float wobble = std::sin(static_cast<float>(frame_.timestamp_us) * 1e-6f);

    // Apply tiny sinusoidal perturbations to the accelerometer axes.
    // X and Y get ±0.01/0.02 g; Z hovers around 1 g ±0.01 g.
    frame_.accel[0] = 0.01f * wobble;            // X-axis accel perturbation
    frame_.accel[1] = 0.02f * wobble;            // Y-axis accel perturbation
    frame_.accel[2] = 1.0f + 0.01f * wobble;    // Z-axis: gravity + small wobble

    // Apply sub-millirad/s perturbations to the gyroscope axes
    frame_.gyro[0] = 0.001f * wobble;   // X-axis angular velocity
    frame_.gyro[1] = 0.002f * wobble;   // Y-axis angular velocity
    frame_.gyro[2] = 0.003f * wobble;   // Z-axis angular velocity

    return frame_;
}

}
