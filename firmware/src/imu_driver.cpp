// Author : Ruben Rehal | Date : November 2025
#include "orion/imu_driver.h"

#include <cmath>

namespace orion {
MockImuDriver::MockImuDriver() {
    frame_.accel = {0.0f, 0.0f, 1.0f};
    frame_.gyro = {0.0f, 0.0f, 0.0f};
    frame_.temperature_c = 25.0f;
    frame_.timestamp_us = 0;
}

ImuFrame MockImuDriver::Read() {
    frame_.timestamp_us += 5000;  // pretend 200 Hz
    const float wobble = std::sin(static_cast<float>(frame_.timestamp_us) * 1e-6f);
    frame_.accel[0] = 0.01f * wobble;
    frame_.accel[1] = 0.02f * wobble;
    frame_.accel[2] = 1.0f + 0.01f * wobble;
    frame_.gyro[0] = 0.001f * wobble;
    frame_.gyro[1] = 0.002f * wobble;
    frame_.gyro[2] = 0.003f * wobble;
    return frame_;
}
}
