// Author : Ruben Rehal | Date : November 2025
#include "orion/sensor_sanity.h"

#include <cmath>

namespace orion {
SensorHealth SensorSanityChecker::Evaluate(const ImuFrame& imu,
                                           const UltrasonicFrame& sonar,
                                           const OpticalFrame& optic) {
    SensorHealth health{};
    const bool imu_range =
        std::fabs(imu.accel[0]) < 8.0f && std::fabs(imu.accel[1]) < 8.0f && std::fabs(imu.accel[2]) < 20.0f;
    const bool gyro_range =
        std::fabs(imu.gyro[0]) < 20.0f && std::fabs(imu.gyro[1]) < 20.0f && std::fabs(imu.gyro[2]) < 20.0f;
    health.imu_ok = imu_range && gyro_range;

    health.ultrasonic_ok = sonar.distance_m > 0.05f && sonar.distance_m < 5.0f && sonar.confidence > 0.2f;
    health.optical_ok = optic.intensity >= 0.0f && optic.intensity <= 2.0f;

    if (initialized_) {
        const uint32_t imu_delta = imu.timestamp_us - last_imu_.timestamp_us;
        const uint32_t sonar_delta = sonar.timestamp_us - last_sonar_.timestamp_us;
        const uint32_t optic_delta = optic.timestamp_us - last_optic_.timestamp_us;
        health.timestamp_sync = imu_delta < 20000 && sonar_delta < 40000 && optic_delta < 40000;
    } else {
        initialized_ = true;
    }

    last_imu_ = imu;
    last_sonar_ = sonar;
    last_optic_ = optic;
    return health;
}
}

