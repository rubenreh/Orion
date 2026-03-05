// Author : Ruben Rehal | Date : November 2025
//
// imu_driver.h — Mock IMU driver used for desktop simulation and testing.
// Implements the abstract ImuDriver interface with deterministic sinusoidal
// data so that the full pipeline can be exercised without real hardware.
#pragma once

#include "sensor_interfaces.h"  // ImuDriver base class, ImuFrame

namespace orion {

// Simulated IMU that generates a gentle sinusoidal wobble around 1 g on the
// Z-axis. Timestamps auto-increment at the 200 Hz loop rate (5 000 µs per
// sample), mimicking a real sensor's clock.
class MockImuDriver : public ImuDriver {
  public:
    MockImuDriver();                   // Initialises the frame to a stationary 1 g reading
    ImuFrame Read() override;          // Advances time by 5 ms and returns a new wobble sample

  private:
    ImuFrame frame_;                   // Persistent state between Read() calls
};

}
