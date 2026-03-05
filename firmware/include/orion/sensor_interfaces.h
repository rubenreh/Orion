// Author : Ruben Rehal | Date : November 2025
//
// sensor_interfaces.h — Abstract base classes (interfaces) for every sensor
// modality used by Orion. Concrete drivers (mock or real hardware) inherit from
// these, allowing the rest of the stack to be hardware-agnostic. This is the
// dependency-injection seam that makes the system testable and portable.
#pragma once

#include "system_types.h"  // ImuFrame, UltrasonicFrame, OpticalFrame

namespace orion {

// Abstract interface for an IMU sensor driver.
// Concrete implementations read from real I²C/SPI hardware or produce mock data.
class ImuDriver {
  public:
    virtual ~ImuDriver() = default;           // Virtual destructor for safe polymorphic deletion
    virtual ImuFrame Read() = 0;              // Returns the latest IMU sample
};

// Abstract interface for an ultrasonic range-finder driver.
// Concrete implementations drive the trigger/echo pins or produce mock data.
class UltrasonicDriver {
  public:
    virtual ~UltrasonicDriver() = default;    // Virtual destructor for safe polymorphic deletion
    virtual UltrasonicFrame Read() = 0;       // Returns the latest sonar sample
};

// Abstract interface for an optical (light/reflectance) sensor driver.
// Concrete implementations read from SPI/ADC hardware or produce mock data.
class OpticalDriver {
  public:
    virtual ~OpticalDriver() = default;       // Virtual destructor for safe polymorphic deletion
    virtual OpticalFrame Read() = 0;          // Returns the latest optical sample
};

}
