// Author : Ruben Rehal | Date : November 2025
#pragma once

#include "system_types.h"

namespace orion {
class ImuDriver {
  public:
    virtual ~ImuDriver() = default;
    virtual ImuFrame Read() = 0;
};

class UltrasonicDriver {
  public:
    virtual ~UltrasonicDriver() = default;
    virtual UltrasonicFrame Read() = 0;
};

class OpticalDriver {
  public:
    virtual ~OpticalDriver() = default;
    virtual OpticalFrame Read() = 0;
};
}
