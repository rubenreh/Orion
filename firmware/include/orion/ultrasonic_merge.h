// Author : Ruben Rehal | Date : November 2025
#pragma once

#include "system_types.h"

namespace orion {
class UltrasonicMerge {
  public:
    UltrasonicMerge();
    UltrasonicFrame Filter(const UltrasonicFrame& raw);

  private:
    UltrasonicFrame state_{};
    bool initialized_ = false;
};
}
