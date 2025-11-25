// Author : Ruben Rehal | Date : November 2025
#pragma once

#include <cstdint>

namespace orion {
class Watchdog {
  public:
    void Configure(uint32_t timeout_us);
    void Kick(uint32_t timestamp_us);
    bool Expired(uint32_t timestamp_us) const;

  private:
    uint32_t timeout_us_ = 0;
    uint32_t last_kick_us_ = 0;
    bool armed_ = false;
};
}

