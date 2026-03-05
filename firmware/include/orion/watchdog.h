// Author : Ruben Rehal | Date : November 2025
//
// watchdog.h — Software watchdog timer modelled after hardware watchdog
// peripherals found on Cortex-M MCUs. The main loop must call Kick() every
// iteration; if more than timeout_us elapses without a kick the watchdog
// expires, indicating a hung or crashed control loop. This is a core
// ISO-26262-style safety mechanism for detecting timing faults.
#pragma once

#include <cstdint>  // uint32_t for microsecond timestamps

namespace orion {

// Software watchdog: must be kicked within timeout_us of the last kick,
// otherwise Expired() returns true and the system should reset.
class Watchdog {
  public:
    // Sets the timeout period (in microseconds) and arms the watchdog.
    // Typically configured to 2× the loop period (10 000 µs for a 200 Hz loop)
    // so that one missed deadline is tolerated but two consecutive misses trigger.
    void Configure(uint32_t timeout_us);

    // Records the current timestamp as proof that the loop completed on time.
    // Must be called once per iteration. Does nothing if the watchdog is unarmed.
    void Kick(uint32_t timestamp_us);

    // Returns true if more than timeout_us has elapsed since the last kick.
    // The main loop checks this after each iteration and breaks if expired.
    bool Expired(uint32_t timestamp_us) const;

  private:
    uint32_t timeout_us_ = 0;        // Configured timeout period in microseconds
    uint32_t last_kick_us_ = 0;      // Timestamp of the most recent Kick()
    bool armed_ = false;              // False until Configure() is called with timeout > 0
};

}
