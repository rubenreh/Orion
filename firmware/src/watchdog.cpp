// Author : Ruben Rehal | Date : November 2025
//
// watchdog.cpp — Software watchdog timer implementation. Models the behaviour
// of a hardware watchdog peripheral: the control loop must call Kick() within
// the configured timeout, otherwise Expired() returns true, indicating a
// timing fault (hung loop, deadlock, or unexpected blocking). The main loop
// uses this to detect and react to overruns — on real hardware this would
// trigger a system reset via the NVIC.

#include "orion/watchdog.h"

namespace orion {

// Arms the watchdog with the given timeout in microseconds and resets the
// last-kick timestamp. Called once during initialisation with 2× the loop
// period (10 000 µs for a 200 Hz loop).
void Watchdog::Configure(uint32_t timeout_us) {
    timeout_us_ = timeout_us;         // Store the timeout threshold
    armed_ = timeout_us_ > 0;         // Arm only if timeout is non-zero
    last_kick_us_ = 0;                // Reset the kick timestamp
}

// Records the current timestamp as proof that the control loop completed
// its iteration on time. Must be called once per loop iteration.
void Watchdog::Kick(uint32_t timestamp_us) {
    if (!armed_) return;              // No-op if the watchdog was never configured
    last_kick_us_ = timestamp_us;     // Update the last-kick timestamp
}

// Returns true if the watchdog has expired (i.e., more time than timeout_us_
// has elapsed since the last Kick()). The main loop checks this after each
// iteration and breaks out of the loop if it returns true.
bool Watchdog::Expired(uint32_t timestamp_us) const {
    if (!armed_) return false;        // Unarmed watchdog never expires
    if (last_kick_us_ == 0) return false;  // Never been kicked yet — don't false-trigger
    // Check if elapsed time since last kick exceeds the configured timeout
    return (timestamp_us - last_kick_us_) > timeout_us_;
}

}
