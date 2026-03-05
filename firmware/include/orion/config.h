// Author : Ruben Rehal | Date : November 2025
//
// config.h — Global compile-time constants for the Orion control loop and safety
// thresholds. Every tunable system-wide parameter lives here so that changing a
// single value propagates consistently across all modules.
#pragma once

namespace orion {

// Target control-loop frequency in Hz. All timing budgets (sensor readout,
// fusion, inference, safety) must complete within 1/kControlLoopHz seconds.
static constexpr float kControlLoopHz = 200.0f;

// Period of one control-loop iteration in milliseconds, derived from the
// frequency above. Used by main.cpp to sleep for the remaining budget.
static constexpr float kLoopPeriodMs = 1000.0f / kControlLoopHz;

// Minimum acceptable supply-rail voltage in volts. Below this the MCU can
// still run, but analog peripherals (ADC, sensor front-ends) become unreliable.
static constexpr float kVoltageMin = 3.0f;

// Critical voltage threshold in volts. Below this, SRAM retention and flash
// reads are no longer guaranteed, so the system must enter a safe fallback.
static constexpr float kVoltageCritical = 2.8f;

// Minimum inference confidence below which the output is considered unreliable
// and the system should log a warning but still operate normally.
static constexpr float kConfidenceMin = 0.35f;

// Hard fallback threshold. If the model's self-reported confidence drops below
// this, the system enters a safe fallback mode and stops actuating.
static constexpr float kConfidenceFallback = 0.2f;

}
