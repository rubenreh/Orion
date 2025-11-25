// Author : Ruben Rehal | Date : November 2025
#pragma once

namespace orion {
static constexpr float kControlLoopHz = 200.0f;
static constexpr float kLoopPeriodMs = 1000.0f / kControlLoopHz;
static constexpr float kVoltageMin = 3.0f;
static constexpr float kVoltageCritical = 2.8f;
static constexpr float kConfidenceMin = 0.35f;
static constexpr float kConfidenceFallback = 0.2f;
}
