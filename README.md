<!-- Author : Ruben Rehal | Date : November 2025 -->
# Orion — Edge-AI Sensor Fusion Controller

Orion is an embedded intelligence stack that fuses IMU, ultrasonic, and optical sensors, executes an int8 neural model, and enforces automotive-style safety guarantees on ultra-low-power MCUs.

## Project Layout
- `firmware/` — production C++ firmware with drivers, fusion, inference, and safety layers.
- `ml/` — lightweight Python utilities for training and exporting quantized weights.
- `tools/` — host-side helpers for data capture and hardware validation.
- `docs/` — supplementary design briefs and timing notes.

## Feature Highlights
1. Deterministic 200 Hz control loop.
2. Custom fixed-point inference core (dense layers + LUT activations).
3. Sensor sanity + watchdog + brownout guard rails.
4. Confidence-gated outputs and fallback behavior.

## Status
The repository provides firmware-ready scaffolding with realistic algorithms and hooks for real sensor integration, intended to showcase embedded AI engineering depth.
