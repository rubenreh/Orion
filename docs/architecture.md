<!-- Author : Ruben Rehal | Date : November 2025 -->
# Orion Architecture Notes

## Control Stack
- **Acquisition** — IMU (I²C), Ultrasonic (PWM/TOF), Optical (SPI) read in interrupt-driven slots.
- **Fusion** — `fusion_pipeline.cpp` aligns timestamps, filters bias, and produces an 8-element feature vector.
- **Inference** — `inference_engine.cpp` executes three int8 dense layers with LUT ReLU/tanh approximations.
- **Safety** — watchdog, power monitor, and sensor sanity enforce deterministic recovery paths.

## Timing Budget (200 Hz)
- Sensor readout: 120 µs (burst DMA + moving-average filters).
- Fusion + sanity: 60 µs.
- Inference: 600 µs on Cortex-M7 @ 400 MHz (fully deterministic).
- Safety/control output: 80 µs.

## Memory Footprint
- Model weights: 256 B.
- Tensor arena: 64 B.
- Stack: 1 KB.
- Total static RAM < 8 KB, flash < 32 KB.

