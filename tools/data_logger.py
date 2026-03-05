# Author : Ruben Rehal | Date : November 2025
"""
data_logger.py — Host-side UART capture utility for Orion hardware validation.
Connects to the MCU's serial port, reads comma-separated sensor/inference data
at 921600 baud, and writes aligned CSV logs for offline analysis. Each row
contains a full snapshot: IMU (6-axis), sonar, optical, and inference output.

Usage:
    python data_logger.py
    (edit the LoggerConfig at the bottom to match your serial port)
"""

from __future__ import annotations

import csv           # CSV writer for structured output
import pathlib       # Path handling for the output file
import time          # time.time() for duration-based capture
from dataclasses import dataclass  # Structured config container

import serial        # pyserial — UART communication with the MCU


@dataclass
class LoggerConfig:
    """Configuration for the serial capture session."""
    port: str                  # Serial port path, e.g., "/dev/tty.usbmodem0001"
    baud: int = 921600         # Baud rate — must match firmware UART config
    duration_s: int = 30       # How many seconds to capture


def capture(cfg: LoggerConfig) -> pathlib.Path:
    """
    Opens the serial port, reads lines for `duration_s` seconds, and writes
    each line as a CSV row. The firmware is expected to emit comma-separated
    values in the order defined by the CSV header below.

    Returns:
        Path to the written CSV file.
    """
    # Output file is created in the current working directory
    path = pathlib.Path("capture.csv")

    # Open both the serial port and the CSV file simultaneously
    with serial.Serial(cfg.port, cfg.baud, timeout=0.1) as ser, path.open("w", newline="") as fh:
        writer = csv.writer(fh)

        # Write the CSV header row describing each column
        writer.writerow(
            [
                "timestamp_us",    # MCU monotonic timestamp in microseconds
                "ax",              # Accelerometer X (g)
                "ay",              # Accelerometer Y (g)
                "az",              # Accelerometer Z (g)
                "gx",              # Gyroscope X (rad/s)
                "gy",              # Gyroscope Y (rad/s)
                "gz",              # Gyroscope Z (rad/s)
                "distance_m",      # Ultrasonic distance (metres)
                "sonar_conf",      # Ultrasonic confidence [0, 1]
                "opt_intensity",   # Optical intensity (normalised)
                "opt_delta",       # Optical frame-to-frame delta
                "confidence",      # Model-reported confidence [0, 1]
                "anomaly",         # Model-reported anomaly score [0, 1]
            ]
        )

        # Compute the wall-clock deadline for the capture session
        deadline = time.time() + cfg.duration_s

        # Read lines until the deadline expires
        while time.time() < deadline:
            # Read one line from the serial port (blocks up to timeout=0.1 s)
            line = ser.readline().decode().strip()

            # Skip empty lines (timeout with no data)
            if not line:
                continue

            # Split the comma-separated values and write as a CSV row
            writer.writerow(line.split(","))

    print(f"[TOOLS] wrote {path}")
    return path


if __name__ == "__main__":
    # Default config: connect to the first USB serial device for 30 seconds
    capture(LoggerConfig(port="/dev/tty.usbmodem0001"))
