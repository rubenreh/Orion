# Author : Ruben Rehal | Date : November 2025
"""
Host-side helper that streams sensor frames over UART and stores aligned CSV logs.
"""

from __future__ import annotations

import csv
import pathlib
import time
from dataclasses import dataclass

import serial


@dataclass
class LoggerConfig:
    port: str
    baud: int = 921600
    duration_s: int = 30


def capture(cfg: LoggerConfig) -> pathlib.Path:
    path = pathlib.Path("capture.csv")
    with serial.Serial(cfg.port, cfg.baud, timeout=0.1) as ser, path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "timestamp_us",
                "ax",
                "ay",
                "az",
                "gx",
                "gy",
                "gz",
                "distance_m",
                "sonar_conf",
                "opt_intensity",
                "opt_delta",
                "confidence",
                "anomaly",
            ]
        )
        deadline = time.time() + cfg.duration_s
        while time.time() < deadline:
            line = ser.readline().decode().strip()
            if not line:
                continue
            writer.writerow(line.split(","))
    print(f"[TOOLS] wrote {path}")
    return path


if __name__ == "__main__":
    capture(LoggerConfig(port="/dev/tty.usbmodem0001"))

