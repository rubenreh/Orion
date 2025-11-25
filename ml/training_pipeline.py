# Author : Ruben Rehal | Date : November 2025
"""
Orion minimal training pipeline. Generates synthetic fused sensor frames,
trains a tiny MLP, and emits quantized weights for `model_data.cpp`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


@dataclass
class TrainingConfig:
    samples: int = 2048
    input_dim: int = 8
    hidden1: int = 12
    hidden2: int = 6
    output_dim: int = 2
    learning_rate: float = 1e-3
    epochs: int = 150


def synthetic_dataset(cfg: TrainingConfig) -> Tuple[np.ndarray, np.ndarray]:
    time = np.linspace(0.0, 10.0, cfg.samples)
    accel = 1.0 + 0.1 * np.sin(2 * math.pi * 0.5 * time)
    gyro = 0.01 * np.cos(2 * math.pi * 0.25 * time)
    sonar = 0.5 + 0.2 * np.sin(2 * math.pi * 0.1 * time)
    optic = 0.5 + 0.3 * np.cos(2 * math.pi * 0.33 * time)
    features = np.stack(
        [
            accel,
            np.abs(gyro),
            sonar,
            optic,
            np.gradient(accel),
            np.gradient(sonar),
            np.gradient(optic),
            np.gradient(gyro),
        ],
        axis=1,
    )
    anomaly = np.clip(np.abs(np.gradient(sonar, time)), 0.0, 1.0)
    confidence = 1.0 - anomaly * 0.5
    labels = np.stack([anomaly, confidence], axis=1)
    return features.astype(np.float32), labels.astype(np.float32)


class TinyMlp:
    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg
        rng = np.random.default_rng(42)
        self.w1 = rng.normal(0, 0.1, (cfg.input_dim, cfg.hidden1))
        self.b1 = np.zeros(cfg.hidden1)
        self.w2 = rng.normal(0, 0.1, (cfg.hidden1, cfg.hidden2))
        self.b2 = np.zeros(cfg.hidden2)
        self.w3 = rng.normal(0, 0.1, (cfg.hidden2, cfg.output_dim))
        self.b3 = np.zeros(cfg.output_dim)

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        h1 = self.relu(x @ self.w1 + self.b1)
        h2 = self.relu(h1 @ self.w2 + self.b2)
        out = self.tanh(h2 @ self.w3 + self.b3)
        return out, [h1, h2]

    def train(self, x: np.ndarray, y: np.ndarray, cfg: TrainingConfig):
        for epoch in range(cfg.epochs):
            pred, caches = self.forward(x)
            loss = np.mean((pred - y) ** 2)
            grad = 2 * (pred - y) / len(x)

            grad_w3 = caches[1].T @ grad
            grad_b3 = grad.sum(axis=0)

            grad_h2 = grad @ self.w3.T
            grad_h2[caches[1] <= 0] = 0

            grad_w2 = caches[0].T @ grad_h2
            grad_b2 = grad_h2.sum(axis=0)

            grad_h1 = grad_h2 @ self.w2.T
            grad_h1[caches[0] <= 0] = 0

            grad_w1 = x.T @ grad_h1
            grad_b1 = grad_h1.sum(axis=0)

            for param, g in [
                (self.w3, grad_w3),
                (self.b3, grad_b3),
                (self.w2, grad_w2),
                (self.b2, grad_b2),
                (self.w1, grad_w1),
                (self.b1, grad_b1),
            ]:
                param -= cfg.learning_rate * g

            if epoch % 25 == 0:
                print(f"[ML] epoch={epoch} loss={loss:.5f}")


def quantize_to_int8(array: np.ndarray, scale: float) -> List[int]:
    scaled = np.clip(np.round(array / scale), -128, 127)
    return scaled.astype(np.int8).tolist()


def emit_header(weights: dict, path: Path):
    path.write_text(json.dumps(weights, indent=2))
    print(f"[ML] wrote weights to {path}")


def main():
    cfg = TrainingConfig()
    x, y = synthetic_dataset(cfg)
    model = TinyMlp(cfg)
    model.train(x, y, cfg)

    payload = {
        "w1": quantize_to_int8(model.w1.T, 0.03125),
        "b1": (model.b1 / (cfg.input_dim * 0.02 * 0.03125)).round().astype(int).tolist(),
        "w2": quantize_to_int8(model.w2.T, 0.03125),
        "b2": (model.b2 / (cfg.hidden1 * 0.04 * 0.03125)).round().astype(int).tolist(),
        "w3": quantize_to_int8(model.w3.T, 0.03125),
        "b3": (model.b3 / (cfg.hidden2 * 0.05 * 0.03125)).round().astype(int).tolist(),
    }

    emit_header(payload, Path(__file__).with_suffix(".json"))


if __name__ == "__main__":
    main()

