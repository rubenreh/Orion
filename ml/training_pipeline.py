# Author : Ruben Rehal | Date : November 2025
"""
training_pipeline.py — Minimal end-to-end training pipeline for the Orion
neural network. Generates synthetic fused sensor frames, trains a 3-layer MLP
(8→12→6→2) from scratch using numpy-only backpropagation, and exports the
weights as int8-quantised JSON that can be manually copied into model_data.cpp.

This file demonstrates the full ML lifecycle for an embedded model:
  1. Synthetic data generation (simulating sensor behaviour)
  2. Forward pass + backprop training (vanilla SGD, MSE loss)
  3. Post-training quantisation (float → int8 with per-layer scales)
  4. Weight export for firmware integration
"""

from __future__ import annotations

import json                       # JSON serialisation for weight export
import math                       # math.pi for sinusoidal data generation
from dataclasses import dataclass # Structured config container
from pathlib import Path          # File path handling
from typing import List, Tuple   # Type hints for function signatures

import numpy as np                # Numerical computing — the only dependency


@dataclass
class TrainingConfig:
    """Hyperparameters and architecture constants for training."""
    samples: int = 2048            # Number of synthetic training samples
    input_dim: int = 8             # Must match model::kInputDim in firmware
    hidden1: int = 12              # Must match model::kHidden1
    hidden2: int = 6               # Must match model::kHidden2
    output_dim: int = 2            # Must match model::kOutputDim
    learning_rate: float = 1e-3    # SGD learning rate
    epochs: int = 150              # Number of training iterations


def synthetic_dataset(cfg: TrainingConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates synthetic training data that mimics the 8-D fused sensor features
    produced by FusionPipeline. Labels are derived from the sonar gradient
    (anomaly = sudden distance change) and a confidence inversely related to
    anomaly severity.

    Returns:
        features: (samples, 8) float32 array of synthetic sensor features
        labels:   (samples, 2) float32 array of [anomaly, confidence] targets
    """
    # Create a time axis spanning 10 seconds with `samples` evenly-spaced points
    time = np.linspace(0.0, 10.0, cfg.samples)

    # Simulate individual sensor signals with sinusoidal patterns
    accel = 1.0 + 0.1 * np.sin(2 * math.pi * 0.5 * time)   # Accel magnitude ~1g ± 0.1
    gyro = 0.01 * np.cos(2 * math.pi * 0.25 * time)          # Gyro magnitude ~0.01 rad/s
    sonar = 0.5 + 0.2 * np.sin(2 * math.pi * 0.1 * time)    # Sonar distance ~0.5m ± 0.2
    optic = 0.5 + 0.3 * np.cos(2 * math.pi * 0.33 * time)   # Optical intensity ~0.5 ± 0.3

    # Stack 8 features to match the FusedFrame layout:
    #   [accel_mag, gyro_mag, sonar_dist, optic_intensity,
    #    d(accel)/dt, d(sonar)/dt, d(optic)/dt, d(gyro)/dt]
    features = np.stack(
        [
            accel,               # Feature 0: accel magnitude
            np.abs(gyro),        # Feature 1: gyro magnitude (absolute value)
            sonar,               # Feature 2: sonar distance
            optic,               # Feature 3: optical intensity
            np.gradient(accel),  # Feature 4: rate of change of accel
            np.gradient(sonar),  # Feature 5: rate of change of sonar
            np.gradient(optic),  # Feature 6: rate of change of optical
            np.gradient(gyro),   # Feature 7: rate of change of gyro
        ],
        axis=1,
    )

    # Labels: anomaly is the absolute rate of change of sonar distance,
    # clipped to [0, 1]. A sudden distance change = high anomaly.
    anomaly = np.clip(np.abs(np.gradient(sonar, time)), 0.0, 1.0)

    # Confidence is inversely related to anomaly: high anomaly → lower confidence
    confidence = 1.0 - anomaly * 0.5

    # Stack into (samples, 2) label array: [anomaly, confidence]
    labels = np.stack([anomaly, confidence], axis=1)

    return features.astype(np.float32), labels.astype(np.float32)


class TinyMlp:
    """
    A from-scratch 3-layer MLP implemented in pure numpy. Matches the firmware
    architecture exactly (8→12→6→2) so that quantised weights can be directly
    exported to model_data.cpp.

    Training uses vanilla SGD with MSE loss and manual backpropagation.
    """

    def __init__(self, cfg: TrainingConfig):
        """Initialises weights with small random normals and biases with zeros."""
        self.cfg = cfg
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility

        # Layer 1: (8, 12) — input to first hidden
        self.w1 = rng.normal(0, 0.1, (cfg.input_dim, cfg.hidden1))
        self.b1 = np.zeros(cfg.hidden1)

        # Layer 2: (12, 6) — first hidden to second hidden
        self.w2 = rng.normal(0, 0.1, (cfg.hidden1, cfg.hidden2))
        self.b2 = np.zeros(cfg.hidden2)

        # Layer 3: (6, 2) — second hidden to output
        self.w3 = rng.normal(0, 0.1, (cfg.hidden2, cfg.output_dim))
        self.b3 = np.zeros(cfg.output_dim)

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation: max(0, x). Used for hidden layers."""
        return np.maximum(0, x)

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """Tanh activation: squashes to [-1, +1]. Used for the output layer."""
        return np.tanh(x)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Forward pass through all three layers.

        Returns:
            out:    (batch, 2) output predictions
            caches: list of hidden activations [h1, h2] needed for backprop
        """
        # Layer 1: linear transform + ReLU activation
        h1 = self.relu(x @ self.w1 + self.b1)

        # Layer 2: linear transform + ReLU activation
        h2 = self.relu(h1 @ self.w2 + self.b2)

        # Layer 3 (output): linear transform + Tanh activation
        out = self.tanh(h2 @ self.w3 + self.b3)

        return out, [h1, h2]  # Return hidden activations for backprop

    def train(self, x: np.ndarray, y: np.ndarray, cfg: TrainingConfig):
        """
        Trains the model using vanilla SGD with MSE loss.
        Manually computes gradients via backpropagation through each layer.
        """
        for epoch in range(cfg.epochs):
            # ── Forward pass ──────────────────────────────────────────────
            pred, caches = self.forward(x)

            # MSE loss: mean of squared differences between prediction and target
            loss = np.mean((pred - y) ** 2)

            # ── Backward pass ─────────────────────────────────────────────
            # Gradient of MSE loss w.r.t. predictions: d_loss/d_pred
            grad = 2 * (pred - y) / len(x)

            # Layer 3 gradients (output layer)
            grad_w3 = caches[1].T @ grad     # d_loss/d_w3 = h2^T · grad
            grad_b3 = grad.sum(axis=0)        # d_loss/d_b3 = sum of grad over batch

            # Backprop through layer 3 into layer 2
            grad_h2 = grad @ self.w3.T        # d_loss/d_h2 = grad · w3^T
            grad_h2[caches[1] <= 0] = 0       # Zero out gradients where ReLU was inactive

            # Layer 2 gradients
            grad_w2 = caches[0].T @ grad_h2   # d_loss/d_w2 = h1^T · grad_h2
            grad_b2 = grad_h2.sum(axis=0)     # d_loss/d_b2 = sum over batch

            # Backprop through layer 2 into layer 1
            grad_h1 = grad_h2 @ self.w2.T     # d_loss/d_h1 = grad_h2 · w2^T
            grad_h1[caches[0] <= 0] = 0       # Zero out gradients where ReLU was inactive

            # Layer 1 gradients
            grad_w1 = x.T @ grad_h1           # d_loss/d_w1 = x^T · grad_h1
            grad_b1 = grad_h1.sum(axis=0)     # d_loss/d_b1 = sum over batch

            # ── SGD parameter update ──────────────────────────────────────
            # Subtract learning_rate × gradient from each parameter
            for param, g in [
                (self.w3, grad_w3),
                (self.b3, grad_b3),
                (self.w2, grad_w2),
                (self.b2, grad_b2),
                (self.w1, grad_w1),
                (self.b1, grad_b1),
            ]:
                param -= cfg.learning_rate * g  # In-place update

            # Log the loss every 25 epochs to monitor convergence
            if epoch % 25 == 0:
                print(f"[ML] epoch={epoch} loss={loss:.5f}")


def quantize_to_int8(array: np.ndarray, scale: float) -> List[int]:
    """
    Post-training quantisation: converts a float weight matrix to int8 by
    dividing by `scale`, rounding to nearest integer, and clamping to [-128, 127].
    This mirrors the quant::Quantize() function in the firmware.
    """
    scaled = np.clip(np.round(array / scale), -128, 127)
    return scaled.astype(np.int8).tolist()


def emit_header(weights: dict, path: Path):
    """
    Writes the quantised weight dictionary as JSON. The JSON output is then
    manually transcribed into the constexpr arrays in model_data.cpp.
    """
    path.write_text(json.dumps(weights, indent=2))
    print(f"[ML] wrote weights to {path}")


def main():
    """
    End-to-end pipeline: generate data → train model → quantise → export.
    """
    cfg = TrainingConfig()

    # Step 1: Generate synthetic sensor data matching the FusedFrame layout
    x, y = synthetic_dataset(cfg)

    # Step 2: Instantiate and train the MLP
    model = TinyMlp(cfg)
    model.train(x, y, cfg)

    # Step 3: Quantise weights and biases for firmware deployment.
    # Weights are quantised with scale=0.03125 (= 1/32 = 2^-5).
    # Biases are scaled to match the int32 accumulator: bias_q = bias / (dim * input_scale * weight_scale).
    payload = {
        "w1": quantize_to_int8(model.w1.T, 0.03125),                                           # Layer 1 weights (transposed to row-major)
        "b1": (model.b1 / (cfg.input_dim * 0.02 * 0.03125)).round().astype(int).tolist(),       # Layer 1 biases
        "w2": quantize_to_int8(model.w2.T, 0.03125),                                           # Layer 2 weights
        "b2": (model.b2 / (cfg.hidden1 * 0.04 * 0.03125)).round().astype(int).tolist(),         # Layer 2 biases
        "w3": quantize_to_int8(model.w3.T, 0.03125),                                           # Layer 3 weights
        "b3": (model.b3 / (cfg.hidden2 * 0.05 * 0.03125)).round().astype(int).tolist(),         # Layer 3 biases
    }

    # Step 4: Write the quantised weights as JSON alongside this script
    emit_header(payload, Path(__file__).with_suffix(".json"))


if __name__ == "__main__":
    main()
