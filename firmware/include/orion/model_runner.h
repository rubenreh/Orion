// Author : Ruben Rehal | Date : November 2025
//
// model_runner.h — High-level orchestrator that wires together the fusion
// pipeline, sensor sanity checker, and inference engine into a single Step()
// call. The main loop interacts with this class exclusively, keeping the
// control loop clean and the internal data flow encapsulated.
#pragma once

#include "fusion_pipeline.h"    // FusionPipeline — sensor filtering + feature packing
#include "inference_engine.h"   // InferenceEngine — int8 neural-network runtime
#include "sensor_sanity.h"      // SensorSanityChecker — plausibility validation
#include "system_types.h"       // ImuFrame, UltrasonicFrame, OpticalFrame, etc.

namespace orion {

// Facade that accepts raw sensor frames and returns an InferenceOutput in one
// call, while also caching the intermediate FusedFrame and SensorHealth for
// inspection by the main loop.
class ModelRunner {
  public:
    // Constructs internal sub-systems and loads the calibrated neural-network model.
    ModelRunner();

    // Executes one full cycle: fuse sensors → check sanity → run inference.
    InferenceOutput Step(const ImuFrame& imu,
                         const UltrasonicFrame& sonar,
                         const OpticalFrame& optic);

    // Accessors for the intermediate results from the most recent Step().
    const FusedFrame& LastFusion() const { return fused_; }
    const SensorHealth& LastHealth() const { return health_; }

  private:
    FusionPipeline fusion_;            // Filters and packs raw sensor data
    InferenceEngine engine_;           // Runs the quantised MLP
    SensorSanityChecker sanity_;       // Validates sensor plausibility
    FusedFrame fused_{};               // Cached fused feature vector
    SensorHealth health_{};            // Cached sensor health flags
};

}
