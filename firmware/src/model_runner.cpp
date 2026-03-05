// Author : Ruben Rehal | Date : November 2025
//
// model_runner.cpp — High-level orchestrator that wires together the fusion
// pipeline, sensor sanity checker, and inference engine. The main loop calls
// Step() with raw sensor frames and gets back an InferenceOutput, keeping the
// control loop clean while all the internal data flow stays encapsulated.

#include "orion/model_runner.h"

#include "orion/config.h"      // (available for threshold access if needed)
#include "orion/model_data.h"  // model::GetCalibratedModel

namespace orion {

// Loads the pre-quantised calibrated model into the inference engine at
// construction time so that the first Step() call is ready to run immediately.
ModelRunner::ModelRunner() {
    engine_.LoadModel(model::GetCalibratedModel());
}

// Executes one full sense-fuse-infer cycle:
//   1. FusionPipeline::Process() — filters raw sensors, packs 8-D feature vector
//   2. SensorSanityChecker::Evaluate() — validates plausibility of raw readings
//   3. InferenceEngine::Run() — quantises features, runs 3-layer MLP, returns output
// The intermediate FusedFrame and SensorHealth are cached for inspection via
// LastFusion() and LastHealth().
InferenceOutput ModelRunner::Step(const ImuFrame& imu,
                                  const UltrasonicFrame& sonar,
                                  const OpticalFrame& optic) {
    // Fuse: filter each modality and pack into the 8-element feature vector
    fused_ = fusion_.Process(imu, sonar, optic);

    // Sanity-check: validate ranges and timestamp synchronisation
    health_ = sanity_.Evaluate(imu, sonar, optic);

    // Infer: quantise the fused features and run the int8 MLP
    return engine_.Run(fused_);
}

}
