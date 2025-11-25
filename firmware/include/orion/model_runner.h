// Author : Ruben Rehal | Date : November 2025
#pragma once

#include "fusion_pipeline.h"
#include "inference_engine.h"
#include "sensor_sanity.h"
#include "system_types.h"

namespace orion {
class ModelRunner {
  public:
    ModelRunner();
    InferenceOutput Step(const ImuFrame& imu,
                         const UltrasonicFrame& sonar,
                         const OpticalFrame& optic);
    const FusedFrame& LastFusion() const { return fused_; }
    const SensorHealth& LastHealth() const { return health_; }

  private:
    FusionPipeline fusion_;
    InferenceEngine engine_;
    SensorSanityChecker sanity_;
    FusedFrame fused_{};
    SensorHealth health_{};
};
}

