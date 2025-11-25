// Author : Ruben Rehal | Date : November 2025
#include "orion/model_runner.h"

#include "orion/config.h"
#include "orion/model_data.h"

namespace orion {
ModelRunner::ModelRunner() {
    engine_.LoadModel(model::GetCalibratedModel());
}

InferenceOutput ModelRunner::Step(const ImuFrame& imu,
                                  const UltrasonicFrame& sonar,
                                  const OpticalFrame& optic) {
    fused_ = fusion_.Process(imu, sonar, optic);
    health_ = sanity_.Evaluate(imu, sonar, optic);
    return engine_.Run(fused_);
}
}

