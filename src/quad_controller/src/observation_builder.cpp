#include "quad_controller/observation_builder.hpp"

#include <cstddef>

namespace quad {

Observation ObservationBuilder::build(const RobotState& state,
                                      const VelocityCommand& command) const noexcept {
    Observation obs{};

    for (std::size_t i = 0; i < 3; ++i) {
        obs[i]     = state.base_linear_velocity[i];
        obs[3 + i] = state.base_angular_velocity[i];
        obs[6 + i] = state.projected_gravity[i];
    }

    obs[9]  = command.vx;
    obs[10] = command.vy;
    obs[11] = command.wz;

    for (std::size_t i = 0; i < ACT_DIM; ++i) {
        obs[12 + i] = state.joint_positions[i] - DEFAULT_POSE[i];
        obs[24 + i] = state.joint_velocities[i];
        obs[36 + i] = previous_action_[i];
    }

    for (std::size_t i = 0; i < SCAN_DIM; ++i) {
        obs[BASE_OBS_DIM + i] = state.height_scan[i];
    }

    return obs;
}

JointArray actionToJointTargets(const JointArray& action) noexcept {
    JointArray targets{};
    for (std::size_t i = 0; i < ACT_DIM; ++i) {
        targets[i] = DEFAULT_POSE[i] + ACTION_SCALE * action[i];
    }
    return targets;
}

Vec3 projectedGravity(float x, float y, float z, float w) noexcept {
    // g_body = R^T * g_world, with g_world = (0, 0, -1) and R the
    // body-to-world rotation built from the quaternion. Only the third column
    // of R is needed, negated.
    return Vec3{-(2.0f * (x * z - w * y)),
                -(2.0f * (y * z + w * x)),
                -(1.0f - 2.0f * (x * x + y * y))};
}

}  // namespace quad
