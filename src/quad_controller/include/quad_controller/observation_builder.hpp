#pragma once

#include "quad_controller/policy_types.hpp"

// ---------------------------------------------------------------------------
// Translation between robot state and the policy's observation vector.
//
// Layout, fixed by how the policy was trained:
//   [ 0: 3] base linear velocity   (body frame)
//   [ 3: 6] base angular velocity  (body frame)
//   [ 6: 9] projected gravity      (body frame)
//   [ 9:12] velocity command
//   [12:24] joint position, relative to the nominal standing pose
//   [24:36] joint velocity
//   [36:48] the action produced on the previous tick
//
// The last block makes this stateful: the policy is fed its own output from the
// previous control step, so the builder owns that history.
// ---------------------------------------------------------------------------

namespace quad {

class ObservationBuilder {
public:
    [[nodiscard]] Observation build(const RobotState& state,
                                    const VelocityCommand& command) const noexcept;

    void setPreviousAction(const JointArray& action) noexcept { previous_action_ = action; }

    [[nodiscard]] const JointArray& previousAction() const noexcept { return previous_action_; }

    void reset() noexcept { previous_action_.fill(0.0f); }

private:
    JointArray previous_action_{};
};

// Policy actions are scaled offsets from the nominal pose, not absolute targets.
[[nodiscard]] JointArray actionToJointTargets(const JointArray& action) noexcept;

// Rotate world gravity into the body frame, given the body orientation as a
// quaternion. This is how the policy perceives which way is down.
[[nodiscard]] Vec3 projectedGravity(float x, float y, float z, float w) noexcept;

}  // namespace quad
