#include <gtest/gtest.h>

#include "quad_controller/observation_builder.hpp"

#include <cmath>

// ---------------------------------------------------------------------------
// These cover the logic that decides whether the robot walks or falls over:
// the observation layout, the action scaling, and the gravity projection.
//
// None of it touches the GPU, so the suite runs anywhere, including CI machines
// without a CUDA device.
// ---------------------------------------------------------------------------

namespace {

constexpr float kTol = 1e-5f;

quad::RobotState makeState() {
    quad::RobotState state;
    state.base_linear_velocity  = {1.0f, 2.0f, 3.0f};
    state.base_angular_velocity = {4.0f, 5.0f, 6.0f};
    state.projected_gravity     = {0.0f, 0.0f, -1.0f};
    for (std::size_t i = 0; i < quad::ACT_DIM; ++i) {
        state.joint_positions[i]  = quad::DEFAULT_POSE[i] + 0.1f * static_cast<float>(i);
        state.joint_velocities[i] = 0.01f * static_cast<float>(i);
    }
    return state;
}

}  // namespace

// --- observation layout -----------------------------------------------------

TEST(ObservationBuilder, PlacesVelocitiesAndCommandInOrder) {
    const quad::ObservationBuilder builder;
    const quad::RobotState state = makeState();
    const quad::VelocityCommand command{0.5f, -0.25f, 0.75f};

    const quad::Observation obs = builder.build(state, command);

    EXPECT_NEAR(obs[0], 1.0f, kTol);
    EXPECT_NEAR(obs[1], 2.0f, kTol);
    EXPECT_NEAR(obs[2], 3.0f, kTol);

    EXPECT_NEAR(obs[3], 4.0f, kTol);
    EXPECT_NEAR(obs[4], 5.0f, kTol);
    EXPECT_NEAR(obs[5], 6.0f, kTol);

    EXPECT_NEAR(obs[6], 0.0f, kTol);
    EXPECT_NEAR(obs[7], 0.0f, kTol);
    EXPECT_NEAR(obs[8], -1.0f, kTol);

    EXPECT_NEAR(obs[9],  0.5f,  kTol);
    EXPECT_NEAR(obs[10], -0.25f, kTol);
    EXPECT_NEAR(obs[11], 0.75f, kTol);
}

// Joint positions enter the observation relative to the standing pose, not
// as absolute angles. Getting this wrong is silent and makes the robot fall.
TEST(ObservationBuilder, JointPositionsAreRelativeToDefaultPose) {
    const quad::ObservationBuilder builder;
    const quad::RobotState state = makeState();

    const quad::Observation obs = builder.build(state, {});

    for (std::size_t i = 0; i < quad::ACT_DIM; ++i) {
        EXPECT_NEAR(obs[12 + i], 0.1f * static_cast<float>(i), kTol)
            << "joint offset mismatch at index " << i;
    }
}

TEST(ObservationBuilder, AtDefaultPoseJointOffsetsAreZero) {
    const quad::ObservationBuilder builder;
    quad::RobotState state;
    state.joint_positions = quad::DEFAULT_POSE;

    const quad::Observation obs = builder.build(state, {});

    for (std::size_t i = 0; i < quad::ACT_DIM; ++i) {
        EXPECT_NEAR(obs[12 + i], 0.0f, kTol) << "index " << i;
    }
}

TEST(ObservationBuilder, JointVelocitiesCopiedVerbatim) {
    const quad::ObservationBuilder builder;
    const quad::RobotState state = makeState();

    const quad::Observation obs = builder.build(state, {});

    for (std::size_t i = 0; i < quad::ACT_DIM; ++i) {
        EXPECT_NEAR(obs[24 + i], 0.01f * static_cast<float>(i), kTol) << "index " << i;
    }
}

// --- previous action feedback ----------------------------------------------

TEST(ObservationBuilder, PreviousActionStartsZeroed) {
    const quad::ObservationBuilder builder;

    const quad::Observation obs = builder.build(quad::RobotState{}, {});

    for (std::size_t i = 36; i < quad::BASE_OBS_DIM; ++i) {
        EXPECT_NEAR(obs[i], 0.0f, kTol) << "index " << i;
    }
}

TEST(ObservationBuilder, PreviousActionIsFedBack) {
    quad::ObservationBuilder builder;

    quad::JointArray action{};
    for (std::size_t i = 0; i < quad::ACT_DIM; ++i) {
        action[i] = 0.5f * static_cast<float>(i);
    }
    builder.setPreviousAction(action);

    const quad::Observation obs = builder.build(quad::RobotState{}, {});

    for (std::size_t i = 0; i < quad::ACT_DIM; ++i) {
        EXPECT_NEAR(obs[36 + i], action[i], kTol) << "index " << i;
    }
}

TEST(ObservationBuilder, ResetClearsPreviousAction) {
    quad::ObservationBuilder builder;

    quad::JointArray action{};
    action.fill(1.0f);
    builder.setPreviousAction(action);
    builder.reset();

    const quad::Observation obs = builder.build(quad::RobotState{}, {});

    for (std::size_t i = 36; i < quad::BASE_OBS_DIM; ++i) {
        EXPECT_NEAR(obs[i], 0.0f, kTol) << "index " << i;
    }
}

// --- terrain height scan ----------------------------------------------------

// The scan occupies everything after the proprioceptive block. A misaligned
// copy here is silent: the policy simply reads terrain that isn't there.
TEST(HeightScan, OccupiesTheTailOfTheObservation) {
    const quad::ObservationBuilder builder;
    quad::RobotState state;
    for (std::size_t i = 0; i < quad::SCAN_DIM; ++i) {
        state.height_scan[i] = 0.01f * static_cast<float>(i);
    }

    const quad::Observation obs = builder.build(state, {});

    for (std::size_t i = 0; i < quad::SCAN_DIM; ++i) {
        EXPECT_NEAR(obs[quad::BASE_OBS_DIM + i], 0.01f * static_cast<float>(i), kTol)
            << "scan index " << i;
    }
}

TEST(HeightScan, DoesNotOverwriteTheProprioceptiveBlock) {
    const quad::ObservationBuilder builder;
    quad::RobotState state;
    state.base_linear_velocity = {1.0f, 2.0f, 3.0f};
    state.height_scan.fill(9.0f);

    const quad::Observation obs = builder.build(state, {});

    EXPECT_NEAR(obs[0], 1.0f, kTol);
    EXPECT_NEAR(obs[1], 2.0f, kTol);
    EXPECT_NEAR(obs[2], 3.0f, kTol);
    EXPECT_NEAR(obs[quad::BASE_OBS_DIM - 1], 0.0f, kTol);
    EXPECT_NEAR(obs[quad::BASE_OBS_DIM], 9.0f, kTol);
}

TEST(HeightScan, DimensionsAddUp) {
    EXPECT_EQ(quad::BASE_OBS_DIM + quad::SCAN_DIM, quad::OBS_DIM);
    EXPECT_EQ(quad::SCAN_DIM, 17 * 11);
}

// --- action scaling ---------------------------------------------------------

TEST(ActionScaling, ZeroActionHoldsDefaultPose) {
    const quad::JointArray action{};

    const quad::JointArray targets = quad::actionToJointTargets(action);

    for (std::size_t i = 0; i < quad::ACT_DIM; ++i) {
        EXPECT_NEAR(targets[i], quad::DEFAULT_POSE[i], kTol) << "index " << i;
    }
}

TEST(ActionScaling, AppliesScaleAsOffsetFromDefault) {
    quad::JointArray action{};
    action.fill(2.0f);

    const quad::JointArray targets = quad::actionToJointTargets(action);

    for (std::size_t i = 0; i < quad::ACT_DIM; ++i) {
        EXPECT_NEAR(targets[i],
                    quad::DEFAULT_POSE[i] + quad::ACTION_SCALE * 2.0f, kTol)
            << "index " << i;
    }
}

TEST(ActionScaling, HandlesNegativeActions) {
    quad::JointArray action{};
    action.fill(-1.0f);

    const quad::JointArray targets = quad::actionToJointTargets(action);

    for (std::size_t i = 0; i < quad::ACT_DIM; ++i) {
        EXPECT_NEAR(targets[i],
                    quad::DEFAULT_POSE[i] - quad::ACTION_SCALE, kTol)
            << "index " << i;
    }
}

// --- gravity projection -----------------------------------------------------

// Upright: gravity points straight down in the body frame.
TEST(ProjectedGravity, IdentityOrientationPointsDown) {
    const quad::Vec3 g = quad::projectedGravity(0.0f, 0.0f, 0.0f, 1.0f);

    EXPECT_NEAR(g[0], 0.0f, kTol);
    EXPECT_NEAR(g[1], 0.0f, kTol);
    EXPECT_NEAR(g[2], -1.0f, kTol);
}

// Rolled 180 degrees: gravity points along the body's +z.
TEST(ProjectedGravity, InvertedOrientationPointsUp) {
    const quad::Vec3 g = quad::projectedGravity(1.0f, 0.0f, 0.0f, 0.0f);

    EXPECT_NEAR(g[0], 0.0f, kTol);
    EXPECT_NEAR(g[1], 0.0f, kTol);
    EXPECT_NEAR(g[2], 1.0f, kTol);
}

// Pitched 90 degrees nose-down: gravity moves onto the body's +x axis.
TEST(ProjectedGravity, NinetyDegreePitchMovesGravityToBodyX) {
    const float s = std::sqrt(0.5f);
    const quad::Vec3 g = quad::projectedGravity(0.0f, s, 0.0f, s);

    EXPECT_NEAR(g[0], 1.0f, 1e-4f);
    EXPECT_NEAR(g[1], 0.0f, 1e-4f);
    EXPECT_NEAR(g[2], 0.0f, 1e-4f);
}

// Whatever the orientation, gravity is a unit vector.
TEST(ProjectedGravity, IsAlwaysUnitLength) {
    const float s = std::sqrt(0.5f);
    const std::array<quad::Vec3, 4> quats = {{
        {0.0f, 0.0f, 0.0f},   // placeholder, replaced below
    }};
    (void)quats;

    struct Q { float x, y, z, w; };
    const std::array<Q, 4> cases = {{
        {0.0f, 0.0f, 0.0f, 1.0f},
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, s,    0.0f, s   },
        {s,    0.0f, s,    0.0f},
    }};

    for (const auto& q : cases) {
        const quad::Vec3 g = quad::projectedGravity(q.x, q.y, q.z, q.w);
        const float norm = std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
        EXPECT_NEAR(norm, 1.0f, 1e-4f);
    }
}
