#pragma once

#include <array>
#include <cstddef>

// ---------------------------------------------------------------------------
// Types describing the policy's input and output contract.
//
// Deliberately free of TensorRT, CUDA, and ROS includes so the logic that uses
// them can be compiled and tested on any machine.
// ---------------------------------------------------------------------------

namespace quad {

inline constexpr int   BASE_OBS_DIM = 48;   // proprioception + command
inline constexpr int   SCAN_DIM     = 187;  // 17 x 11 terrain height grid
inline constexpr int   OBS_DIM      = BASE_OBS_DIM + SCAN_DIM;
inline constexpr int   ACT_DIM      = 12;
inline constexpr float ACTION_SCALE = 0.5f;

// DOF order as reported by the Isaac Sim articulation.
inline constexpr std::array<const char*, ACT_DIM> JOINT_NAMES = {
    "LF_HAA", "LH_HAA", "RF_HAA", "RH_HAA",
    "LF_HFE", "LH_HFE", "RF_HFE", "RH_HFE",
    "LF_KFE", "LH_KFE", "RF_KFE", "RH_KFE"};

// Nominal standing pose, in the same order.
inline constexpr std::array<float, ACT_DIM> DEFAULT_POSE = {
    0.0f,  0.0f,  0.0f,  0.0f,
    0.4f, -0.4f,  0.4f, -0.4f,
   -0.8f,  0.8f, -0.8f,  0.8f};

using Vec3        = std::array<float, 3>;
using JointArray  = std::array<float, ACT_DIM>;
using ScanArray   = std::array<float, SCAN_DIM>;
using Observation = std::array<float, OBS_DIM>;

// Everything the policy needs to know about the robot this tick.
struct RobotState {
    Vec3       base_linear_velocity{};                  // body frame
    Vec3       base_angular_velocity{};                 // body frame
    Vec3       projected_gravity{0.0f, 0.0f, -1.0f};    // body frame
    JointArray joint_positions{};
    JointArray joint_velocities{};
    ScanArray  height_scan{};                           // base_z - hit_z - 0.5
};

// Desired base velocity: forward, lateral, yaw rate.
struct VelocityCommand {
    float vx{0.0f};
    float vy{0.0f};
    float wz{0.0f};
};

}  // namespace quad
