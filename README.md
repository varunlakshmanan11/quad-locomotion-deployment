# Quadruped Locomotion Policy Deployment

A reinforcement learning locomotion policy trained in NVIDIA Isaac Lab, compiled
to a TensorRT engine, and deployed as a real-time C++ ROS 2 node that closes the
control loop against Isaac Sim. The robot perceives the terrain under it through
a simulated height scanner, walks over uneven ground, and navigates a waypoint
course autonomously.

The interesting part of this project is not the policy. It is what happens
between "the policy is trained" and "the robot moves", which is where the real
engineering lives.

---

## Pipeline

```
Isaac Lab (RSL-RL, PPO)
      |  ONNX export
policy.onnx                       235 observations -> 12 joint actions
      |  trtexec
policy.engine                     GPU-specific optimized binary
      |
quad_node (C++)                   assembles observations, runs inference,
      ^                           publishes joint position targets
      |
Isaac Sim                         Anymal-C, procedural rough terrain,
                                  187-ray height scanner
```

Two ROS 2 nodes, both C++:

- **`quad_node`** is the controller. It consumes `/joint_states`, `/odom` and
  `/height_scan`, assembles the observation vector, runs TensorRT inference, and
  publishes joint position targets on `/joint_command`.
- **`waypoint_navigator`** is the planner above it. It reads the robot pose and
  publishes velocity commands on `/cmd_vel` to steer a closed course.

---

## The observation contract

The policy consumes 235 values in a fixed layout. Getting any block wrong fails
silently: nothing errors, the robot simply falls over.

| Slots | Content |
|---|---|
| 0-2 | base linear velocity, body frame |
| 3-5 | base angular velocity, body frame |
| 6-8 | projected gravity, body frame |
| 9-11 | velocity command (vx, vy, wz) |
| 12-23 | joint positions, relative to the nominal standing pose |
| 24-35 | joint velocities |
| 36-47 | the policy's own action from the previous tick |
| 48-234 | terrain height scan, 17 x 11 grid |

Actions are scaled offsets rather than absolute targets:

```
target_joint_pos = default_pose + 0.5 * action
```

Three details that are easy to get wrong and impossible to notice afterwards:
joint positions enter as a **difference** from the standing pose; the previous
action is fed back in, so the controller is stateful across ticks; and projected
gravity is computed by rotating world gravity into the body frame, which is how
the policy perceives its own tilt.

The height scan grid was not assumed. Its ordering was extracted directly from
Isaac Lab's own pattern generator: x varies fastest across 17 columns from -0.8
to 0.8 m, y is the outer loop across 11 rows from -0.5 to 0.5 m. Each value is
`base_z - ray_hit_z - 0.5`, clipped to +/-1, which was verified empirically
against a running training environment before any of it was reimplemented.

---

## Measured results

### Control loop latency

Latency was instrumented per tick: the wait between robot state arriving and
inference starting, the inference itself, and the total from state arrival to
command publication.

The first implementation ran the control loop on a wall timer, which is the
obvious way to hold a fixed 50 Hz. Measuring it showed that the timer was
responsible for almost all of the latency:

| | Timer-driven | Event-driven | Change |
|---|---|---|---|
| Queue delay (mean) | 10,295 us | 2.4 us | 4300x |
| Inference (mean) | 1,225 us | 1,227 us | unchanged |
| End-to-end (mean) | 11,554 us | 1,260 us | 9.2x |
| End-to-end (p99) | 23,518 us | 2,282 us | 10x |
| End-to-end (max) | 184,949 us | 2,720 us | 68x |

The mean queue delay was almost exactly half the 20 ms control period, which is
the signature of a timer firing independently of message arrival: on average,
state sat waiting half a period before anything looked at it. The policy was
acting on 10 ms stale data.

Running inference directly from the state callback removed that wait. Inference
time is unchanged across the two configurations, which is the control confirming
the change touched only scheduling.

The tail matters more than the mean here. The old p99 of 23.5 ms exceeded the
20 ms control budget, so roughly one tick in a hundred was late. The worst
observed case is now 2.7 ms.

The composition also inverted. Before the change, 89% of end-to-end latency was
waiting. After it, 97% is inference, so the remaining bottleneck is GPU
contention with the renderer rather than anything in the control path.

### Inference cost in isolation versus in situ

The same engine benchmarks at roughly 7 us standalone and roughly 1,225 us with
Isaac Sim rendering and stepping physics on the same GPU. Isolated inference
benchmarks are not a useful predictor of real system cost when the accelerator
is shared.

### Terrain traversal limit

The deployed policy was swept across terrain roughness to find where it stops
working:

| Terrain relief | Straight and gentle turns | Commanded turns |
|---|---|---|
| low | traverses | traverses |
| medium | traverses | falls |
| high | falls | falls |

Turning tolerates less roughness than straight-line walking, which is consistent
with turning loading the legs asymmetrically. This is the kind of boundary that
only shows up once a policy is actually deployed rather than evaluated inside
its training environment.

---

## Two bugs worth describing

**Angular velocity units.** Isaac Sim's odometry publishes angular velocity in
degrees per second. The policy was trained on radians. Observation slots 3-5
were therefore scaled up by a factor of 57.

This was invisible during straight-line walking, because 57 times a value near
zero is still near zero. It only appeared under commanded yaw, where the policy
saw itself spinning violently and over-corrected until it fell. A unit mismatch
that hides in one region of the input space and is catastrophic in another is a
good argument for testing the observation pipeline directly rather than
inferring its correctness from whether the robot stays upright.

**Actuator model mismatch.** Anymal-C trains against a learned LSTM model of the
ANYdrive actuator's torque response. A plain PD controller is a different plant,
and the policy's actions produce different torques than it learned to expect.
Running the joints at gains tuned to make the robot stand (stiffness 250) rather
than gains matching the training actuator (stiffness 40, damping 5, with an
80 Nm effort cap) was enough to make an otherwise working policy collapse.

This is a sim-to-sim instance of the gap that makes sim-to-real transfer hard,
and it is visible here because the deployment target is explicit rather than
implied.

---

## Code structure

```
include/quad_controller/
  cuda_raii.hpp            RAII ownership for CUDA and TensorRT resources
  policy_types.hpp         observation layout, joint order, standing pose
  observation_builder.hpp  state -> observation, action -> joint targets
  policy_engine.hpp        IPolicyEngine interface, TensorRT implementation
src/
  observation_builder.cpp  quad_core: pure logic, no CUDA, no ROS
  policy_engine.cpp        quad_trt: TensorRT behind a pimpl
  quad_node.cpp            ROS 2 glue
  waypoint_navigator.cpp   course-following planner
  benchmark.cpp            engine comparison tool
test/
  test_observation_builder.cpp   18 tests
```

**The library is split by GPU dependency, not by convenience.** `quad_core`
contains the observation assembly and action scaling and links against nothing
but the standard library, so its tests run on any machine including CI without a
GPU. Tests that cannot run get ignored, so this split is what makes the test
suite load-bearing rather than decorative.

**Inference sits behind an interface.** `IPolicyEngine` exposes `infer()` and
nothing else; `TensorRtPolicy` implements it with a pimpl so that TensorRT and
CUDA headers never reach consumers. The benchmark tool loads two engines through
that interface and compares them without knowing what either one is.

**Every GPU resource has an owner.** `DeviceBuffer` and `CudaStream` are
move-only with deleted copies, since copying a GPU handle would double-free.
Every TensorRT object is held in a `unique_ptr` with an explicit deleter. The
result is that `quad_node` has no destructor at all: teardown happens in reverse
declaration order with no manual bookkeeping and no leak on an early return.

**Joints are mapped by name.** The node reads the joint name list from the first
`/joint_states` message and builds a lookup table, rather than trusting message
ordering to match the policy's expected DOF order.

---

## Tests

18 GoogleTest cases covering the parts where a mistake is silent:

- observation layout, block by block
- joint positions entering as offsets from the standing pose
- previous-action feedback and its reset behaviour
- action scaling into joint targets
- gravity projection across several orientations, including unit-length invariance
- height scan placement, and that it does not overwrite the proprioceptive block

```bash
colcon test --packages-select quad_controller
colcon test-result --verbose
```

---

## Running it

```bash
# terminal 1 - simulator
source /opt/ros/humble/setup.bash
cd <isaac-sim> && ./python.sh -u <repo>/setup_scene_rough.py

# terminal 2 - controller
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run quad_controller quad_node

# terminal 3 - waypoint course
ros2 run quad_controller waypoint_navigator \
  --ros-args -p waypoints_x:="[3.0, 3.0, -3.0, -3.0]" \
             -p waypoints_y:="[-3.0, 3.0, 3.0, -3.0]"
```

Latency samples are written to `/tmp/quad_latency.csv`.

---

## Environment

Ubuntu 22.04, ROS 2 Humble, CUDA 12.8, TensorRT 11.1, Isaac Sim 5.1,
Isaac Lab 2.3.0, RTX 4070 Laptop (8 GB).

---

## Limitations

- **Sim-to-sim, not sim-to-real.** The deployment target is Isaac Sim, not
  hardware. The transport, timing and observation-assembly problems are real;
  the contact dynamics are not.
- **The actuator model is approximated.** The policy trained against a learned
  LSTM actuator network and is deployed onto PD-controlled joints. The gains are
  matched to NVIDIA's own PD equivalent of the same actuator, but it remains an
  approximation, and the measured traversal limit reflects that.
- **Terrain is procedurally generated**, not the structured curriculum of stairs,
  slopes and discrete obstacles the policy trained on. It is rough ground, but it
  is not the same distribution.
- **The GPU is shared with the renderer**, which dominates the remaining latency.
  A headless configuration would give a cleaner measurement of inference cost in
  the control loop.
- **Single robot, single policy.** No fault handling, no recovery behaviour
  beyond an automatic reset when the base drops below a height threshold.
