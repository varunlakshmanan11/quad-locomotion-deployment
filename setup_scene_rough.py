from isaacsim import SimulationApp

sim_app = SimulationApp({"headless": False})

import math

import numpy as np
import omni.graph.core as og
import omni.kit.app
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from omni.physx import get_physx_scene_query_interface
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics, Vt

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
    "isaacsim.ros2.bridge", True
)

# ---------------------------------------------------------------------------
# Constants that must match how the policy was trained.
# ---------------------------------------------------------------------------

DEFAULT_POS = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.4, -0.4, 0.4, -0.4, -0.8, 0.8, -0.8, 0.8],
    dtype=np.float32,
)

# GridPatternCfg(resolution=0.1, size=[1.6, 1.0]) expands to 17 x 11 points,
# with x varying fastest. Verified against Isaac Lab's own pattern function.
SCAN_RES = 0.1
SCAN_X = [round(-0.8 + SCAN_RES * i, 3) for i in range(17)]
SCAN_Y = [round(-0.5 + SCAN_RES * i, 3) for i in range(11)]
GRID = [(x, y) for y in SCAN_Y for x in SCAN_X]
assert len(GRID) == 187

HEIGHT_OFFSET = 0.5      # subtracted from every scan value
SCAN_CLIP = 1.0          # values are clipped to +/- this
RAY_START_ABOVE = 2.0    # ray origin height above the robot base
RAY_MAX_DIST = 10.0

TERRAIN_PATH = "/World/ground"
ROBOT_PATH = "/World/anymal"
ARTICULATION_PATH = "/World/anymal/base"


# ---------------------------------------------------------------------------
# Terrain
# ---------------------------------------------------------------------------

def build_heightfield(cells: int, amplitude: float, seed: int) -> np.ndarray:
    """Smooth random heights, flattened near the origin so the robot can spawn."""
    rng = np.random.default_rng(seed)
    coarse = rng.uniform(-amplitude, amplitude, (28, 28))

    scale = max(1, (cells + 1) // 28 + 1)
    field = np.kron(coarse, np.ones((scale, scale)))[: cells + 1, : cells + 1]

    for _ in range(1):
        field = (
            field
            + np.roll(field, 1, 0)
            + np.roll(field, -1, 0)
            + np.roll(field, 1, 1)
            + np.roll(field, -1, 1)
        ) / 5.0

    # Rough all the way to the centre: the robot spawns above the surface and
    # settles onto it, so no level pad is needed.
    return field


def spawn_terrain(stage, extent: float = 20.0, cells: int = 96,
                  amplitude: float = 0.10, seed: int = 7) -> None:
    heights = build_heightfield(cells, amplitude, seed)
    step = extent / cells
    origin = -extent / 2.0

    points = []
    for iy in range(cells + 1):
        for ix in range(cells + 1):
            points.append(
                Gf.Vec3f(origin + ix * step, origin + iy * step, float(heights[iy, ix]))
            )

    indices = []
    for iy in range(cells):
        for ix in range(cells):
            a = iy * (cells + 1) + ix
            b = a + 1
            c = a + (cells + 1)
            d = c + 1
            indices.extend([a, b, d, a, d, c])

    mesh = UsdGeom.Mesh.Define(stage, TERRAIN_PATH)
    mesh.CreatePointsAttr(Vt.Vec3fArray(points))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * (len(indices) // 3)))
    mesh.CreateSubdivisionSchemeAttr().Set("none")

    prim = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("none")

    print(f"terrain: {len(points)} verts, {len(indices) // 3} tris, amplitude {amplitude}")


# ---------------------------------------------------------------------------
# Height scan
# ---------------------------------------------------------------------------

class HeightScanner:
    """Casts the policy's scan pattern downward and returns the observation block.

    The pattern is yaw-aligned: it rotates with the robot's heading but stays
    level under pitch and roll, matching RayCasterCfg(ray_alignment="yaw").
    """

    def __init__(self) -> None:
        self._query = get_physx_scene_query_interface()
        self._last = [0.0] * len(GRID)

    @staticmethod
    def _yaw_from_quat(w: float, x: float, y: float, z: float) -> float:
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def scan(self, base_pos, base_quat) -> list:
        bx, by, bz = float(base_pos[0]), float(base_pos[1]), float(base_pos[2])
        yaw = self._yaw_from_quat(*(float(v) for v in base_quat))
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        origin_z = bz + RAY_START_ABOVE

        values = []
        for ox, oy in GRID:
            wx = bx + ox * cos_y - oy * sin_y
            wy = by + ox * sin_y + oy * cos_y

            hit = self._query.raycast_closest(
                [wx, wy, origin_z], [0.0, 0.0, -1.0], RAY_MAX_DIST
            )
            if hit["hit"]:
                hit_z = hit["position"][2]
                value = bz - hit_z - HEIGHT_OFFSET
            else:
                # No ground found: report the clip floor rather than a spike.
                value = -SCAN_CLIP

            values.append(float(np.clip(value, -SCAN_CLIP, SCAN_CLIP)))

        self._last = values
        return values


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

world = World(stage_units_in_meters=1.0, physics_dt=1.0/200.0, rendering_dt=1.0/60.0)
stage = omni.usd.get_context().get_stage()

dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome.CreateIntensityAttr(1200.0)
distant = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
distant.CreateIntensityAttr(2500.0)

spawn_terrain(stage)

assets = get_assets_root_path()
add_reference_to_stage(
    assets + "/Isaac/Robots/ANYbotics/anymal_c/anymal_c.usd", ROBOT_PATH
)

robot = Articulation(ARTICULATION_PATH, name="anymal")
world.scene.add(robot)
world.reset()

print("JOINT ORDER:", robot.dof_names)

n_dof = robot.num_dof
# Gains matched to ANYDRIVE_3_SIMPLE_ACTUATOR_CFG, which is what the policy
# was trained against. Torque is capped at the same effort limit.
robot.set_gains(kps=np.full((1, n_dof), 120.0), kds=np.full((1, n_dof), 6.0))
robot.set_max_efforts(np.full((1, n_dof), 80.0))
robot.set_joint_positions(DEFAULT_POS.reshape(1, n_dof))
robot.set_joint_position_targets(DEFAULT_POS.reshape(1, n_dof))
robot.set_world_poses(positions=np.array([[0.0, 0.0, 0.7]]))

# ---------------------------------------------------------------------------
# ROS 2 graph
# ---------------------------------------------------------------------------

og.Controller.edit(
    {"graph_path": "/ROSGraph", "evaluator_name": "execution"},
    {
        og.Controller.Keys.CREATE_NODES: [
            ("tick", "omni.graph.action.OnPlaybackTick"),
            ("ctx", "isaacsim.ros2.bridge.ROS2Context"),
            ("pub", "isaacsim.ros2.bridge.ROS2PublishJointState"),
            ("sub", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
            ("art", "isaacsim.core.nodes.IsaacArticulationController"),
            ("time", "isaacsim.core.nodes.IsaacReadSimulationTime"),
            ("odom", "isaacsim.core.nodes.IsaacComputeOdometry"),
            ("odompub", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
            ("scanpub", "isaacsim.ros2.bridge.ROS2Publisher"),
        ],
        og.Controller.Keys.CONNECT: [
            ("tick.outputs:tick", "pub.inputs:execIn"),
            ("tick.outputs:tick", "sub.inputs:execIn"),
            ("tick.outputs:tick", "art.inputs:execIn"),
            ("tick.outputs:tick", "odom.inputs:execIn"),
            ("tick.outputs:tick", "scanpub.inputs:execIn"),
            ("odom.outputs:execOut", "odompub.inputs:execIn"),
            ("ctx.outputs:context", "pub.inputs:context"),
            ("ctx.outputs:context", "sub.inputs:context"),
            ("ctx.outputs:context", "odompub.inputs:context"),
            ("ctx.outputs:context", "scanpub.inputs:context"),
            ("time.outputs:simulationTime", "pub.inputs:timeStamp"),
            ("time.outputs:simulationTime", "odompub.inputs:timeStamp"),
            ("sub.outputs:jointNames", "art.inputs:jointNames"),
            ("sub.outputs:positionCommand", "art.inputs:positionCommand"),
            ("odom.outputs:linearVelocity", "odompub.inputs:linearVelocity"),
            ("odom.outputs:angularVelocity", "odompub.inputs:angularVelocity"),
            ("odom.outputs:position", "odompub.inputs:position"),
            ("odom.outputs:orientation", "odompub.inputs:orientation"),
        ],
        og.Controller.Keys.SET_VALUES: [
            ("pub.inputs:topicName", "joint_states"),
            ("sub.inputs:topicName", "joint_command"),
            ("pub.inputs:targetPrim", ARTICULATION_PATH),
            ("art.inputs:targetPrim", ARTICULATION_PATH),
            ("odom.inputs:chassisPrim", ARTICULATION_PATH),
            ("odompub.inputs:topicName", "odom"),
            ("odompub.inputs:odomFrameId", "odom"),
            ("odompub.inputs:chassisFrameId", "base_link"),
            ("scanpub.inputs:topicName", "height_scan"),
            ("scanpub.inputs:messagePackage", "std_msgs"),
            ("scanpub.inputs:messageSubfolder", "msg"),
            ("scanpub.inputs:messageName", "Float32MultiArray"),
        ],
    },
)

# Dump the generic publisher's attribute names once, so the data input can be
# addressed correctly if the naming differs from expectation.
with open("/tmp/scanpub_attrs.txt", "w") as report:
    for attr in og.Controller.node("/ROSGraph/scanpub").get_attributes():
        report.write(attr.get_name() + "\n")

import os

RESET_FLAG = "/tmp/reset_robot"


def reset_robot():
    """Return the robot to the spawn pose without rebuilding the scene."""
    robot.set_world_poses(
        positions=np.array([[0.0, 0.0, 0.7]]),
        orientations=np.array([[1.0, 0.0, 0.0, 0.0]]),
    )
    try:
        robot.set_velocities(np.zeros((1, 6), dtype=np.float32))
    except Exception:
        pass
    robot.set_joint_positions(DEFAULT_POS.reshape(1, n_dof))
    robot.set_joint_velocities(np.zeros((1, n_dof), dtype=np.float32))
    robot.set_joint_position_targets(DEFAULT_POS.reshape(1, n_dof))
    print("robot reset")


scanner = HeightScanner()
scan_attr = og.Controller.attribute("/ROSGraph/scanpub.inputs:data")
offset_attr = og.Controller.attribute("/ROSGraph/scanpub.inputs:layout:data_offset")
og.Controller.set(offset_attr, 0)

world.play()

frame = 0
fallen_for = 0
while sim_app.is_running():
    world.step(render=True)

    positions, orientations = robot.get_world_poses()

    if os.path.exists(RESET_FLAG):
        os.remove(RESET_FLAG)
        reset_robot()
        fallen_for = 0

    # base sitting low for a sustained period means it has fallen over
    if float(positions[0][2]) < 0.30:
        fallen_for += 1
        if fallen_for > 90:
            reset_robot()
            fallen_for = 0
    else:
        fallen_for = 0

    values = scanner.scan(positions[0], orientations[0])

    try:
        og.Controller.set(scan_attr, np.asarray(values, dtype=np.float32))
    except Exception as exc:  # attribute name mismatch shows up immediately
        if frame == 0:
            print("SCAN PUBLISH FAILED:", exc)

    if frame % 200 == 0:
        arr = np.asarray(values)
        print(
            f"scan  min {arr.min():+.3f}  max {arr.max():+.3f}  mean {arr.mean():+.3f}"
        )
    frame += 1

sim_app.close()
