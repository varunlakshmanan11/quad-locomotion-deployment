from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": False})

import numpy as np
import omni.graph.core as og
import omni.kit.app
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.prims import Articulation

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
    "isaacsim.ros2.bridge", True)

DEFAULT_POS = np.array([0.0, 0.0, 0.0, 0.0,
                        0.4, -0.4, 0.4, -0.4,
                        -0.8, 0.8, -0.8, 0.8], dtype=np.float32)

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

assets = get_assets_root_path()
add_reference_to_stage(assets + "/Isaac/Robots/ANYbotics/anymal_c/anymal_c.usd",
                       "/World/anymal")

robot = Articulation("/World/anymal/base", name="anymal")
world.scene.add(robot)
world.reset()

print("JOINT ORDER:", robot.dof_names)

# ANYdrive-like position gains, and start in the standing pose
n = robot.num_dof
robot.set_gains(kps=np.full((1, n), 250.0), kds=np.full((1, n), 8.0))
robot.set_joint_positions(DEFAULT_POS.reshape(1, n))
robot.set_joint_position_targets(DEFAULT_POS.reshape(1, n))
robot.set_world_poses(positions=np.array([[0.0, 0.0, 0.65]]))

og.Controller.edit(
    {"graph_path": "/ROSGraph", "evaluator_name": "execution"},
    {
        og.Controller.Keys.CREATE_NODES: [
            ("tick", "omni.graph.action.OnPlaybackTick"),
            ("ctx",  "isaacsim.ros2.bridge.ROS2Context"),
            ("pub",  "isaacsim.ros2.bridge.ROS2PublishJointState"),
            ("sub",  "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
            ("art",  "isaacsim.core.nodes.IsaacArticulationController"),
            ("time", "isaacsim.core.nodes.IsaacReadSimulationTime"),
            ("odom", "isaacsim.core.nodes.IsaacComputeOdometry"),
            ("odompub", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
        ],
        og.Controller.Keys.CONNECT: [
            ("tick.outputs:tick", "pub.inputs:execIn"),
            ("tick.outputs:tick", "sub.inputs:execIn"),
            ("tick.outputs:tick", "art.inputs:execIn"),
            ("ctx.outputs:context", "pub.inputs:context"),
            ("ctx.outputs:context", "sub.inputs:context"),
            ("time.outputs:simulationTime", "pub.inputs:timeStamp"),
            ("sub.outputs:jointNames",      "art.inputs:jointNames"),
            ("sub.outputs:positionCommand", "art.inputs:positionCommand"),
            ("tick.outputs:tick", "odom.inputs:execIn"),
            ("odom.outputs:execOut", "odompub.inputs:execIn"),
            ("ctx.outputs:context", "odompub.inputs:context"),
            ("time.outputs:simulationTime", "odompub.inputs:timeStamp"),
            ("odom.outputs:linearVelocity",  "odompub.inputs:linearVelocity"),
            ("odom.outputs:angularVelocity", "odompub.inputs:angularVelocity"),
            ("odom.outputs:position",        "odompub.inputs:position"),
            ("odom.outputs:orientation",     "odompub.inputs:orientation"),
        ],
        og.Controller.Keys.SET_VALUES: [
            ("pub.inputs:topicName", "joint_states"),
            ("sub.inputs:topicName", "joint_command"),
            ("pub.inputs:targetPrim", "/World/anymal/base"),
            ("art.inputs:targetPrim", "/World/anymal/base"),
            ("odom.inputs:chassisPrim", "/World/anymal/base"),
            ("odompub.inputs:topicName", "odom"),
            ("odompub.inputs:odomFrameId", "odom"),
            ("odompub.inputs:chassisFrameId", "base_link"),
        ],
    },
)


# --- diagnostic dump ---
with open("/tmp/graph_report.txt", "w") as f:
    for nn in ("odom", "odompub"):
        try:
            node = og.Controller.node("/ROSGraph/" + nn)
            f.write("NODE /ROSGraph/%s  type=%s\n" % (nn, node.get_type_name()))
            for a in node.get_attributes():
                f.write("   " + a.get_name() + "\n")
        except Exception as e:
            f.write("MISSING %s : %r\n" % (nn, e))
# --- end diagnostic ---

world.play()
while sim_app.is_running():
    world.step(render=True)
sim_app.close()
