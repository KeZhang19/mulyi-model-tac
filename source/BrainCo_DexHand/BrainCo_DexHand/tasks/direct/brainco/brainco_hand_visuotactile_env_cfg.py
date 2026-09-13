# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent configuration for the direct Revo3 cube visuotactile task."""

from pathlib import Path
import sys

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from BrainCo_DexHand.assets import BRAINCO_CFG


def _find_project_root(start: Path) -> Path:
    for parent in (start.resolve(), *start.resolve().parents):
        if (parent / "tacmap").is_dir() and (parent / "integrate").is_dir():
            return parent
    raise RuntimeError(f"Could not locate the project root from {start}")


PROJECT_ROOT = _find_project_root(Path(__file__))
TACMAP_ROOT = PROJECT_ROOT / "tacmap"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(TACMAP_ROOT) not in sys.path:
    sys.path.append(str(TACMAP_ROOT))



TACTILE_FINGER_LINKS = (
    ("little", "right_little_DIP_Link", "tactileSensor_map_4F"),
    ("ring", "right_ring_DIP_Link", "tactileSensor_map_4F"),
    ("middle", "right_middle_DIP_Link", "tactileSensor_map_4F"),
    ("index", "right_index_DIP_Link", "tactileSensor_map_4F"),
    ("thumb", "right_thumb_DIP_Link", "tactileSensor_map_TH"),
)


@configclass
class BrainCoVisuotactileHandEnvCfg(DirectRLEnvCfg):
    """Repose-cube parameters plus independent five-finger visual tactile input."""

    decimation = 4
    episode_length_s = 15.0
    action_space = 21
    state_observation_dim = 152
    tactile_finger_count = len(TACTILE_FINGER_LINKS)
    tactile_projection_dim = 64  # resolved from the bundle before Gym space creation
    observation_history_length = 1
    single_frame_observation_dim = state_observation_dim + tactile_finger_count * tactile_projection_dim
    observation_space = single_frame_observation_dim * observation_history_length
    state_space = 0
    asymmetric_obs = False
    obs_type = "full"
    visuotactile_observation_terms = ("state", "aligned_tactile_z")
    tactile_policy_checkpoint = str(PROJECT_ROOT / "runs/revo3_encoder_exports/sim_policy_encoder.pt")
    tactile_encoder_chunk_size = 256
    tactile_encoder_id = ""
    tactile_reconstruction_diagnostics = False
    tactile_marker_layout_path = str(
        PROJECT_ROOT / "assets/revo21_right_touch/marker_positions/vitai_4fingers/marker_positions.npz"
    )
    tactile_calibration_urdf = str(
        PROJECT_ROOT / "assets/revo21_right_touch/urdf/revo21_dv2_urdf_right-touch.SLDASM.urdf"
    )
    tactile_rgb_background_path = str(
        PROJECT_ROOT / "vitai_4Fingers-320*240/marker_annotations/reference_median.png"
    )
    tactile_taxim_chunk_size = 32
    tactile_taxim_with_shadow = False

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_patch_count=2**20,
        ),
    )

    robot_cfg: ArticulationCfg = BRAINCO_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    actuated_joint_names = [
        "right_thumb_CMP_joint",
        "right_thumb_CMR_joint",
        "right_thumb_MCP_joint",
        "right_thumb_PIP_joint",
        "right_thumb_DIP_joint",
        "right_index_MPR_joint",
        "right_index_MCP_joint",
        "right_index_PIP_joint",
        "right_index_DIP_joint",
        "right_middle_MPR_joint",
        "right_middle_MCP_joint",
        "right_middle_PIP_joint",
        "right_middle_DIP_joint",
        "right_ring_MPR_joint",
        "right_ring_MCP_joint",
        "right_ring_PIP_joint",
        "right_ring_DIP_joint",
        "right_little_MPR_joint",
        "right_little_MCP_joint",
        "right_little_PIP_joint",
        "right_little_DIP_joint",
    ]
    fingertip_body_names = [link_name for _, link_name, _ in TACTILE_FINGER_LINKS]

    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
            scale=(1, 1, 1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, -0.11, 0.56),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker_visuotactile",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(1, 1, 1),
            ),
            "target_dot": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        },
    )

    # Tactile rendering is substantially heavier than the state-only task.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=32,
        env_spacing=0.75,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    reset_position_noise = 0.01
    reset_dof_pos_noise = 0.2
    reset_dof_vel_noise = 0.0
    dist_reward_scale = -10.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    action_penalty_scale = -0.0002
    reach_goal_bonus = 250
    fall_penalty = 0
    fall_dist = 0.24
    vel_obs_scale = 0.2
    success_tolerance = 0.2
    max_consecutive_success = 0
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0

    # Revision 1 remains selectable when playing historical absolute-action policies.
    repose_training_revision = 2
    repose_joint_noise_fraction = 0.1
    repose_position_noise = 0.003
    repose_object_rotation_noise_deg = 10.0
    repose_action_speed = 1.5  # joint radians/second, integrated once per control step
    repose_action_filter = 0.5
    repose_goal_min_deg = 20.0
    repose_goal_max_degrees = (45.0, 60.0, 90.0, 120.0, 180.0)
    repose_curriculum_enabled = True
    repose_curriculum_min_episodes = 1024
    repose_curriculum_success_threshold = 0.6
    repose_curriculum_interval = 300
    repose_success_hold_steps = 3
    repose_success_position = 0.08
    repose_success_linear_speed = 0.2
    repose_success_angular_speed = 2.0
    repose_progress_scale = 10.0
    repose_orientation_scale = 0.25
    repose_position_scale = -5.0
    repose_position_deadzone = 0.04
    repose_action_scale = -0.01
    repose_action_delta_scale = -0.02
    repose_success_bonus = 25.0
    repose_fall_penalty = -10.0
