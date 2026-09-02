# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.converters import UrdfConverterCfg


_REPO_ROOT = Path(__file__).resolve().parents[4]

ALOHA_TACTILE_URDF_PATH = _REPO_ROOT / "assets" / "urdf" / "aloha_tactile" / "aloha_tactile.urdf"
ALOHA_TACTILE_USD_DIR = _REPO_ROOT / "assets" / "usd" / "generated" / "aloha_tactile"

ALOHA_TACTILE_JOINT_ORDER = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_left_finger",
    "left_right_finger",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_left_finger",
    "right_right_finger",
]

ALOHA_TACTILE_ELASTOMER_LINK_NAMES = [
    "left_arm_elastomer_left",
    "left_arm_elastomer_right",
    "right_arm_elastomer_left",
    "right_arm_elastomer_right",
]

ALOHA_TACTILE_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(ALOHA_TACTILE_URDF_PATH),
        fix_base=False,
        merge_fixed_joints=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=400.0,
                damping=40.0,
            )
        ),
        usd_dir=str(ALOHA_TACTILE_USD_DIR),
        force_usd_conversion=False,
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "left_waist": 0.0,
            "left_shoulder": -0.16,
            "left_elbow": 1.15,
            "left_forearm_roll": 0.0,
            "left_wrist_angle": -0.5,
            "left_wrist_rotate": 0.0,
            "left_left_finger": 0.042,
            "left_right_finger": -0.042,
            "right_waist": 0.0,
            "right_shoulder": -0.16,
            "right_elbow": 1.15,
            "right_forearm_roll": 0.0,
            "right_wrist_angle": -0.5,
            "right_wrist_rotate": 0.0,
            "right_left_finger": 0.042,
            "right_right_finger": -0.042,
        },
    ),
    actuators={
        "all": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=400.0,
            damping=40.0,
        )
    },
    soft_joint_pos_limit_factor=1.0,
)
