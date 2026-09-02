# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


TIANJI_REVO3_RIGHT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(
            Path(__file__).resolve().parents[4]
            / "assets"
            / "urdf"
            / "Tianji_Revo3"
            / "urdf"
            / "Tianji_Revo3_Right_visual.usd"
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            fix_root_link=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            # Tianji arm joints.
            "Joint1_R": 0.0,
            "Joint2_R": -1.57079632679,
            "Joint3_R": 0.0,
            "Joint4_R": 1.57079632679,
            "Joint5_R": 0.0,
            "Joint6_R": -0.7,
            "Joint7_R": 0.0,
            # Revo3 right hand joints from Tianji_Revo3_Right.urdf.
            "right_thumbcmp_roll_joint": 0.0,
            "right_thumbcmr_roll_joint": 0.0,
            "right_thumbmcp_roll_joint": 0.0,
            "right_thumbpip_roll_joint": 0.0,
            "right_thumbdip_roll_joint": 0.0,
            "right_indexmcp_yaw_joint": 0.0,
            "right_indexmcp_roll_joint": 0.0,
            "right_indexpip_roll_joint": 0.0,
            "right_indexdip_roll_joint": 0.0,
            "right_midmcp_yaw_joint": 0.0,
            "right_midmcp_roll_joint": 0.0,
            "right_midpip_roll_joint": 0.0,
            "right_middip_roll_joint": 0.0,
            "right_ringmcp_yaw_joint": 0.0,
            "right_ringmcp_roll_joint": 0.0,
            "right_ringpip_roll_joint": 0.0,
            "right_ringdip_roll_joint": 0.0,
            "right_pinkymcp_yaw_joint": 0.0,
            "right_pinkymcp_roll_joint": 0.0,
            "right_pinkypip_roll_joint": 0.0,
            "right_pinkydip_roll_joint": 0.0,
        },
    ),
    actuators={
        "tianji_arm_base": ImplicitActuatorCfg(
            joint_names_expr=["Joint[1-2]_R"],
            stiffness=300.0,
            damping=45.0,
            friction=1.0,
        ),
        "tianji_arm_mid": ImplicitActuatorCfg(
            joint_names_expr=["Joint[3-4]_R"],
            stiffness=220.0,
            damping=30.0,
            friction=1.0,
        ),
        "tianji_arm_wrist": ImplicitActuatorCfg(
            joint_names_expr=["Joint[5-7]_R"],
            stiffness=50.0,
            damping=15.0,
            friction=0.5,
        ),
        "brainco_hand": ImplicitActuatorCfg(
            joint_names_expr=["right_.*_joint"],
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
