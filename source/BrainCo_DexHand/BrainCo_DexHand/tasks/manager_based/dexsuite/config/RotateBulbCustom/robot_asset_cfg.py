# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from .robot_contract import ASSET_ROOT, INITIAL_JOINT_POS, ROBOT_ORIENTATION, ROBOT_POSITION


# Local copy of the donor's Rizon4 + Revo3 assembly, including collision filters.
ROBOT_USD_PATH = ASSET_ROOT / "robot/usd/rizon4_training.usda"

TIANJI_REVO3_RIGHT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ROBOT_USD_PATH),
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
            enabled_self_collisions=True,
            fix_root_link=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=ROBOT_POSITION,
        rot=ROBOT_ORIENTATION,
        joint_pos=dict(INITIAL_JOINT_POS),
    ),
    actuators={
        "flexiv_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-4]"],
            effort_limit_sim=300.0,
            stiffness=300.0,
            damping=45.0,
            friction=1.0,
        ),
        "flexiv_forearm": ImplicitActuatorCfg(
            joint_names_expr=["joint[5-7]"],
            effort_limit_sim=300.0,
            stiffness=50.0,
            damping=15.0,
        ),
        "revo3_hand": ImplicitActuatorCfg(
            joint_names_expr=["right_.*_joint"],
            effort_limit_sim=0.5,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
