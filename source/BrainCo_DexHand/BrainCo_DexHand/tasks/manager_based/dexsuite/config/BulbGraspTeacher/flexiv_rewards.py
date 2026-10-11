# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
# Isolated V2-test teacher port. Source and SHA256 are recorded in port_manifest.json.
import torch

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.managers import SceneEntityCfg

from isaaclab.sensors import CameraCfg, ContactSensor, ContactSensorCfg, TiledCameraCfg

def flexiv_hand_contacts(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:

    # Get contact sensors for Flexiv hand fingertips (names must match `finger_tip_body_list`)
    thumb_contact_sensor: ContactSensor = env.scene.sensors['right_thumb_DIP_Link_object_s']
    index_contact_sensor: ContactSensor = env.scene.sensors['right_index_DIP_Link_object_s']
    middle_contact_sensor: ContactSensor = env.scene.sensors['right_middle_DIP_Link_object_s']
    ring_contact_sensor: ContactSensor = env.scene.sensors['right_ring_DIP_Link_object_s']
    little_contact_sensor: ContactSensor = env.scene.sensors['right_little_DIP_Link_object_s']
 
    
    # Get contact forces
    thumb_contact = thumb_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    index_contact = index_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    middle_contact = middle_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    ring_contact = ring_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    little_contact = little_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)

    
    # Calculate contact force magnitudes
    thumb_contact_mag = torch.norm(thumb_contact, dim=-1)
    index_contact_mag = torch.norm(index_contact, dim=-1)
    middle_contact_mag = torch.norm(middle_contact, dim=-1)
    ring_contact_mag = torch.norm(ring_contact, dim=-1)
    little_contact_mag = torch.norm(little_contact, dim=-1)

    good_contact_cond = (thumb_contact_mag > threshold)& (
        (index_contact_mag > threshold) | (middle_contact_mag > threshold) | (ring_contact_mag > threshold) | (little_contact_mag > threshold) 
    )
    
    return good_contact_cond

def flexiv_position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg, align_asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of commanded position using tanh kernel, gated by hand contact presence."""
    from isaaclab.assets import RigidObject
    from isaaclab.utils.math import combine_frame_transforms
    
    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    distance = torch.norm(object.data.root_pos_w - des_pos_w, dim=1)
    return (1 - torch.tanh(distance / std)) * flexiv_hand_contacts(env, 1.0).float()
