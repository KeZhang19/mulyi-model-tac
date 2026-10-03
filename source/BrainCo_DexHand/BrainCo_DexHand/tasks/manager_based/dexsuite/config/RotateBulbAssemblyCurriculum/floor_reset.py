"""Horizontal, collision-height-correct tabletop resets; no settling substeps."""
import math

import torch

from ..RotateBulbCustom import assembly


def reset_floor_bulb(env, ids, template, **params):
    # Reuse support placement and all authoritative failure/phase timer resets.
    # Sample yaw exactly once below; an upright bulb's yaw is not its tilt.
    yaw_range = params.pop("yaw_range", (0., 0.))
    assembly.reset_bulb_assembly_on_table(env, ids, **params, yaw_range=(0., 0.))
    bulb = env.scene["object"]
    pose = bulb.data.root_state_w[ids, :7].clone()
    heading = torch.empty(len(ids), device=env.device).uniform_(*yaw_range) + math.pi / 2
    pose[:, 3:] = 0
    pose[:, 3] = torch.cos(heading / 2)
    pose[:, 6] = torch.sin(heading / 2)
    # Pure world yaw leaves the horizontal asset's local Z equal to world Z.
    # Use enabled collision geometry, not the upright thread's X extent.
    bottom = min(v[2] for v in template["bulb_collision_vertices"])
    pose[:, 2] = (env.scene["table"].data.root_pos_w[ids, 2]
                  + env.cfg.scene.table.spawn.size[2] / 2 - bottom + assembly.DESKTOP_CLEARANCE_M)
    bulb.write_root_pose_to_sim(pose, env_ids=ids)
    env._bulb_assembly_state.reset(env, ids)
