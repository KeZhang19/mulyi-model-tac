"""Shallow insertion geometry for the free bulb and its matching socket."""
import math

import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul


def match_socket_scale_to_bulb(env, env_ids):
    """After bulb scale randomization, before physics starts, match each socket.

    Preserve the full inherited bulb size distribution and original clearances.
    Cache the authored scale for target/depth calculations; never scale live bodies.
    """
    from pxr import Gf, UsdGeom

    if env.sim.is_playing():
        raise RuntimeError("Socket scale matching must run in prestartup.")
    if env_ids is not None:
        raise ValueError("Socket scale matching is a scene-wide prestartup event.")
    scales = []
    for path in env.scene.env_prim_paths:
        bulb = env.scene.stage.GetPrimAtPath(f"{path}/Object")
        socket = env.scene.stage.GetPrimAtPath(f"{path}/Socket")
        value = bulb.GetAttribute("xformOp:scale").Get()
        if value is None or min(value) <= 0 or max(value) - min(value) > 1e-6:
            raise ValueError(f"Expected a positive uniform bulb scale at {path}: {value}")
        attr = socket.GetAttribute("xformOp:scale")
        if not attr:
            attr = UsdGeom.Xformable(socket).AddScaleOp(UsdGeom.XformOp.PrecisionDouble).GetAttr()
        attr.Set(Gf.Vec3d(*value))
        scales.append(float(value[0]))
    env._bulb_socket_scales = torch.tensor(scales, device=env.device)


def bulb_tip_position_w(obj, scales, cfg):
    """World position of the authored thread-front axial center, including scale."""
    offset = torch.tensor(cfg.bulb_tip_offset_m, device=scales.device) * scales[:, None]
    return obj.data.root_pos_w + quat_apply(obj.data.root_quat_w, offset)


def insertion_tip_target(socket, scales, cfg, depth_m=None, phase=None):
    """Direct world tip target and bulb orientation; no center compensation."""
    depth_m = cfg.insertion_depth_m if depth_m is None else depth_m
    offset = torch.zeros_like(socket.data.root_pos_w)
    offset[:, 0] = cfg.socket_mouth_x_m * scales - depth_m
    tip_w = socket.data.root_pos_w + quat_apply(socket.data.root_quat_w, offset)
    extension = cfg.socket_mouth_x_m - cfg.source_thread_tip_x_m - depth_m / scales
    spin = (cfg.reference_spin_rad + (extension - cfg.reference_extension_m) * (2 * math.pi / cfg.thread_pitch_m)
            if phase is None else phase)
    spin_quat = torch.zeros_like(socket.data.root_quat_w)
    spin_quat[:, 0], spin_quat[:, 1] = torch.cos(spin / 2), torch.sin(spin / 2)
    quat_w = quat_mul(socket.data.root_quat_w, spin_quat)
    return tip_w, quat_w


def insertion_target(socket, scales, cfg, depth_m=None, phase=None):
    """Center pose for geometry probes that place a manufactured bulb state.

    Production commands use insertion_tip_target directly. This helper does not
    write poses or add physical guidance.
    """
    tip_w, quat_w = insertion_tip_target(socket, scales, cfg, depth_m, phase)
    tip_offset = torch.tensor(cfg.bulb_tip_offset_m, device=scales.device) * scales[:, None]
    return tip_w - quat_apply(quat_w, tip_offset), quat_w


def insertion_errors(obj, socket, scales, cfg):
    """Measure the thread axis at its tip and at the mouth, in socket coordinates.

    Testing both points rejects a tilted tip that entered off-axis or a bulb
    beside/through the socket wall. Spin around the thread axis is unrestricted.
    """
    tip_w = bulb_tip_position_w(obj, scales, cfg)
    tip_s = quat_apply_inverse(socket.data.root_quat_w, tip_w - socket.data.root_pos_w)
    outward = torch.zeros_like(tip_s)
    outward[:, 0] = 1
    axis_s = quat_apply_inverse(socket.data.root_quat_w, quat_apply(obj.data.root_quat_w, outward))
    depth = cfg.socket_mouth_x_m * scales - tip_s[:, 0]
    axis_x = axis_s[:, 0].clamp(min=1e-6)
    at_mouth = tip_s[:, 1:] + axis_s[:, 1:] * (depth / axis_x)[:, None]
    radial = torch.maximum(torch.linalg.vector_norm(tip_s[:, 1:], dim=-1),
                           torch.linalg.vector_norm(at_mouth, dim=-1))
    tilt = torch.atan2(torch.linalg.vector_norm(axis_s[:, 1:], dim=-1), axis_s[:, 0])
    speed = torch.linalg.vector_norm(obj.data.root_lin_vel_w - socket.data.root_lin_vel_w, dim=-1)
    angular_speed = torch.linalg.vector_norm(obj.data.root_ang_vel_w - socket.data.root_ang_vel_w, dim=-1)
    return depth, radial, tilt, speed, angular_speed
