"""Fresh pressure-pad poses at the existing parent sensor's sampling rate.

Only the guided rotation factory opts in. The separate fingertip/table force
sensors, contact reporting, collision geometry and pressure calibration remain
unchanged. ``fresh`` is the full ContactSensor reference implementation;
``pose_only`` reads the same PhysX transforms without an unused force reduction.
"""
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import convert_quat
from BrainCo_DexHand.force_map.warp_sdf_tactile_sensor import WarpSdfTactileSensor


class PressurePoseContactSensor(ContactSensor):
    """Internal pose provider; its force buffers are deliberately never consumed."""

    def _update_buffers_impl(self, env_ids):
        if len(env_ids) == self._num_envs:
            env_ids = slice(None)
        pose = self.body_physx_view.get_transforms().view(-1, self._num_bodies, 7)
        self._data.pos_w[env_ids] = pose[env_ids, :, :3]
        # Do not mutate the PhysX tensor returned by get_transforms in place.
        self._data.quat_w[env_ids] = convert_quat(pose[env_ids, :, 3:], to="wxyz")


class FreshPoseWarpSdfTactileSensor(WarpSdfTactileSensor):
    @staticmethod
    def _create_pose_sensor(pose_cfg):
        # The parent invokes this provider only when pressure is sampled. A
        # zero internal period makes update(0, force_recompute=True) refresh it.
        pose_cfg.update_period = 0.0
        return ContactSensor(pose_cfg)


class PoseOnlyWarpSdfTactileSensor(WarpSdfTactileSensor):
    @staticmethod
    def _create_pose_sensor(pose_cfg):
        pose_cfg.update_period = 0.0
        return PressurePoseContactSensor(pose_cfg)


def install_rotate_bulb_pressure_pose(cfg):
    """Resolve after Hydra; preserve Assembly, free-contact mode and legacy runs."""
    if getattr(cfg, "bulb_assembly_task", False) or cfg.bulb_pure_collision:
        return
    mode = getattr(cfg, "pressure_pose_mode", "legacy")
    providers = {"legacy": WarpSdfTactileSensor, "fresh": FreshPoseWarpSdfTactileSensor,
                 "pose_only": PoseOnlyWarpSdfTactileSensor}
    if mode not in providers:
        raise ValueError("pressure_pose_mode must be legacy, fresh, or pose_only")
    count = 0
    for sensor in vars(cfg.scene).values():
        if isinstance(getattr(sensor, "class_type", None), type) and issubclass(sensor.class_type, WarpSdfTactileSensor):
            sensor.class_type = providers[mode]
            count += 1
    if count != 11:
        raise ValueError(f"Rotate-Bulb pressure pose contract requires 11 pads, found {count}")
    print(f"[INFO][RotateBulbCustom] pressure_pose_mode={mode}, pressure pads={count}", flush=True)
