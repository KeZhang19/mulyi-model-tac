"""Per-environment curriculum state, physical prefix snapshots and episode logs."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import torch

from ..RotateBulbCustom import assembly
from ..RotateBulbCustom.assembly_env_cfg import assembly_hand_shell_distance
from ..RotateBulbCustom.assembly_lift import bulb_collision_vertices, bulb_table_clearance
from ..RotateBulbCustom.env_cfg import reset_robot_to_grasp, rotate_bulb_finger_forces
from .core import HoldCounter, LIFT_LEVELS, StageScheduler, StateBank, settings_dict, soft_grasp_quality
from .floor_reset import reset_floor_bulb
from .distributed import restore_worker_state

ROOT = Path(__file__).resolve().parents[8]
PREGRASP_PATH = ROOT / "assets/rotate_bulb_assembly_curriculum/floor_pregrasp_v2.json"
EVENTS = ("grasp", "stable_grasp", "lift", "hover", "alignment", "engagement", "insertion", "final")


def load_pregrasp():
    data = json.loads(PREGRASP_PATH.read_text())
    if data.get("schema") != 1 or len(data.get("approach_joint_positions", [])) != 3:
        raise ValueError("Invalid assembly curriculum pregrasp template")
    for path, expected in data["asset_hashes"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Pregrasp template asset changed: {path}")
    return data


def curriculum_contract(env):
    """Stage-independent physical/interface signature; fixture use is stored separately."""
    robot = env.scene["robot"]
    parent = Path(__file__).resolve().parents[1] / "RotateBulbCustom"
    source = {name: hashlib.sha256((parent / name).read_bytes()).hexdigest() for name in
              ("assembly.py", "assembly_env_cfg.py", "assembly_guidance.py", "assembly_lift.py",
               "actions.py", "robot_asset_cfg.py", "env_cfg.py", "self_collision.py", "tactile.py")}
    return {
        "schema": 2, "curriculum_version": 2, "seed": int(env.cfg.seed), "joint_names": list(robot.joint_names),
        "action": env.action_manager.get_term("action").action_contract(),
        "observations": env.tactile_policy_contract,
        "sources": source, "pregrasp": hashlib.sha256(PREGRASP_PATH.read_bytes()).hexdigest(),
        "curriculum_sources": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                               for name in ("core.py", "runtime.py", "env_cfg.py", "environment.py", "insertion.py",
                                            "floor_reset.py", "rewards.py", "distributed.py")},
        "hover_m": env.cfg.assembly_guidance.hover_clearance_m,
        "sim_dt": env.cfg.sim.dt, "decimation": env.cfg.decimation, "gravity": list(env.cfg.sim.gravity),
        "solver": {k: getattr(env.cfg.sim.physx, k) for k in
                   ("solver_type", "min_position_iteration_count", "max_position_iteration_count",
                    "min_velocity_iteration_count", "max_velocity_iteration_count")},
        "pure_collision": bool(env.cfg.bulb_pure_collision),
    }


class CurriculumRuntime:
    def __init__(self, env):
        self.env, self.cfg = env, env.cfg.bulb_curriculum
        self.cfg.validate()
        self.template = load_pregrasp()
        self.scheduler = StageScheduler(self.cfg)
        self.contract = curriculum_contract(env)
        self.bank = StateBank(self.cfg.bank_capacity, self.contract)
        self.rng = torch.Generator().manual_seed(int(env.cfg.seed))
        n, device = env.num_envs, env.device
        self.source = torch.full((n,), self.cfg.stage, dtype=torch.long, device=device)
        self.level = torch.zeros(n, dtype=torch.long, device=device)
        self.serial = torch.zeros(n, dtype=torch.long, device=device)
        self.active = torch.zeros(n, dtype=torch.bool, device=device)
        self.sample_id = torch.full((n,), -1, dtype=torch.long, device=device)
        self.validated = torch.zeros(n, dtype=torch.bool, device=device)
        self.reset_failed = torch.zeros(n, dtype=torch.bool, device=device)
        self.initial_clearance = torch.zeros(n, device=device)
        self.start_pose = torch.zeros(n, 7, device=device)
        self.grasp_anchor = torch.zeros(n, 7, device=device)
        self.reset_relative_pose = torch.zeros(n, 7, device=device)
        self.palm_id = env.scene["robot"].body_names.index("palm")
        self.best = torch.zeros(n, 3, device=device)
        self.initialized = torch.zeros(n, dtype=torch.bool, device=device)
        self.hold_credit = torch.zeros(n, device=device)
        self.reward_value = torch.zeros(n, device=device)
        self.stage_success = torch.zeros(n, dtype=torch.bool, device=device)
        self.ever = torch.zeros(n, len(EVENTS), dtype=torch.bool, device=device)
        self.first = torch.full((n, len(EVENTS)), -1., device=device)
        self.holds = [HoldCounter(n, device) for _ in range(3)]
        self.last_step = -1
        self.pending = {}
        self.transitions = 0
        self.episodes = 0
        self.finished_records = []
        self.vertices = bulb_collision_vertices(env.scene.stage,
            env.scene["object"].root_physx_view.prim_paths[0], env.device)
        self.joint1_id = env.scene["robot"].joint_names.index("joint1")
        self.joint1_previous = torch.zeros(n, device=device)
        self.joint1_travel = torch.zeros(n, device=device)
        self.joint1_max_excursion = torch.zeros(n, device=device)
        self.log_path = Path(env.cfg.log_dir) / "curriculum_episodes.jsonl" if env.cfg.log_dir else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.cfg.checkpoint:
            checkpoint = torch.load(self.cfg.checkpoint, map_location="cpu", weights_only=False)
            saved = (checkpoint.get("infos") or {}).get("bulb_curriculum")
            if saved is None:
                raise ValueError("Curriculum checkpoint is missing its scheduler and prefix bank")
            self.load_state_dict(saved, transfer=self.cfg.transfer)
        elif self.cfg.stage > 0:
            raise ValueError("Stage1–full require a curriculum predecessor checkpoint; use original task for a fresh full run")
        if self.cfg.stage in (2, 3) and len(self.bank.samples[self.cfg.stage]) < self.cfg.min_bank_size:
            raise ValueError(f"Stage{self.cfg.stage} needs {self.cfg.min_bank_size} successful predecessor episodes")

    def state_dict(self):
        return {"schema": 1, "settings": settings_dict(self.cfg), "scheduler": self.scheduler.state_dict(),
                "bank": self.bank.state_dict(), "rng": self.rng.get_state(),
                "transitions": self.transitions, "episodes": self.episodes}

    def load_state_dict(self, saved, *, transfer=False):
        saved = restore_worker_state(saved, seed=int(self.env.cfg.seed),
                                     world_size=int(os.environ.get("WORLD_SIZE", "1")),
                                     rank=int(os.environ.get("RANK", "0")), evaluation=self.cfg.evaluation)
        if saved.get("schema") != 1:
            raise ValueError("Unsupported curriculum checkpoint schema")
        previous = saved["settings"]
        if transfer:
            valid = ((self.cfg.stage == previous["stage"] == 0 and previous["fixed_bulb"] and not self.cfg.fixed_bulb)
                     or (self.cfg.stage == previous["stage"] + 1 and not previous["fixed_bulb"]))
            if not valid or not saved["scheduler"]["ready"]:
                raise ValueError("Transfer requires a successful immediate predecessor (including Stage0-free)")
        else:
            ignored = {"checkpoint", "transfer", "evaluation"}
            if any(previous.get(k) != v for k, v in settings_dict(self.cfg).items() if k not in ignored):
                raise ValueError("Resume curriculum settings differ; use explicit adjacent-stage transfer")
            self.scheduler.load_state_dict(saved["scheduler"])
            if not self.cfg.evaluation and self.scheduler.ready:
                # A stopped nomination can resume after a failed independent check.
                # Require fresh complete windows before nominating it again.
                self.scheduler.ready = False
                self.scheduler.count = self.scheduler.successes = self.scheduler.passed_windows = 0
        # Evaluation seeds change reset sampling only; training banks remain tied to their parent seed.
        bank_state = saved["bank"]
        if self.cfg.evaluation:
            expected = dict(bank_state["contract"])
            expected["seed"] = self.contract["seed"]
            if expected != self.contract:
                raise ValueError("Evaluation curriculum bank contract mismatch")
            bank_state = {**bank_state, "contract": self.contract}
        self.bank.load_state_dict(bank_state)
        if not self.cfg.evaluation:
            self.rng.set_state(saved["rng"])
        self.transitions = saved["transitions"]
        self.episodes = saved["episodes"]

    def reset_physics(self, ids, table_params):
        """Runs inside the reset event, before the manager histories are cleared."""
        env, cfg = self.env, self.cfg
        ids = torch.as_tensor(ids, device=env.device, dtype=torch.long)
        self.pending.clear()
        sources = torch.full((len(ids),), cfg.stage, dtype=torch.long)
        if not cfg.evaluation and cfg.stage:
            replay = torch.rand(len(ids), generator=self.rng) < cfg.previous_fraction
            previous = torch.full_like(sources, cfg.stage - 1) if cfg.stage < 4 else torch.randint(
                1, 4, (len(ids),), generator=self.rng)
            sources = torch.where(replay, previous, sources)
        for source in (2, 3):
            if (sources == source).any() and not self.bank.samples[source]:
                raise ValueError(f"Requested unavailable Stage{source} bank")
        self.source[ids] = sources.to(env.device)
        self.level[ids] = self.scheduler.level
        self.sample_id[ids] = -1
        # Batch each tabletop source/level. Randomization is sampled once per reset.
        robot = env.scene["robot"]
        reset_robot_to_grasp(env, ids)
        for source in (0, 1, 4):
            selected = ids[self.source[ids] == source]
            if not selected.numel():
                continue
            path_index, xy, yaw = (2, .05, torch.pi / 2) if source == 4 else (
                LIFT_LEVELS[self.scheduler.level] if source == 1 and cfg.stage == 1 else
                ((2, .05, torch.pi / 2) if source == 1 else (0, 0., 0.)))
            params = {**table_params, "x_range": (-xy, xy), "y_range": (-xy, xy), "yaw_range": (-yaw, yaw)}
            reset_floor_bulb(env, selected, self.template, **params)
            if path_index < 2:
                by_name = dict(zip(self.template["joint_names"], self.template["approach_joint_positions"][path_index]))
                q = torch.tensor([by_name[name] for name in robot.joint_names], device=env.device).expand(len(selected), -1)
                robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=selected)
                robot.set_joint_position_target(q, env_ids=selected)
        for i, source in zip(ids.cpu().tolist(), sources.tolist()):
            if source not in (2, 3):
                continue
            index, item = self.bank.sample(source, self.rng)
            self.sample_id[i] = index
            self.pending[i] = item
            self.restore_physics(i, item["state"])
            selected = torch.tensor([i], device=env.device)
            env._bulb_assembly_state.reset(env, selected)
            # These two facts have a real physical prefix, not a sampled pose label.
            state = env._bulb_assembly_state
            state.grasped[selected] = True
            state.lifted[selected] = True
            state.phase[selected] = assembly.LIFTED
            env._assembly_hold_time[selected] = 0
            env._assembly_no_contact_time[selected] = 0

    def restore_physics(self, i, states):
        """Restore a joint snapshot; this alone does not certify a valid contact reset."""
        env = self.env
        selected = torch.tensor([i], device=env.device)
        for name in ("robot", "object", "support", "table"):
            root = states[name].to(env.device).unsqueeze(0).clone()
            root[:, :3] += env.scene.env_origins[selected]
            env.scene[name].write_root_pose_to_sim(root[:, :7], env_ids=selected)
            env.scene[name].write_root_velocity_to_sim(root[:, 7:], env_ids=selected)
        env.scene["robot"].write_joint_state_to_sim(
            states["q"].to(env.device)[None], states["qd"].to(env.device)[None], env_ids=selected)

    def after_reset(self, ids):
        env = self.env
        self.active[ids] = True
        self.serial[ids] += 1
        self.validated[ids] = self.source[ids] < 2
        self.validated[ids[self.source[ids] == 4]] = True
        self.reset_failed[ids] = False
        self.stage_success[ids] = False
        self.ever[ids] = False
        self.first[ids] = -1
        self.initialized[ids] = False
        self.best[ids] = 0
        self.hold_credit[ids] = 0
        self.initial_clearance[ids] = bulb_table_clearance(env, self.vertices, ids)
        self.start_pose[ids] = env.scene["object"].data.root_state_w[ids, :7]
        self.grasp_anchor[ids] = 0
        self.joint1_previous[ids] = env.scene["robot"].data.joint_pos[ids, self.joint1_id]
        self.joint1_travel[ids] = 0
        self.joint1_max_excursion[ids] = 0
        for hold in self.holds:
            hold.reset(ids)
        action, robot = env.action_manager.get_term("action"), env.scene["robot"]
        for i in self.pending:
            selected = torch.tensor([i], device=env.device)
            s = self.pending[i]["state"]
            for key, setter in (("target", robot.set_joint_position_target),
                                ("velocity_target", robot.set_joint_velocity_target),
                                ("effort_target", robot.set_joint_effort_target)):
                setter(s[key].to(env.device)[None], env_ids=selected)
            action._position_targets[i] = s["action_target"].to(env.device)
            action._raw_actions[i] = s["raw_action"].to(env.device)
            action._processed_actions[i] = s["action_delta"].to(env.device)
            env.action_manager.action[i] = s["last_action"].to(env.device)
            env.action_manager.prev_action[i] = s["previous_action"].to(env.device)
            self.initial_clearance[i] = s["desktop_clearance"].to(env.device)
            self.reset_relative_pose[i] = s["hand_bulb_relative_pose"].to(env.device)
            # Already-earned lift credit must not be paid by full-stage bank resets.
            for name in ("lift_best_progress", "lift_stable_once"):
                if name in env.reward_manager.active_terms:
                    term = env.reward_manager.get_term_cfg(name).func
                    term.initial_clearance[i] = self.initial_clearance[i]
                    term.best_quality[i] = 1.
                    term.paid[i] = True
        self.last_step = -1

    def relative_pose(self):
        robot, bulb = self.env.scene["robot"].data, self.env.scene["object"].data
        pos, quat = assembly.subtract_frame_transforms(robot.body_pos_w[:, self.palm_id],
            robot.body_quat_w[:, self.palm_id], bulb.root_pos_w, bulb.root_quat_w)
        return torch.cat((pos, quat), dim=-1)

    def update(self):
        env, cfg = self.env, self.cfg
        if self.last_step == env.common_step_counter:
            return
        self.last_step = env.common_step_counter
        self.transitions += env.num_envs
        q1 = env.scene["robot"].data.joint_pos[:, self.joint1_id]
        self.joint1_travel += (q1 - self.joint1_previous).abs() * self.active
        self.joint1_previous.copy_(q1)
        reference = env.scene["robot"].data.default_joint_pos[:, self.joint1_id]
        self.joint1_max_excursion = torch.maximum(self.joint1_max_excursion, (q1 - reference).abs())
        state = env._bulb_assembly_state.update(env)
        forces = rotate_bulb_finger_forces(env)
        thumb, other = forces[:, 0], forces[:, 1:].amax(-1)
        grasp = (thumb > 1.) & (other > 1.)
        relative = self.relative_pose()
        new_contact = grasp & (self.holds[0].steps == 0)
        self.grasp_anchor[new_contact] = relative[new_contact]
        slip = (relative[:, :3] - self.grasp_anchor[:, :3]).norm(dim=-1)
        angle = 2 * torch.acos((relative[:, 3:] * self.grasp_anchor[:, 3:]).sum(-1).abs().clamp(0, 1))
        stable_relative = (slip <= .005) & (angle <= torch.pi / 36)
        bulb = env.scene["object"].data
        c = bulb_table_clearance(env, self.vertices)
        table = env.scene.sensors["bulb_table_contact"].data.force_matrix_w
        table_force = (table.norm(dim=-1).flatten(1).sum(-1) if table is not None else torch.full_like(c, float("inf")))
        slow = (bulb.root_lin_vel_w.norm(dim=-1) < .05) & (bulb.root_ang_vel_w.norm(dim=-1) < .5)
        finite = torch.isfinite(c) & torch.isfinite(bulb.root_state_w).all(-1)
        held_lift = grasp & finite & (c >= .03) & (c - self.initial_clearance >= .03) & (table_force < .1) & slow
        hover = (grasp & finite & slow & (table_force < .1) & state.lifted
                 & ((state.depth + env.cfg.assembly_guidance.hover_clearance_m).abs() < .01)
                 & (state.radial_error < .006) & (state.axis_error < .2))
        bank_source = (self.source == 2) | (self.source == 3)
        age = env.episode_length_buf
        deadline = bank_source & ~self.validated & (age >= cfg.validation_steps)
        # The CURRENT policy controls every recovery step. No hidden hold controller or physics steps.
        restored = grasp & finite & slow & (c >= .03) & (table_force < .1)
        restored &= (relative[:, :3] - self.reset_relative_pose[:, :3]).norm(dim=-1) <= .005
        restored &= 2 * torch.acos((relative[:, 3:] * self.reset_relative_pose[:, 3:]).sum(-1).abs().clamp(0, 1)) <= torch.pi / 36
        restored &= (self.source != 3) | hover
        self.reset_failed |= deadline & ~restored
        self.validated |= deadline & restored
        valid = finite & self.active & self.validated & ~self.reset_failed
        for counter, condition in zip(self.holds, (grasp & slow & stable_relative, held_lift, hover)):
            counter.update(condition & valid)
        stable_grasp = self.holds[0].reached(cfg.grasp_hold_s, env.step_dt, strict=True)
        lifted = self.holds[1].reached(cfg.lift_hold_s, env.step_dt)
        hovered = self.holds[2].reached(cfg.hover_hold_s, env.step_dt)
        events = torch.stack((state.grasped, stable_grasp, lifted, hovered, state.aligned,
                              state.engaged, state.installed, state.success), dim=-1) & valid[:, None]
        # Inherited prefix events are not counted as achievements of this episode.
        events[bank_source, :3] = False
        events[self.source == 3, 3] = False
        first = events & ~self.ever
        self.first.copy_(torch.where(first, (age * env.step_dt)[:, None], self.first))
        self.ever |= events
        achieved = torch.stack((stable_grasp, lifted, hovered, state.success, state.success), dim=-1)
        self.stage_success |= achieved.gather(1, self.source[:, None]).squeeze(1) & valid
        # A Stage3 reset is not allowed to receive hover credit unless physically reconstructed.
        if cfg.stage == 3:
            for name in ("position_tracking", "insertion_tracking"):
                if name in env.reward_manager.active_terms:
                    term = env.reward_manager.get_term_cfg(name).func
                    term.hover_reached |= hover & valid
        if cfg.version == 2:
            # v2's reward graph owns all shaping. The legacy curriculum
            # potential was otherwise still recomputed every control step.
            self.reward_value.zero_()
            return
        quality = soft_grasp_quality(forces)
        distance = torch.nan_to_num(assembly_hand_shell_distance(env), nan=1., posinf=1.)
        approach = 1 - torch.tanh(distance / .08)
        height = ((c - self.initial_clearance) / .03).clamp(0, 1)
        position = 1 - torch.tanh(torch.sqrt((state.depth + env.cfg.assembly_guidance.hover_clearance_m).square()
                                           + state.radial_error.square()) / .2)
        axis = torch.exp(-(state.axis_error / .5).square())
        carry = quality * (c / .03).clamp(0, 1) * axis * position
        lift = quality * height
        potentials = torch.stack((approach, quality, lift if cfg.stage <= 1 else carry), -1)
        potentials = torch.nan_to_num(potentials, nan=0., posinf=0., neginf=0.)
        start = valid & ~self.initialized
        self.best[start] = potentials[start]
        self.initialized |= valid
        progress = (potentials - self.best).clamp_min(0) * valid[:, None]
        self.best.copy_(torch.maximum(self.best, torch.where(valid[:, None], potentials, self.best)))
        budgets = potentials.new_tensor([2., 3., 0. if cfg.stage == 0 else 8.])
        remaining_hold = (1.0 - self.hold_credit).clamp_min(0)
        hold = torch.minimum(remaining_hold, quality * env.step_dt) * valid
        self.hold_credit += hold
        self.reward_value.copy_(((progress * budgets).sum(-1) + 2. * hold) / env.step_dt)
        if cfg.stage >= 3:
            self.reward_value.zero_()  # Stage3 uses original insertion; full uses every original term.

    def snapshot(self, i):
        env, robot = self.env, self.env.scene["robot"]
        action = env.action_manager.get_term("action")
        result = {}
        for name in ("robot", "object", "support", "table"):
            result[name] = env.scene[name].data.root_state_w[i].clone()
            result[name][:3] -= env.scene.env_origins[i]
        result.update(q=robot.data.joint_pos[i], qd=robot.data.joint_vel[i],
                      target=robot.data.joint_pos_target[i], velocity_target=robot.data.joint_vel_target[i],
                      effort_target=robot.data.joint_effort_target[i], action_target=action.position_targets[i],
                      raw_action=action.raw_actions[i], action_delta=action.action_deltas[i],
                      last_action=env.action_manager.action[i], previous_action=env.action_manager.prev_action[i],
                      desktop_clearance=self.initial_clearance[i], hand_bulb_relative_pose=self.relative_pose()[i])
        return result

    def finish(self, ids):
        env = self.env
        records = []
        ids = ids[self.active[ids] & (env.episode_length_buf[ids] > 0)]
        if not len(ids):
            return records
        # A bounded number of batch transfers, instead of O(envs * fields)
        # GPU synchronizations inside the Python episode loop.
        fields = {name: getattr(self, name)[ids].cpu().tolist() for name in
                  ("source", "level", "serial", "sample_id", "validated", "reset_failed",
                   "stage_success", "ever", "first", "start_pose", "joint1_travel", "joint1_max_excursion")}
        steps = env.episode_length_buf[ids].cpu().tolist()
        terms = {name: env.termination_manager.get_term(name)[ids].cpu().tolist()
                 for name in env.termination_manager.active_terms}
        rewards = {name: env.reward_manager._episode_sums[name][ids].cpu().tolist()
                   for name in env.reward_manager.active_terms} if self.cfg.evaluation else {}
        for row, i in enumerate(ids.cpu().tolist()):
            flags = {name: values[row] for name, values in terms.items()}
            failed = any(flags.get(k, False) for k in ("non_finite", "object_out_of_bound", "bulb_dropped", "lost_grasp", "curriculum_reset_failed"))
            success = fields["stage_success"][row] and not failed
            source, level = fields["source"][row], fields["level"][row]
            record = {"episode_id": self.episodes, "env_id": i, "env_episode": fields["serial"][row],
                      "seed": int(env.cfg.seed), "stage": self.cfg.stage, "source_stage": source,
                      "level": level, "scheduler_version": self.scheduler.version,
                      "fixed_fixture": self.cfg.fixed_bulb, "sample_id": fields["sample_id"][row],
                      "reset_valid": fields["validated"][row], "reset_failure": fields["reset_failed"][row],
                      "steps": steps[row], "success": success,
                      "events": dict(zip(EVENTS, fields["ever"][row])),
                      "first_seconds": dict(zip(EVENTS, fields["first"][row])),
                      "termination": flags, "initial_pose_w": fields["start_pose"][row],
                      "joint1_travel_rad": fields["joint1_travel"][row],
                      "joint1_max_excursion_rad": fields["joint1_max_excursion"][row],
                      "rewards": {name: (values[row] if math.isfinite(values[row]) else None)
                                  for name, values in rewards.items()}}
            if success and source == self.cfg.stage and self.cfg.stage in (1, 2) and not self.cfg.evaluation:
                self.bank.add(self.cfg.stage + 1, self.snapshot(i), {
                    "source_stage": self.cfg.stage, "physical_success": True, "seed": int(env.cfg.seed),
                    "episode_id": self.episodes, "step": int(env.common_step_counter),
                    "level": level, "source_checkpoint": self.cfg.checkpoint})
            self.scheduler.observe(source, level, success)
            self.episodes += 1
            records.append(record)
        self.finished_records.extend(records)
        # Retain a bounded queue for an evaluation consumer; JSONL is the complete record.
        self.finished_records = self.finished_records[-max(1000, env.num_envs * 2):]
        if records and self.log_path:
            with self.log_path.open("a") as stream:
                stream.writelines(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n" for record in records)
        return records


def reset_curriculum_scene(env, env_ids, x_range=(0., 0.), y_range=(0., 0.), table_half_height=0.,
                           support_bottom_offset=.044732484966516495, support_pose=None,
                           desktop_offset_y=assembly.DESKTOP_OFFSET_Y_M, yaw_range=(0., 0.)):
    params = dict(x_range=x_range, y_range=y_range, table_half_height=table_half_height,
                  support_bottom_offset=support_bottom_offset, support_pose=support_pose,
                  desktop_offset_y=desktop_offset_y, yaw_range=yaw_range)
    runtime = getattr(env, "_bulb_curriculum", None)
    if runtime is None:
        ids = assembly._ensure_ids(env, env_ids)
        reset_floor_bulb(env, ids, load_pregrasp(), **{**params, "x_range": (0., 0.), "y_range": (0., 0.), "yaw_range": (0., 0.)})
        reset_robot_to_grasp(env, env_ids)
    else:
        runtime.reset_physics(env_ids, params)


def curriculum_reward(env):
    runtime = getattr(env, "_bulb_curriculum", None)
    if runtime is None:
        return torch.zeros(env.num_envs, device=env.device)
    runtime.update()
    return runtime.reward_value


def curriculum_stage_complete(env):
    runtime = getattr(env, "_bulb_curriculum", None)
    if runtime is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    runtime.update()
    return runtime.stage_success if runtime.cfg.stage < 3 else torch.zeros_like(runtime.stage_success)


def curriculum_reset_failed(env):
    runtime = getattr(env, "_bulb_curriculum", None)
    if runtime is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    runtime.update()
    return runtime.reset_failed
