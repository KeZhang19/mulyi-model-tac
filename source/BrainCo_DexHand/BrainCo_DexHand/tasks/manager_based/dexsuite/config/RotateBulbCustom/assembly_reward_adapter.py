"""Isaac reward-only adapter and episode diagnostics for the assembly ledger."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from isaaclab.managers import ManagerTermBase, RewardTermCfg

from . import assembly
from .assembly_lift import bulb_collision_vertices, bulb_table_clearance
from .assembly_reward_v2 import AssemblyRewardLedger, COMPONENTS, REWARD_VERSION, RewardSample
from .env_cfg import rotate_bulb_finger_forces, rotate_bulb_hand_shell_distance


def reward_contract(old_reward):
    parent = Path(__file__).parent
    return {
        "version": REWARD_VERSION, "OLD_REWARD": bool(old_reward),
        "sources": {name: hashlib.sha256((parent / name).read_bytes()).hexdigest()
                    for name in ("assembly_reward_v2.py", "assembly_reward_adapter.py")},
    }


def configure_assembly_rewards(cfg):
    """Only replace rewards, after Hydra resolves OLD_REWARD; retain old path."""
    if type(cfg.OLD_REWARD) is not bool:
        raise ValueError("OLD_REWARD must be True or False")
    # Remove a previous configuration of this adapter if a factory is reused.
    for name in COMPONENTS:
        term = getattr(cfg.rewards, name, None)
        if term is not None and getattr(term, "func", None) is reward_component:
            delattr(cfg.rewards, name)
    if cfg.OLD_REWARD:
        cfg.rewards.reward_debug = RewardTermCfg(func=AssemblyRewardAdapter, weight=1., params={"component": None})
        return
    # Keep exactly the original two action costs. No physical/task writes.
    for name, term in list(vars(cfg.rewards).items()):
        if term is not None and hasattr(term, "weight") and name not in ("action_l2", "action_rate_l2"):
            setattr(cfg.rewards, name, None)
    for name in COMPONENTS:
        setattr(cfg.rewards, name, RewardTermCfg(
            func=reward_component, weight=1., params={"component": name}))
    # Exactly one class term owns reset; all other components read its cache.
    cfg.rewards.contact_stability = RewardTermCfg(
        func=AssemblyRewardAdapter, weight=1., params={"component": "contact_stability"})


def reward_component(env, component):
    adapter = env._assembly_reward_adapter
    adapter.update()
    return adapter.ledger.credits[component] / env.step_dt


class AssemblyRewardAdapter(ManagerTermBase):
    """One update/control-step, one reset/environment, zero success side effects.

    reward/* values are mean completed-episode credits (not rates or divided
    by max episode length). reason tags are multi-label fractions of sampled
    steps; JSONL also records last predicates and hold interruptions.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.env = env
        settings = getattr(env.cfg, "bulb_curriculum", None)
        stage = settings.stage if settings is not None else 4
        holds = (settings.grasp_hold_s, settings.lift_hold_s, settings.hover_hold_s) if settings else (1., .3, .5)
        self.ledger = AssemblyRewardLedger(env.num_envs, env.device, env.step_dt, stage, holds)
        self.palm = env.scene["robot"].body_names.index("palm")
        self.vertices = bulb_collision_vertices(env.scene.stage, env.scene["object"].root_physx_view.prim_paths[0], env.device)
        self.last_step = -1
        self.last_source = torch.full((env.num_envs,), stage, device=env.device, dtype=torch.long)
        self.last_success = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.action_sums = {name: torch.zeros(env.num_envs, device=env.device) for name in ("action_l2", "action_rate_l2")}
        self.episode_serial = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.old_reward = env.cfg.OLD_REWARD
        self.log_path = Path(env.cfg.log_dir) / "reward_debug_episodes.jsonl" if env.cfg.log_dir else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            (self.log_path.parent / "assembly_reward_contract.json").write_text(
                json.dumps(reward_contract(self.old_reward), indent=2) + "\n")
        env._assembly_reward_adapter = self

    def __call__(self, env, component=None):
        self.update()
        if component is None:
            return torch.zeros(env.num_envs, device=env.device)
        return self.ledger.credits[component] / env.step_dt

    def update(self):
        env, ledger = self.env, self.ledger
        if self.last_step == env.common_step_counter:
            return
        self.last_step = env.common_step_counter
        state = getattr(env, "_bulb_assembly_state", None)
        if state is None:
            return
        # State geometry and final success were already updated by termination.
        # Do not call assembly_success, runtime.update, or advance physics here.
        bulb, robot, support = env.scene["object"].data, env.scene["robot"].data, env.scene["support"].data
        relative_pos, relative_quat = assembly.subtract_frame_transforms(
            robot.body_pos_w[:, self.palm], robot.body_quat_w[:, self.palm], bulb.root_pos_w, bulb.root_quat_w)
        matrix = env.scene.sensors["bulb_table_contact"].data.force_matrix_w
        table = (matrix.norm(dim=-1).flatten(1).sum(-1) if matrix is not None else
                 torch.full((env.num_envs,), float("inf"), device=env.device))
        runtime = getattr(env, "_bulb_curriculum", None)
        valid = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        failed = torch.zeros_like(valid)
        for name in ("non_finite", "object_out_of_bound", "bulb_dropped", "lost_grasp", "curriculum_reset_failed"):
            if name in env.termination_manager.active_terms:
                failed |= env.termination_manager.get_term(name)
        if runtime is not None:
            valid &= runtime.active & runtime.validated & ~runtime.reset_failed
            ledger.initial_clearance.copy_(runtime.initial_clearance)
            source = runtime.source
            steps = torch.stack([counter.steps for counter in runtime.holds], -1)
            anchor = runtime.grasp_anchor
        else:
            source, steps, anchor = self.last_source, None, None
        axis_world = assembly.quat_apply(support.root_quat_w, torch.tensor(
            [1., 0., 0.], device=env.device).expand(env.num_envs, -1))
        axis_palm = assembly.quat_apply_inverse(robot.body_quat_w[:, self.palm], axis_world)
        sample = RewardSample(
            forces=rotate_bulb_finger_forces(env),
            relative_pose=torch.cat((relative_pos, relative_quat), -1),
            clearance=bulb_table_clearance(env, self.vertices), table_force=table,
            linear_speed=bulb.root_lin_vel_w.norm(dim=-1), angular_speed=bulb.root_ang_vel_w.norm(dim=-1),
            pair_distance=rotate_bulb_hand_shell_distance(env), depth=state.depth,
            radial=state.radial_error, angle=state.axis_error, helical=state.helical_error,
            relative_angle=state.relative_angle, advance=state.depth + state.anchor_x - assembly.ENTRY_ROOT_X_M,
            lifted=state.lifted, tracking=state.tracking_thread, engaged=state.engaged,
            installed=state.installed, socket_contact=state.last_contact, final_success=state.success,
            final_hold=env._assembly_hold_time, axis_in_palm=axis_palm,
            valid=valid, failed=failed, source=source,
            curriculum_steps=steps, curriculum_anchor=anchor,
        )
        ledger.update(sample)
        self.last_source.copy_(source)
        self.last_success.copy_((runtime.stage_success if runtime is not None else state.success) & ~failed)
        if self.old_reward:
            # Diagnostic counters still run; do not label shadow v2 credit as
            # actual old reward or change the old reward sum by a single bit.
            for value in ledger.credits.values():
                value.zero_()
            for value in ledger.episode_sums.values():
                value.zero_()
        for name, total in self.action_sums.items():
            term = getattr(env.cfg.rewards, name)
            total += term.func(env, **term.params) * term.weight * env.step_dt

    def reset(self, env_ids=None):
        env, ledger = self.env, self.ledger
        ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
        finished = ids[ledger.elapsed[ids] > 0]
        if finished.numel():
            self._log_finished(finished)
        clearance = bulb_table_clearance(env, self.vertices, ids)
        ledger.reset(ids, clearance)
        for value in self.action_sums.values():
            value[ids] = 0
        self.last_success[ids] = False
        self.episode_serial[ids] += 1
        # Do not invalidate the global step cache on partial reset: another
        # reward consumer in the same step must not update surviving rows twice.

    def _log_finished(self, ids):
        env, ledger = self.env, self.ledger
        metrics = env.extras.setdefault("log", {})
        sums = {**ledger.episode_sums, **self.action_sums}
        for name, values in sums.items():
            metrics["reward/" + name] = values[ids].mean().clone()
        metrics["reward/OLD_REWARD"] = torch.tensor(float(self.old_reward), device=env.device)
        debug = []
        for stage, reasons in enumerate(ledger.reason_masks):
            for reason in reasons:
                metrics[f"stage{stage}_fail_reason/{reason}"] = (
                    ledger.reason_steps[stage][reason][ids].float() / ledger.elapsed[ids].clamp_min(1)).mean()
            debug.append({
                "last": {k: v[ids].cpu().tolist() for k, v in reasons.items()},
                "steps": {k: v[ids].cpu().tolist() for k, v in ledger.reason_steps[stage].items()},
                "interruptions": {k: v[ids].cpu().tolist() for k, v in ledger.interruptions[stage].items()},
            })
        if self.log_path is None:
            return
        values = {k: v[ids].cpu().tolist() for k, v in sums.items()}
        serial, source = self.episode_serial[ids].cpu().tolist(), self.last_source[ids].cpu().tolist()
        successes = self.last_success[ids].cpu().tolist()
        elapsed, maximum = ledger.elapsed[ids].cpu().tolist(), ledger.max_holds[ids].cpu().tolist()
        records = []
        for row, env_id in enumerate(ids.cpu().tolist()):
            record = {"env_id": env_id, "episode": serial[row], "stage": ledger.stage,
                      "source_stage": source[row], "OLD_REWARD": self.old_reward,
                      "reward_version": REWARD_VERSION, "success": successes[row],
                      "steps": elapsed[row], "max_hold_steps": maximum[row],
                      "reward_components": {k: v[row] for k, v in values.items()},
                      "diagnostic_semantics": "multi_label_predicates_not_exclusive_termination_reasons"}
            for stage, entry in enumerate(debug):
                record[f"stage{stage}_fail_reason"] = [k for k, v in entry["last"].items() if v[row]]
                record[f"stage{stage}_fail_steps"] = {k: v[row] for k, v in entry["steps"].items()}
                record[f"stage{stage}_hold_interruptions"] = {k: v[row] for k, v in entry["interruptions"].items()}
            records.append(json.dumps(record, allow_nan=False) + "\n")
        with self.log_path.open("a") as stream:
            stream.writelines(records)
