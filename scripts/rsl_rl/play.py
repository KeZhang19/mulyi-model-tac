# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
from play_camera import add_camera_args, validate_camera_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL checkpoint or preview its recording camera.")
parser.add_argument("--video", action="store_true", default=False, help="Record one policy playback video and exit.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--bulb-legacy-diagnostic", action="store_true",
    help="Diagnose an original Custom bulb checkpoint with clipped actions and the revised environment rules.",
)
# append RSL-RL cli arguments
add_camera_args(parser)
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
validate_camera_args(parser, args_cli)
# always enable cameras to record video
if args_cli.video or args_cli.screenshot or args_cli.camera_preview:
    args_cli.enable_cameras = True
    if args_cli.num_envs is None:
        args_cli.num_envs = 1

# Use this checkout even when another BrainCo project is installed editable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source/BrainCo_DexHand"))

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time
from datetime import datetime, timezone

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)
import BrainCo_DexHand  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    from play_camera import apply_play_camera, configure_play_camera, save_camera_preview

    camera_rotation = configure_play_camera(env_cfg, task_name, args_cli)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    if args_cli.camera_preview:
        screenshot = args_cli.screenshot or Path("logs/camera_preview") / task_name / f"{stamp}.png"
        # Scene inspection needs no tactile encoder checkpoint or policy features.
        if hasattr(env_cfg, "tactile_policy_enabled"):
            env_cfg.tactile_policy_enabled = False
        if hasattr(env_cfg, "observation_history_length"):
            env_cfg.observation_history_length = 1
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
        try:
            env.reset(seed=env_cfg.seed)
            apply_play_camera(env, camera_rotation, camera=args_cli.camera)
            save_camera_preview(env, screenshot, task=task_name, camera_preset=args_cli.camera)
        finally:
            env.close()
        return

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    from bulb_training import is_bulb_task

    is_custom_bulb = is_bulb_task(task_name)
    if args_cli.bulb_legacy_diagnostic and not is_custom_bulb:
        raise ValueError("--bulb-legacy-diagnostic is only supported by Rotate-Bulb-Custom")
    legacy_bulb = False
    if is_custom_bulb:
        from bulb_training import configure_bulb_checkpoint

        legacy_bulb = configure_bulb_checkpoint(
            agent_cfg, resume_path, legacy_diagnostic=args_cli.bulb_legacy_diagnostic,
        )
        if legacy_bulb:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            log_dir = os.path.abspath(os.path.join("logs", "diagnostics", "bulb_legacy", stamp))
            print(f"[INFO] Legacy bulb diagnostic output: {log_dir}")

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    if hasattr(env_cfg, "repose_training_revision"):
        from repose_training import configure_repose_checkpoint

        env_cfg.repose_training_revision = configure_repose_checkpoint(agent_cfg, resume_path)
        env_cfg.repose_curriculum_enabled = False

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video or args_cli.screenshot else None)
    apply_play_camera(env, camera_rotation, camera=args_cli.camera)

    from BrainCo_DexHand.tactile_representation.policy import prepare_policy_run

    prepare_policy_run(env, log_dir, resume_path=resume_path)
    if is_custom_bulb:
        from bulb_training import prepare_bulb_policy_run

        prepare_bulb_policy_run(env, log_dir, resume_path=resume_path, legacy_diagnostic=legacy_bulb)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    video_dir = args_cli.video_dir or Path(log_dir) / "videos" / "play"
    video_prefix = f"{task_name}-{Path(resume_path).stem}-{stamp}"
    if args_cli.video:
        video_kwargs = {
            "video_folder": str(video_dir.expanduser().resolve()),
            "name_prefix": video_prefix,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording policy playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    if getattr(env.unwrapped, "_repose_training", None) is not None:
        from repose_run_state import install_repose_run_state

        install_repose_run_state(runner, env.unwrapped)
    runner.load(resume_path, map_location=agent_cfg.device)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    if is_custom_bulb:
        from bulb_training import bulb_inference_policy

        policy = bulb_inference_policy(policy, legacy_diagnostic=legacy_bulb)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(log_dir, "exported")
    from repose_training import repose_policy_for_export

    if is_custom_bulb:
        from bulb_training import bulb_policy_for_export

        export_policy = bulb_policy_for_export(policy_nn, legacy_diagnostic=legacy_bulb)
    else:
        export_policy = repose_policy_for_export(policy_nn)
    export_policy_as_jit(export_policy, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(export_policy, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt
    bulb_recorder = None
    if legacy_bulb:
        from bulb_diagnostics import BulbDiagnosticRecorder

        bulb_recorder = BulbDiagnosticRecorder(env, log_dir)

    # reset environment
    obs = env.get_observations()
    if args_cli.video or args_cli.screenshot:
        screenshot = args_cli.screenshot or video_dir / f"{video_prefix}-preview.png"
        save_camera_preview(env, screenshot, task=task_name, checkpoint=resume_path, camera_preset=args_cli.camera)
    timestep = 0
    # Closing RecordVideo also flushes a partial recording on interruption.
    try:
        while simulation_app.is_running():
            start_time = time.time()
            with torch.inference_mode():
                actions = policy(obs)
                obs, rewards, dones, _ = env.step(actions)
                if bulb_recorder is not None:
                    bulb_recorder.record(actions, rewards, dones)
                policy_nn.reset(dones)
            if args_cli.video:
                timestep += 1
                if timestep == args_cli.video_length:
                    break

            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        if bulb_recorder is not None:
            bulb_recorder.close()
        env.close()


if __name__ == "__main__":
    # run the main function
    try:
        main()
    finally:
        simulation_app.close(wait_for_replicator=False)
