# Copyright (c) 2021-2026, ETH Zurich, NVIDIA CORPORATION, and BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Effective-action recurrent student for the Flexiv v2 distillation task."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from .student_teacher_depth_dagger_v5 import (
    DextrahStereoDepthTransformerStudent,
    StudentTeacherDepthDaggerV5,
    V5_DEPTH_OUTPUT_SIZE,
)


V2_DISTILLATION_ACTION_CONTRACT_NAME = "flexiv_v2_distillation_effective_raw"
V2_DISTILLATION_ACTION_CONTRACT_VERSION = 3
V2_DISTILLATION_ACTION_CONTRACT_BUFFER = "v2_distillation_action_contract_version"
V2_DISTILLATION_RAW_ACTION_LIMIT = 1.0
V2_DISTILLATION_RELATIVE_ACTION_SCALE = 0.1
V2_DISTILLATION_PROCESSED_ACTION_LIMIT = 0.1
V2_TEST_DISTILLATION_ACTION_CONTRACT_NAME = (
    "flexiv_v2_test_distillation_split_effective_raw"
)
V2_TEST_DISTILLATION_ACTION_CONTRACT_VERSION = 4
V2_TEST_DISTILLATION_RAW_ACTION_LIMITS = (1.0,) * 7 + (2.0,) * 21
V2_TEST_DISTILLATION_PROCESSED_ACTION_LIMITS = (0.1,) * 7 + (0.2,) * 21
V2_TEST_DISTILLATION_V5_ACTION_CONTRACT_VERSION = 5
V2_TEST_DISTILLATION_V5_RAW_ACTION_LIMITS = (2.5,) * 7 + (6.0,) * 21
V2_TEST_DISTILLATION_V5_PROCESSED_ACTION_LIMITS = (0.25,) * 7 + (0.6,) * 21
# Source-compatible aliases retained for existing dual-camera callers.  Contract
# v5 is now shared by both mono and dual-camera v2-test students.
V2_TEST_DUAL_CAMERA_DISTILLATION_ACTION_CONTRACT_VERSION = (
    V2_TEST_DISTILLATION_V5_ACTION_CONTRACT_VERSION
)
V2_TEST_DUAL_CAMERA_DISTILLATION_RAW_ACTION_LIMITS = (
    V2_TEST_DISTILLATION_V5_RAW_ACTION_LIMITS
)
V2_TEST_DUAL_CAMERA_DISTILLATION_PROCESSED_ACTION_LIMITS = (
    V2_TEST_DISTILLATION_V5_PROCESSED_ACTION_LIMITS
)


def make_v2_distillation_action_contract() -> dict[str, object]:
    """Return the serializable effective-action contract stored in checkpoints."""

    return {
        "name": V2_DISTILLATION_ACTION_CONTRACT_NAME,
        "version": V2_DISTILLATION_ACTION_CONTRACT_VERSION,
        "output_mode": "effective_raw_clipped",
        "teacher_target_mode": "clip_raw_to_effective",
        "previous_action_mode": "effective_raw",
        "raw_action_bounds": [
            -V2_DISTILLATION_RAW_ACTION_LIMIT,
            V2_DISTILLATION_RAW_ACTION_LIMIT,
        ],
        "relative_action_scale": V2_DISTILLATION_RELATIVE_ACTION_SCALE,
        "processed_action_bounds": [
            -V2_DISTILLATION_PROCESSED_ACTION_LIMIT,
            V2_DISTILLATION_PROCESSED_ACTION_LIMIT,
        ],
    }


def validate_v2_distillation_action_contract(contract: object) -> dict[str, object]:
    """Validate checkpoint metadata against the one supported v2 contract."""

    expected = make_v2_distillation_action_contract()
    if not isinstance(contract, Mapping):
        raise ValueError("v2 distillation checkpoint is missing action-contract metadata")
    actual = dict(contract)
    if actual != expected:
        raise ValueError(
            "v2 distillation checkpoint action-contract metadata is incompatible: "
            f"checkpoint={actual!r}, expected={expected!r}"
        )
    return actual


def make_v2_test_distillation_action_contract() -> dict[str, object]:
    """Return the legacy split arm/hand action contract used by monocular v2-test."""

    return {
        "name": V2_TEST_DISTILLATION_ACTION_CONTRACT_NAME,
        "version": V2_TEST_DISTILLATION_ACTION_CONTRACT_VERSION,
        "output_mode": "effective_raw_clipped",
        "teacher_target_mode": "clip_raw_to_effective",
        "previous_action_mode": "effective_raw",
        "raw_action_limits": list(V2_TEST_DISTILLATION_RAW_ACTION_LIMITS),
        "relative_action_scale": V2_DISTILLATION_RELATIVE_ACTION_SCALE,
        "processed_action_limits": list(
            V2_TEST_DISTILLATION_PROCESSED_ACTION_LIMITS
        ),
    }


def make_v2_test_dual_camera_distillation_action_contract() -> dict[str, object]:
    """Return the teacher-matched split action contract used by dual-camera v2-test."""

    return make_v2_test_distillation_v5_action_contract()


def make_v2_test_distillation_v5_action_contract() -> dict[str, object]:
    """Return the teacher-matched v5 action contract shared by mono and stereo."""

    return {
        "name": V2_TEST_DISTILLATION_ACTION_CONTRACT_NAME,
        "version": V2_TEST_DISTILLATION_V5_ACTION_CONTRACT_VERSION,
        "output_mode": "effective_raw_clipped",
        "teacher_target_mode": "clip_raw_to_effective",
        "previous_action_mode": "effective_raw",
        "raw_action_limits": list(V2_TEST_DISTILLATION_V5_RAW_ACTION_LIMITS),
        "relative_action_scale": V2_DISTILLATION_RELATIVE_ACTION_SCALE,
        "processed_action_limits": list(
            V2_TEST_DISTILLATION_V5_PROCESSED_ACTION_LIMITS
        ),
    }


def validate_v2_test_distillation_action_contract(
    contract: object,
) -> dict[str, object]:
    """Validate checkpoint metadata against the v2-test split contract."""

    expected = make_v2_test_distillation_action_contract()
    if not isinstance(contract, Mapping):
        raise ValueError(
            "v2-test distillation checkpoint is missing action-contract metadata"
        )
    actual = dict(contract)
    if actual != expected:
        raise ValueError(
            "v2-test distillation checkpoint action-contract metadata is "
            f"incompatible: checkpoint={actual!r}, expected={expected!r}"
        )
    return actual


def validate_v2_test_dual_camera_distillation_action_contract(
    contract: object,
) -> dict[str, object]:
    """Validate checkpoint metadata against the dual-camera v2-test contract."""

    return validate_v2_test_distillation_v5_action_contract(contract)


def validate_v2_test_distillation_v5_action_contract(
    contract: object,
) -> dict[str, object]:
    """Validate checkpoint metadata against the shared v2-test v5 contract."""

    expected = make_v2_test_distillation_v5_action_contract()
    if not isinstance(contract, Mapping):
        raise ValueError(
            "v2-test contract-v5 checkpoint is missing action-contract metadata"
        )
    actual = dict(contract)
    if actual != expected:
        raise ValueError(
            "v2-test contract-v5 checkpoint action-contract "
            f"metadata is incompatible: checkpoint={actual!r}, expected={expected!r}"
        )
    return actual


def v2_test_distillation_checkpoint_action_contract_version(
    checkpoint: object,
) -> int:
    """Return and validate the v2-test action contract declared by a checkpoint."""

    if not isinstance(checkpoint, Mapping):
        raise ValueError("v2-test checkpoint payload must be a mapping")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("v2-test checkpoint is missing model_state_dict")
    value = state_dict.get(V2_DISTILLATION_ACTION_CONTRACT_BUFFER)
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise ValueError(
            "v2-test checkpoint is missing its scalar action-contract buffer"
        )
    version = int(value.item())
    infos = checkpoint.get("infos")
    contract = (
        infos.get("v2_distillation_action_contract")
        if isinstance(infos, Mapping)
        else None
    )
    if version == V2_TEST_DISTILLATION_ACTION_CONTRACT_VERSION:
        validate_v2_test_distillation_action_contract(contract)
    elif version == V2_TEST_DISTILLATION_V5_ACTION_CONTRACT_VERSION:
        validate_v2_test_distillation_v5_action_contract(contract)
    else:
        raise ValueError(
            "unsupported v2-test distillation action-contract version: "
            f"{version!r}"
        )
    return version


class StudentTeacherV2Distillation(StudentTeacherDepthDaggerV5):
    """V5 visual trunk with bounded, behaviorally identifiable v2 actions."""

    action_contract_name = V2_DISTILLATION_ACTION_CONTRACT_NAME
    action_contract_version = V2_DISTILLATION_ACTION_CONTRACT_VERSION

    def __init__(self, *args, **kwargs):
        # V2 localizes the current object from the current image alone. The
        # action head still consumes the fused low-dimensional/visual history
        # through the inherited LSTM.
        kwargs["auxiliary_input_source"] = "visual"
        super().__init__(*args, **kwargs)
        # The inherited V5 buffer denotes the unbounded raw-action contract. It
        # must not appear in v2 checkpoints, whose semantics are intentionally
        # independent rather than a new revision of contract 3.
        del self.action_output_contract_version
        self.register_buffer(
            V2_DISTILLATION_ACTION_CONTRACT_BUFFER,
            torch.tensor(self.action_contract_version, dtype=torch.int64),
        )
        self._last_unclipped_action_mean = torch.zeros_like(self._last_action_mean)
        self.last_teacher_clip_fraction = 0.0
        self.last_student_clip_fraction = 0.0
        self.last_teacher_raw_min = 0.0
        self.last_teacher_raw_max = 0.0
        self.last_student_raw_min = 0.0
        self.last_student_raw_max = 0.0
        self._reset_action_diagnostic_accumulators()

    def effective_raw_action(self, actions: torch.Tensor) -> torch.Tensor:
        return actions.clamp(
            -V2_DISTILLATION_RAW_ACTION_LIMIT,
            V2_DISTILLATION_RAW_ACTION_LIMIT,
        )

    def effective_raw_action_limits(self, actions: torch.Tensor) -> torch.Tensor:
        return actions.new_full((actions.shape[-1],), V2_DISTILLATION_RAW_ACTION_LIMIT)

    def action_contract(self) -> dict[str, object]:
        return make_v2_distillation_action_contract()

    def normalize_actions(
        self,
        actions: torch.Tensor,
        clip: bool = True,
    ) -> torch.Tensor:
        return self.effective_raw_action(actions) if clip else actions

    def _reset_action_diagnostic_accumulators(self) -> None:
        for role in ("teacher", "student"):
            setattr(self, f"_{role}_diagnostic_elements", 0)
            setattr(self, f"_{role}_diagnostic_clipped", 0)
            setattr(self, f"_{role}_diagnostic_raw_min", float("inf"))
            setattr(self, f"_{role}_diagnostic_raw_max", float("-inf"))

    def _record_raw_diagnostics(
        self,
        role: str,
        actions: torch.Tensor,
    ) -> None:
        element_count = int(actions.numel())
        previous_elements = int(
            getattr(self, f"_{role}_diagnostic_elements", 0)
        )
        limits = self.effective_raw_action_limits(actions)
        clipped_count = int((actions.abs() > limits).sum().item())
        total_elements = previous_elements + element_count
        total_clipped = int(
            getattr(self, f"_{role}_diagnostic_clipped", 0)
        ) + clipped_count
        raw_min = min(
            float(getattr(self, f"_{role}_diagnostic_raw_min", float("inf"))),
            float(actions.min().item()),
        )
        raw_max = max(
            float(getattr(self, f"_{role}_diagnostic_raw_max", float("-inf"))),
            float(actions.max().item()),
        )
        setattr(self, f"_{role}_diagnostic_elements", total_elements)
        setattr(self, f"_{role}_diagnostic_clipped", total_clipped)
        setattr(self, f"_{role}_diagnostic_raw_min", raw_min)
        setattr(self, f"_{role}_diagnostic_raw_max", raw_max)
        setattr(
            self,
            f"last_{role}_clip_fraction",
            total_clipped / max(total_elements, 1),
        )
        setattr(self, f"last_{role}_raw_min", raw_min)
        setattr(self, f"last_{role}_raw_max", raw_max)

    @property
    def unclipped_action_mean(self) -> torch.Tensor:
        return self._last_unclipped_action_mean

    def act_inference(self, obs):
        raw_actions = super().act_inference(obs)
        effective_actions = self.effective_raw_action(raw_actions)
        self._last_unclipped_action_mean = raw_actions
        self._last_action_mean = effective_actions
        self._record_raw_diagnostics("student", raw_actions)
        return effective_actions

    def evaluate(self, obs):
        raw_actions = super().evaluate(obs)
        effective_actions = self.effective_raw_action(raw_actions)
        self._record_raw_diagnostics("teacher", raw_actions)
        return effective_actions

    def load_state_dict(self, state_dict, strict=True):
        if any(key.startswith("actor.") for key in state_dict):
            # PPO teacher checkpoints do not carry a student action contract.
            return super().load_state_dict(state_dict, strict=strict)
        if any(key.startswith("student.") for key in state_dict):
            contract = state_dict.get(V2_DISTILLATION_ACTION_CONTRACT_BUFFER)
            if (
                not isinstance(contract, torch.Tensor)
                or contract.numel() != 1
                or int(contract.item()) != self.action_contract_version
                or "action_output_contract_version" in state_dict
            ):
                raise ValueError(
                    f"This {self.action_contract_name} student checkpoint uses an incompatible "
                    "action-space contract. Contract-3/raw V5 checkpoints cannot be "
                    "resumed or played with this task; retrain the student."
                )
            nn.Module.load_state_dict(self, state_dict, strict=strict)
            self.loaded_teacher = True
            if self.enable_hybrid_rl:
                self.hybrid_teacher_rl_initialized = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            self._hidden_state = None
            return True
        raise ValueError("state_dict does not contain student or teacher parameters")

    def diagnostics(self):
        values = {
            "v2_distillation_action_loss": self.last_action_loss,
            "v2_distillation_auxiliary_object_position_loss": self.last_auxiliary_loss,
            "v2_distillation_action_contract_version": float(
                self.v2_distillation_action_contract_version.item()
            ),
            "v2_distillation_teacher_clip_fraction": self.last_teacher_clip_fraction,
            "v2_distillation_student_clip_fraction": self.last_student_clip_fraction,
            "v2_distillation_teacher_raw_min": self.last_teacher_raw_min,
            "v2_distillation_teacher_raw_max": self.last_teacher_raw_max,
            "v2_distillation_student_raw_min": self.last_student_raw_min,
            "v2_distillation_student_raw_max": self.last_student_raw_max,
        }
        self._reset_action_diagnostic_accumulators()
        return values


class StudentTeacherV2TestDistillationV4(StudentTeacherV2Distillation):
    """Legacy v2-test student with arm ±1 and hand ±2 effective-raw actions."""

    action_contract_name = V2_TEST_DISTILLATION_ACTION_CONTRACT_NAME
    action_contract_version = V2_TEST_DISTILLATION_ACTION_CONTRACT_VERSION
    raw_action_limits = V2_TEST_DISTILLATION_RAW_ACTION_LIMITS

    def effective_raw_action(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.shape[-1] != len(self.raw_action_limits):
            raise ValueError(
                "v2-test distillation requires 28 action dimensions, "
                f"got {actions.shape[-1]}."
            )
        limits = actions.new_tensor(self.raw_action_limits)
        return torch.maximum(torch.minimum(actions, limits), -limits)

    def effective_raw_action_limits(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.shape[-1] != len(self.raw_action_limits):
            raise ValueError(
                "v2-test distillation requires 28 action dimensions, "
                f"got {actions.shape[-1]}."
            )
        return actions.new_tensor(self.raw_action_limits)

    def action_contract(self) -> dict[str, object]:
        return make_v2_test_distillation_action_contract()

    def normalize_actions(
        self,
        actions: torch.Tensor,
        clip: bool = True,
    ) -> torch.Tensor:
        limits = actions.new_tensor(self.raw_action_limits)
        normalized = actions / limits
        return normalized.clamp(-1.0, 1.0) if clip else normalized


class StudentTeacherV2TestDistillation(StudentTeacherV2TestDistillationV4):
    """Current mono v2-test student matching the Stable teacher action limits."""

    action_contract_version = V2_TEST_DISTILLATION_V5_ACTION_CONTRACT_VERSION
    raw_action_limits = V2_TEST_DISTILLATION_V5_RAW_ACTION_LIMITS

    def action_contract(self) -> dict[str, object]:
        return make_v2_test_distillation_v5_action_contract()


class StudentTeacherV2TestDualCameraDistillation(
    StudentTeacherV2TestDistillation
):
    """V2-test stereo student matching the teacher's arm/hand action limits."""

    action_contract_version = (
        V2_TEST_DUAL_CAMERA_DISTILLATION_ACTION_CONTRACT_VERSION
    )
    raw_action_limits = V2_TEST_DUAL_CAMERA_DISTILLATION_RAW_ACTION_LIMITS

    def action_contract(self) -> dict[str, object]:
        return make_v2_test_dual_camera_distillation_action_contract()

    @staticmethod
    def _student_model_class():
        return DextrahStereoDepthTransformerStudent

    def __init__(self, *args, **kwargs):
        secondary_group = kwargs.get("student_secondary_depth_obs_group")
        if not isinstance(secondary_group, str) or not secondary_group:
            raise ValueError(
                "dual-camera v2-test requires student_secondary_depth_obs_group"
            )
        depth_output_size = tuple(
            int(value)
            for value in kwargs.get("depth_output_size", V5_DEPTH_OUTPUT_SIZE)
        )
        if depth_output_size != V5_DEPTH_OUTPUT_SIZE:
            raise ValueError(
                "dual-camera v2-test requires depth_output_size "
                f"{V5_DEPTH_OUTPUT_SIZE}, got {depth_output_size}"
            )
        super().__init__(*args, **kwargs)


__all__ = [
    "StudentTeacherV2Distillation",
    "StudentTeacherV2TestDistillation",
    "StudentTeacherV2TestDistillationV4",
    "StudentTeacherV2TestDualCameraDistillation",
    "V2_DISTILLATION_ACTION_CONTRACT_BUFFER",
    "V2_DISTILLATION_ACTION_CONTRACT_NAME",
    "V2_DISTILLATION_ACTION_CONTRACT_VERSION",
    "V2_DISTILLATION_PROCESSED_ACTION_LIMIT",
    "V2_DISTILLATION_RAW_ACTION_LIMIT",
    "V2_DISTILLATION_RELATIVE_ACTION_SCALE",
    "V2_TEST_DISTILLATION_ACTION_CONTRACT_NAME",
    "V2_TEST_DISTILLATION_ACTION_CONTRACT_VERSION",
    "V2_TEST_DISTILLATION_PROCESSED_ACTION_LIMITS",
    "V2_TEST_DISTILLATION_RAW_ACTION_LIMITS",
    "V2_TEST_DISTILLATION_V5_ACTION_CONTRACT_VERSION",
    "V2_TEST_DISTILLATION_V5_PROCESSED_ACTION_LIMITS",
    "V2_TEST_DISTILLATION_V5_RAW_ACTION_LIMITS",
    "V2_TEST_DUAL_CAMERA_DISTILLATION_ACTION_CONTRACT_VERSION",
    "V2_TEST_DUAL_CAMERA_DISTILLATION_PROCESSED_ACTION_LIMITS",
    "V2_TEST_DUAL_CAMERA_DISTILLATION_RAW_ACTION_LIMITS",
    "make_v2_distillation_action_contract",
    "make_v2_test_dual_camera_distillation_action_contract",
    "make_v2_test_distillation_action_contract",
    "make_v2_test_distillation_v5_action_contract",
    "validate_v2_distillation_action_contract",
    "validate_v2_test_dual_camera_distillation_action_contract",
    "validate_v2_test_distillation_action_contract",
    "validate_v2_test_distillation_v5_action_contract",
    "v2_test_distillation_checkpoint_action_contract_version",
]
