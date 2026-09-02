from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "source" / "BrainCo_DexHand"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from BrainCo_DexHand.tactile_representation import (  # noqa: E402
    CrossModalTactileNetworkCfg,
    RobustCrossModalTactileNetwork,
)


@pytest.fixture()
def network_cfg() -> CrossModalTactileNetworkCfg:
    return CrossModalTactileNetworkCfg(
        image_height=32,
        image_width=48,
        marker_count=12,
        marker_input_dim=5,
        marker_output_dim=2,
        marker_summary_tokens=4,
        d_model=64,
        num_heads=4,
        image_base_channels=8,
        marker_transformer_layers=1,
        fusion_layers=2,
        ffn_ratio=2,
        decoder_base_channels=32,
        dropout=0.0,
    )


def make_inputs(cfg: CrossModalTactileNetworkCfg, batch_size: int = 2):
    return {
        "rgb": torch.rand(batch_size, cfg.rgb_channels, cfg.image_height, cfg.image_width),
        "depth": torch.rand(batch_size, cfg.depth_channels, cfg.image_height, cfg.image_width),
        "marker": torch.rand(batch_size, cfg.marker_count, cfg.marker_input_dim),
    }


def test_default_marker_schema_uses_calibration_as_input_and_motion_as_output():
    cfg = CrossModalTactileNetworkCfg()

    assert cfg.marker_input_dim == 5
    assert cfg.marker_output_dim == 2
    assert cfg.marker_summary_tokens == 4


def test_full_input_reconstructs_all_modalities(network_cfg):
    model = RobustCrossModalTactileNetwork(network_cfg).eval()
    inputs = make_inputs(network_cfg)

    with torch.no_grad():
        output = model(**inputs)

    assert output["latent"].shape == (2, network_cfg.d_model)
    assert output["rgb_recon"].shape == inputs["rgb"].shape
    assert output["depth_recon"].shape == inputs["depth"].shape
    assert output["marker_recon"].shape == (
        2,
        network_cfg.marker_count,
        network_cfg.marker_output_dim,
    )
    weights = torch.stack(tuple(output["weights"].values()), dim=1)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1.0e-6)
    assert torch.isfinite(output["latent"]).all()
    marker_start, marker_end = output["token_slices"]["marker"]
    assert marker_end - marker_start == network_cfg.marker_summary_tokens


def test_decode_from_latent_reconstructs_all_modalities_without_dynamic_context(
    network_cfg,
):
    cfg = CrossModalTactileNetworkCfg(
        **{
            **network_cfg.__dict__,
            "rgb_reference_residual": True,
            "rgb_depth_spatial_skip": True,
            "depth_rgb_spatial_skip": True,
            "marker_static_context": True,
            "marker_image_spatial_context": True,
        }
    )
    model = RobustCrossModalTactileNetwork(cfg).train()
    inputs = make_inputs(cfg)
    reference = torch.rand_like(inputs["rgb"])
    valid = inputs["marker"][..., 4] > 0.5
    latent = torch.randn(2, cfg.d_model, requires_grad=True)

    output = model.decode_from_latent(
        latent,
        rgb_reference=reference,
        marker_positions=inputs["marker"][..., :2],
        marker_valid_mask=valid,
    )
    loss = (
        output["rgb_recon"].mean()
        + output["depth_recon"].mean()
        + output["marker_recon"].square().mean()
    )
    loss.backward()

    assert output["rgb_recon"].shape == inputs["rgb"].shape
    assert output["depth_recon"].shape == inputs["depth"].shape
    assert output["marker_recon"].shape == (2, cfg.marker_count, cfg.marker_output_dim)
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()
    with pytest.raises(ValueError, match="rgb_reference is required"):
        model.decode_from_latent(
            latent.detach(),
            marker_positions=inputs["marker"][..., :2],
            marker_valid_mask=valid,
        )


def test_depth_decoder_starts_from_sparse_no_contact_prior(network_cfg):
    model = RobustCrossModalTactileNetwork(network_cfg).eval()
    inputs = make_inputs(network_cfg)

    with torch.no_grad():
        output = model(**inputs)

    assert float(output["depth_recon"].mean()) < 0.02
    assert float(output["depth_recon"].max()) < 0.03


def test_rgb_reference_branch_starts_as_identity_and_accepts_optional_reference(network_cfg):
    cfg = CrossModalTactileNetworkCfg(**{
        **network_cfg.__dict__,
        "rgb_reference_residual": True,
    })
    model = RobustCrossModalTactileNetwork(cfg).eval()
    inputs = make_inputs(cfg)
    reference = torch.rand_like(inputs["rgb"])

    with torch.no_grad():
        output = model(**inputs, rgb_reference=reference)

    assert torch.equal(output["rgb_recon"], reference)
    assert torch.isfinite(output["rgb_recon"]).all()


def test_rgb_reference_residual_range_is_configurable(network_cfg):
    cfg = CrossModalTactileNetworkCfg(**{
        **network_cfg.__dict__,
        "rgb_reference_residual": True,
        "rgb_reference_max_delta": 0.5,
    })
    model = RobustCrossModalTactileNetwork(cfg).eval()
    inputs = make_inputs(cfg, batch_size=1)
    reference = torch.full_like(inputs["rgb"], 0.2)
    model.rgb_decoder.reference_residual_head.weight.data.zero_()
    model.rgb_decoder.reference_residual_head.bias.data.fill_(10.0)

    with torch.no_grad():
        output = model(**inputs, rgb_reference=reference)

    assert torch.allclose(
        output["rgb_recon"], torch.full_like(reference, 0.7), atol=1.0e-4
    )


def test_restoration_requires_rgb_reference_when_residual_branch_is_enabled(network_cfg):
    cfg = CrossModalTactileNetworkCfg(**{
        **network_cfg.__dict__,
        "rgb_reference_residual": True,
        "predicted_soft_restoration": True,
    })
    model = RobustCrossModalTactileNetwork(cfg).eval()
    inputs = make_inputs(cfg)

    with pytest.raises(ValueError, match="rgb_reference is required"):
        model.restore_degraded_observation(**inputs)


def test_depth_spatial_skip_accepts_rgb_token_grid(network_cfg):
    cfg = CrossModalTactileNetworkCfg(**{
        **network_cfg.__dict__,
        "depth_rgb_spatial_skip": True,
    })
    model = RobustCrossModalTactileNetwork(cfg).eval()
    inputs = make_inputs(cfg)

    with torch.no_grad():
        output = model(**inputs)
        rgb_shifted = dict(inputs)
        rgb_shifted["rgb"] = inputs["rgb"] * 0.5
        shifted = model(**rgb_shifted)

    assert output["depth_recon"].shape == inputs["depth"].shape
    assert torch.isfinite(output["depth_recon"]).all()
    assert not torch.allclose(output["depth_recon"], shifted["depth_recon"])


def test_rgb_spatial_skip_accepts_depth_token_grid(network_cfg):
    cfg = CrossModalTactileNetworkCfg(**{
        **network_cfg.__dict__,
        "rgb_depth_spatial_skip": True,
    })
    model = RobustCrossModalTactileNetwork(cfg).eval()
    inputs = make_inputs(cfg)

    with torch.no_grad():
        output = model(**inputs)
        depth_shifted = dict(inputs)
        depth_shifted["depth"] = inputs["depth"] * 0.5
        shifted = model(**depth_shifted)

    assert output["rgb_recon"].shape == inputs["rgb"].shape
    assert torch.isfinite(output["rgb_recon"]).all()
    assert not torch.allclose(output["rgb_recon"], shifted["rgb_recon"])


def test_marker_static_context_is_available_when_marker_modality_is_masked(network_cfg):
    cfg = CrossModalTactileNetworkCfg(**{
        **network_cfg.__dict__,
        "marker_static_context": True,
    })
    model = RobustCrossModalTactileNetwork(cfg).eval()
    inputs = make_inputs(cfg)
    valid = torch.ones(2, cfg.marker_count, dtype=torch.bool)
    mask = torch.tensor([[True, True, False], [True, True, False]])

    with torch.no_grad():
        first = model(
            **inputs,
            marker_valid_mask=valid,
            modality_mask=mask,
        )
        shifted = dict(inputs)
        shifted["marker"] = inputs["marker"].clone()
        shifted["marker"][..., 0] = (shifted["marker"][..., 0] + 0.2).clamp(0.0, 1.0)
        second = model(
            **shifted,
            marker_valid_mask=valid,
            modality_mask=mask,
        )

    assert torch.equal(first["weights"]["marker"], torch.zeros(2))
    assert not torch.allclose(first["marker_recon"], second["marker_recon"])


def test_marker_image_context_uses_rgb_and_depth_without_dynamic_marker_motion(network_cfg):
    cfg = CrossModalTactileNetworkCfg(**{
        **network_cfg.__dict__,
        "marker_static_context": True,
        "marker_image_spatial_context": True,
    })
    model = RobustCrossModalTactileNetwork(cfg).eval()
    inputs = make_inputs(cfg)
    valid = torch.ones(2, cfg.marker_count, dtype=torch.bool)
    marker_masked = torch.tensor([[True, True, False], [True, True, False]])

    with torch.no_grad():
        first = model(
            **inputs,
            marker_valid_mask=valid,
            modality_mask=marker_masked,
        )
        rgb_shifted = dict(inputs)
        rgb_shifted["rgb"] = inputs["rgb"] * 0.25
        second = model(
            **rgb_shifted,
            marker_valid_mask=valid,
            modality_mask=marker_masked,
        )

    assert not torch.allclose(first["marker_recon"], second["marker_recon"])


def test_decoder_spatial_paths_respect_detached_reliability_gate(network_cfg):
    cfg = CrossModalTactileNetworkCfg(**{
        **network_cfg.__dict__,
        "cross_modal_reliability": True,
        "predicted_soft_restoration": True,
        "detach_reliability_gate": True,
        "rgb_depth_spatial_skip": True,
        "depth_rgb_spatial_skip": True,
        "marker_image_spatial_context": True,
    })
    model = RobustCrossModalTactileNetwork(cfg).train()
    inputs = make_inputs(cfg)

    output = model(**inputs)
    (
        output["rgb_recon"].mean()
        + output["depth_recon"].mean()
        + output["marker_recon"].mean()
    ).backward()

    # Decoder losses must not be able to make an unreliable sensor appear
    # healthy through a direct spatial bypass; only quality supervision should
    # update these heads when the gate is detached.
    assert all(
        parameter.grad is None
        for head in model.reliability_heads.values()
        for parameter in head.parameters()
    )


def test_marker_static_decoder_falls_back_when_marker_is_physically_missing(network_cfg):
    cfg = CrossModalTactileNetworkCfg(**{
        **network_cfg.__dict__,
        "marker_static_context": True,
    })
    model = RobustCrossModalTactileNetwork(cfg).eval()
    inputs = make_inputs(cfg)

    with torch.no_grad():
        output = model(rgb=inputs["rgb"])

    assert output["marker_recon"].shape == (2, cfg.marker_count, 2)
    assert torch.isfinite(output["marker_recon"]).all()


def test_marker_summaries_respond_to_motion_not_only_static_positions(network_cfg):
    model = RobustCrossModalTactileNetwork(network_cfg).eval()
    marker = torch.zeros(2, network_cfg.marker_count, 5)
    marker[..., 0] = torch.linspace(0.1, 0.9, network_cfg.marker_count)
    marker[..., 1] = torch.linspace(0.2, 0.8, network_cfg.marker_count)
    marker[..., 4] = 1.0
    marker[1, : network_cfg.marker_count // 2, 2] = 0.6
    valid = torch.ones(2, network_cfg.marker_count, dtype=torch.bool)

    with torch.no_grad():
        summaries, summary_valid = model.marker_encoder(marker, valid)

    assert summaries.shape == (2, network_cfg.marker_summary_tokens, network_cfg.d_model)
    assert bool(summary_valid.all())
    assert torch.linalg.vector_norm(summaries[1] - summaries[0]) > 1.0e-4


@pytest.mark.parametrize(
    ("used", "missing"),
    [
        (("rgb",), ("depth", "marker")),
        (("depth",), ("rgb", "marker")),
        (("marker",), ("rgb", "depth")),
        (("rgb", "depth"), ("marker",)),
        (("rgb", "marker"), ("depth",)),
        (("depth", "marker"), ("rgb",)),
    ],
)
def test_any_non_empty_modality_subset_can_forward(network_cfg, used, missing):
    model = RobustCrossModalTactileNetwork(network_cfg).eval()
    inputs = make_inputs(network_cfg)
    selected = {name: value if name in used else None for name, value in inputs.items()}

    with torch.no_grad():
        output = model(**selected)

    assert output["latent"].shape == (2, network_cfg.d_model)
    for name in missing:
        assert torch.count_nonzero(output["weights"][name]) == 0
        assert torch.count_nonzero(output["reliability"][name]) == 0
    weights = torch.stack(tuple(output["weights"].values()), dim=1)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1.0e-6)


def test_per_sample_modality_mask_is_strict(network_cfg):
    model = RobustCrossModalTactileNetwork(network_cfg).eval()
    inputs = make_inputs(network_cfg, batch_size=3)
    mask = torch.tensor(
        [
            [True, False, False],
            [False, True, False],
            [False, False, True],
        ]
    )

    with torch.no_grad():
        output = model(**inputs, modality_mask=mask)

    weights = torch.stack(tuple(output["weights"].values()), dim=1)
    assert torch.equal(weights > 0.0, mask)
    assert torch.allclose(weights.sum(dim=1), torch.ones(3), atol=1.0e-6)


def test_clean_reliability_probabilities_are_normalized_without_logit_competition(network_cfg):
    model = RobustCrossModalTactileNetwork(network_cfg).eval()
    inputs = make_inputs(network_cfg)
    logits = (8.0, 6.0, 3.0)
    for name, logit in zip(("rgb", "depth", "marker"), logits, strict=True):
        for parameter in model.reliability_heads[name].parameters():
            parameter.data.zero_()
        model.reliability_heads[name].mlp[-1].bias.data.fill_(logit)

    with torch.no_grad():
        output = model.encode_with_diagnostics(**inputs)

    probabilities = torch.sigmoid(torch.tensor(logits))
    expected = probabilities / probabilities.sum()
    actual = torch.stack(tuple(output["weights"].values()), dim=1)
    assert torch.allclose(actual, expected.expand_as(actual), atol=1.0e-6)
    assert bool((actual[:, 2] > 0.30).all())


def test_restoration_inference_detects_and_excludes_low_quality_observation(network_cfg):
    model = RobustCrossModalTactileNetwork(network_cfg).eval()
    inputs = make_inputs(network_cfg)
    for name, logit in zip(("rgb", "depth", "marker"), (-8.0, 8.0, 8.0), strict=True):
        for parameter in model.reliability_heads[name].parameters():
            parameter.data.zero_()
        model.reliability_heads[name].mlp[-1].bias.data.fill_(logit)

    with torch.no_grad():
        output = model.restore_degraded_observation(**inputs, quality_threshold=0.9)

    assert torch.equal(output["detected_degraded_modality"], torch.zeros(2, dtype=torch.int64))
    assert torch.equal(
        output["restoration_modality_mask"],
        torch.tensor([[False, True, True], [False, True, True]]),
    )
    assert torch.equal(output["restored_depth"], inputs["depth"])
    assert torch.equal(output["restored_marker_motion"], inputs["marker"][..., 2:4])
    assert not torch.equal(output["restored_rgb"], inputs["rgb"])


def test_predicted_restoration_keeps_all_observations_and_uses_soft_quality_gate(network_cfg):
    cfg = CrossModalTactileNetworkCfg(**{
        **network_cfg.__dict__,
        "cross_modal_reliability": True,
        "predicted_soft_restoration": True,
        "detach_reliability_gate": True,
    })
    model = RobustCrossModalTactileNetwork(cfg).eval()
    inputs = make_inputs(cfg)
    for name, logit in zip(("rgb", "depth", "marker"), (-8.0, 8.0, 8.0), strict=True):
        for parameter in model.reliability_heads[name].parameters():
            parameter.data.zero_()
        model.reliability_heads[name].mlp[-1].bias.data.fill_(logit)

    with torch.no_grad():
        output = model.restore_degraded_observation(**inputs, quality_threshold=0.9)

    assert torch.equal(output["detected_degraded_modality"], torch.zeros(2, dtype=torch.int64))
    assert bool(output["restoration_modality_mask"].all())
    assert bool((output["restoration_blend"][:, 0] >= 0.989).all())
    assert torch.count_nonzero(output["restoration_blend"][:, 1:]) == 0
    rgb_blend = output["restoration_blend"][:, 0, None, None, None]
    expected_rgb = inputs["rgb"] * (1.0 - rgb_blend) + output["rgb_recon"] * rgb_blend
    assert torch.allclose(output["restored_rgb"], expected_rgb)
    assert torch.equal(output["restored_depth"], inputs["depth"])
    assert torch.equal(output["restored_marker_motion"], inputs["marker"][..., 2:4])


def test_cross_modal_rgb_quality_changes_when_only_a_peer_changes(network_cfg):
    torch.manual_seed(13)
    cfg = CrossModalTactileNetworkCfg(**{
        **network_cfg.__dict__,
        "cross_modal_reliability": True,
    })
    model = RobustCrossModalTactileNetwork(cfg).eval()
    inputs = make_inputs(cfg, batch_size=1)

    with torch.no_grad():
        first = model.encode_with_diagnostics(**inputs)["reliability"]["rgb"]
        changed = dict(inputs)
        changed["depth"] = 1.0 - inputs["depth"]
        second = model.encode_with_diagnostics(**changed)["reliability"]["rgb"]

    assert not torch.allclose(first, second, atol=1.0e-7, rtol=0.0)


def test_empty_subset_is_rejected(network_cfg):
    model = RobustCrossModalTactileNetwork(network_cfg)
    inputs = make_inputs(network_cfg)
    mask = torch.tensor([[True, False, False], [False, False, False]])

    with pytest.raises(ValueError, match="at least one valid modality"):
        model(**inputs, modality_mask=mask)


def test_all_invalid_marker_row_needs_another_modality(network_cfg):
    model = RobustCrossModalTactileNetwork(network_cfg)
    inputs = make_inputs(network_cfg)
    marker_valid = torch.ones(2, network_cfg.marker_count, dtype=torch.bool)
    marker_valid[1] = False
    mask = torch.tensor([[False, False, True], [True, False, True]])

    with torch.no_grad():
        output = model(
            rgb=inputs["rgb"],
            marker=inputs["marker"],
            modality_mask=mask,
            marker_valid_mask=marker_valid,
        )

    assert output["weights"]["marker"][1] == 0.0
    assert output["weights"]["rgb"][1] == 1.0


def test_fusion_modality_mass_is_independent_of_valid_token_count(network_cfg):
    model = RobustCrossModalTactileNetwork(network_cfg).eval()
    inputs = make_inputs(network_cfg, batch_size=2)
    marker_valid = torch.zeros(2, network_cfg.marker_count, dtype=torch.bool)
    marker_valid[0, :2] = True
    marker_valid[1, :10] = True

    # Equal reliability logits make every present modality weight exactly one third.
    for head in model.reliability_heads.values():
        for parameter in head.parameters():
            parameter.data.zero_()

    # Remove content logits from the reported final attention layer so its mass
    # is determined only by the reliability-aware modality prior.
    final_attention = model.fusion.layers[-1].attention
    final_attention.query_projection.weight.data.zero_()
    final_attention.query_projection.bias.data.zero_()

    with torch.no_grad():
        output = model.encode_with_diagnostics(
            **inputs,
            marker_valid_mask=marker_valid,
        )

    expected_mass = torch.full(
        (2, network_cfg.num_heads),
        1.0 / 3.0,
        dtype=output["attention"].dtype,
    )
    for start, end in output["token_slices"].values():
        modality_mass = output["attention"][..., start:end].sum(dim=-1)
        assert torch.allclose(modality_mass, expected_mass, atol=1.0e-6)


def test_gradient_reaches_every_model_stage(network_cfg):
    model = RobustCrossModalTactileNetwork(network_cfg).train()
    output = model(**make_inputs(network_cfg))
    loss = (
        output["latent"].square().mean()
        + output["rgb_recon"].mean()
        + output["depth_recon"].mean()
        + output["marker_recon"].square().mean()
        + sum(value.mean() for value in output["reliability"].values())
    )
    loss.backward()

    representative_gradients = (
        model.rgb_encoder.stem[0].weight.grad,
        model.depth_encoder.stem[0].weight.grad,
        model.marker_encoder.motion_mlp[0].weight.grad,
        model.reliability_heads["rgb"].mlp[0].weight.grad,
        model.fusion.layers[0].attention.query_projection.weight.grad,
        model.rgb_decoder.output_head.weight.grad,
        model.depth_decoder.output_head.weight.grad,
        model.marker_decoder.output_head[-1].weight.grad,
    )
    assert all(gradient is not None for gradient in representative_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in representative_gradients)


def test_zero_depth_has_bounded_encoder_gradients(network_cfg):
    torch.manual_seed(0)
    model = RobustCrossModalTactileNetwork(network_cfg).train()
    inputs = make_inputs(network_cfg)
    inputs["depth"].zero_()

    output = model(**inputs)
    loss = output["latent"].square().mean() + output["depth_recon"].mean()
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.depth_encoder.parameters()
        if parameter.grad is not None
    ]
    squared_norm = sum(gradient.float().square().sum() for gradient in gradients)
    gradient_norm = squared_norm.sqrt()

    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert gradient_norm < 1.0e3
