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
    TriModalCrossAutoencoder,
    TriModalCrossAutoencoderCfg,
)


@pytest.fixture()
def cfg() -> TriModalCrossAutoencoderCfg:
    return TriModalCrossAutoencoderCfg(
        image_height=32,
        image_width=48,
        marker_count=12,
        marker_summary_tokens=4,
        d_model=64,
        num_heads=4,
        image_base_channels=8,
        marker_transformer_layers=1,
        cross_layers=2,
        cross_summary_tokens=2,
        ffn_ratio=2,
        decoder_base_channels=32,
        dropout=0.0,
    )


def _inputs(cfg: TriModalCrossAutoencoderCfg, batch_size: int = 2):
    marker = torch.rand(batch_size, cfg.marker_count, cfg.marker_input_dim)
    marker[..., 4] = 1.0
    return {
        "rgb": torch.rand(batch_size, cfg.rgb_channels, cfg.image_height, cfg.image_width),
        "depth": torch.rand(
            batch_size, cfg.depth_channels, cfg.image_height, cfg.image_width
        ),
        "marker": marker,
    }


def test_cross_autoencoder_reconstructs_three_modalities(cfg):
    model = TriModalCrossAutoencoder(cfg).eval()
    inputs = _inputs(cfg)

    with torch.no_grad():
        output = model(**inputs)

    assert output["latent"].shape == (2, cfg.d_model)
    assert output["rgb_recon"].shape == inputs["rgb"].shape
    assert output["depth_recon"].shape == inputs["depth"].shape
    assert output["marker_recon"].shape == (2, cfg.marker_count, 2)
    assert set(output["cross_tokens"]) == {"rgb", "depth", "marker"}
    assert all(torch.isfinite(value).all() for value in output["modality_latents"].values())


def test_each_crossed_stream_changes_when_a_peer_changes(cfg):
    torch.manual_seed(11)
    model = TriModalCrossAutoencoder(cfg).eval()
    inputs = _inputs(cfg, batch_size=1)

    with torch.no_grad():
        first = model.encode_with_diagnostics(**inputs)
        changed = dict(inputs)
        changed["depth"] = 1.0 - inputs["depth"]
        second = model.encode_with_diagnostics(**changed)

    assert not torch.allclose(
        first["cross_tokens"]["rgb"], second["cross_tokens"]["rgb"]
    )
    assert not torch.allclose(
        first["cross_tokens"]["marker"], second["cross_tokens"]["marker"]
    )


def test_cross_autoencoder_gradients_reach_all_stages(cfg):
    model = TriModalCrossAutoencoder(cfg).train()
    output = model(**_inputs(cfg))
    loss = (
        output["rgb_recon"].mean()
        + output["depth_recon"].mean()
        + output["marker_recon"].square().mean()
        + output["latent"].square().mean()
    )
    loss.backward()

    gradients = (
        model.rgb_encoder.stem[0].weight.grad,
        model.depth_encoder.stem[0].weight.grad,
        model.marker_encoder.motion_mlp[0].weight.grad,
        model.cross_blocks[0].cross_attention["rgb"].in_proj_weight.grad,
        model.latent_projection[0].weight.grad,
        model.rgb_decoder.output_head.weight.grad,
        model.depth_decoder.output_head.weight.grad,
        model.marker_decoder.output_head[-1].weight.grad,
    )
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_cross_autoencoder_requires_three_observations(cfg):
    model = TriModalCrossAutoencoder(cfg)
    inputs = _inputs(cfg)

    with pytest.raises(TypeError):
        model(rgb=inputs["rgb"], depth=inputs["depth"])


def test_all_invalid_markers_fall_back_to_rgb_depth_cross_context(cfg):
    model = TriModalCrossAutoencoder(cfg).eval()
    inputs = _inputs(cfg)
    inputs["marker"][..., 4] = 0.0

    with torch.no_grad():
        output = model(**inputs)

    assert torch.count_nonzero(output["token_masks"]["marker"]) == 0
    assert torch.isfinite(output["latent"]).all()
    assert torch.isfinite(output["marker_recon"]).all()
