from __future__ import annotations

from pathlib import Path
import runpy
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "source" / "BrainCo_DexHand"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from BrainCo_DexHand.tactile_representation import (  # noqa: E402
    TriModalCrossAutoencoder,
    TriModalCrossAutoencoderCfg,
)
from BrainCo_DexHand.tactile_representation.training import (  # noqa: E402
    TactileNormalization,
)
from BrainCo_DexHand.tactile_representation.training.tri_modal_objectives import (  # noqa: E402
    TriModalAutoencoderLossCfg,
    compute_tri_modal_autoencoder_objective,
)


def _batch(batch_size: int = 3):
    normalization = TactileNormalization(
        image_height=32,
        image_width=48,
        marker_count=12,
        marker_motion_scale_px=5.0,
    )
    marker_valid = torch.ones(batch_size, 12, dtype=torch.bool)
    marker_valid[:, -2:] = False
    marker = torch.rand(batch_size, 12, 5)
    marker[..., 2:4] = marker[..., 2:4] * 1.2 - 0.6
    marker[..., 4] = marker_valid
    marker[..., :4].masked_fill_(~marker_valid.unsqueeze(-1), 0.0)
    rgb = torch.rand(batch_size, 3, 32, 48)
    return {
        "rgb": rgb,
        "rgb_reference": rgb * 0.9,
        "depth": torch.rand(batch_size, 1, 32, 48) * 0.4,
        "marker": marker,
        "marker_valid": marker_valid,
        "contact": torch.ones(batch_size, dtype=torch.bool),
    }, normalization


def _model() -> TriModalCrossAutoencoder:
    return TriModalCrossAutoencoder(
        TriModalCrossAutoencoderCfg(
            image_height=32,
            image_width=48,
            marker_count=12,
            d_model=64,
            num_heads=4,
            image_base_channels=8,
            decoder_base_channels=32,
            marker_transformer_layers=1,
            marker_summary_tokens=4,
            cross_layers=1,
            cross_summary_tokens=2,
            ffn_ratio=2,
            dropout=0.0,
        )
    )


def test_tri_modal_training_objective_reconstructs_all_outputs_and_backpropagates():
    batch, normalization = _batch()
    model = _model().train()

    loss, metrics = compute_tri_modal_autoencoder_objective(
        model,
        batch,
        normalization=normalization,
        cfg=TriModalAutoencoderLossCfg(),
        generator=torch.Generator().manual_seed(7),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["loss/rgb"] > 0.0
    assert metrics["loss/depth"] > 0.0
    assert metrics["loss/marker"] > 0.0
    assert model.rgb_encoder.stem[0].weight.grad is not None
    assert model.depth_encoder.stem[0].weight.grad is not None
    assert model.marker_encoder.motion_mlp[0].weight.grad is not None
    assert model.cross_blocks[0].cross_attention["rgb"].in_proj_weight.grad is not None


def test_tri_modal_validation_modes_cover_clean_and_each_degraded_modality():
    trainer = runpy.run_path(
        str(REPO_ROOT / "scripts/tactile_representation/train_tri_modal_cross_autoencoder.py")
    )

    assert trainer["EVALUATION_MODES"] == ("clean", "rgb", "depth", "marker")
    assert trainer["_lr_lambda"](
        999, schedule="fixed", epochs=1000, warmup_epochs=0
    ) == 1.0


def test_tri_modal_training_defaults_select_independent_output_directory():
    trainer = runpy.run_path(
        str(REPO_ROOT / "scripts/tactile_representation/train_tri_modal_cross_autoencoder.py")
    )
    args = trainer["parse_args"]([])

    assert args.output == Path("runs/revo3_tri_modal_cross_v1")
    assert args.batch_size == 32
    assert args.grad_accum_steps == 2
    assert args.cross_layers == 2
    assert args.lr_schedule == "fixed"
