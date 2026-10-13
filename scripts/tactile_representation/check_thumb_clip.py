#!/usr/bin/env python3
"""CPU regression checks for native thumb pixels, old weights and CLIP wiring."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/BrainCo_DexHand"))
from BrainCo_DexHand.tactile_representation.config import CrossModalTactileNetworkCfg
from BrainCo_DexHand.tactile_representation.tri_modal_config import TriModalCrossAutoencoderCfg
from BrainCo_DexHand.tactile_representation.models.encoders import ImageTokenEncoder
from BrainCo_DexHand.tactile_representation.models.latent_alignment import TactileLatentAlignmentNetwork
from BrainCo_DexHand.tactile_representation.policy import build_tactile_encoder
from BrainCo_DexHand.tactile_representation.thumb_alignment import validate_thumb_checkpoint
from BrainCo_DexHand.tactile_representation.training.alignment_domain import AlignmentDomain
from BrainCo_DexHand.tactile_representation.training.data import TactileNormalization


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialization", type=Path, required=True)
    parser.add_argument("--real-domain", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, default=None,
                        help="Optional saved pre-change model files for exact numerical regression")
    args = parser.parse_args(argv)
    args.real_domain = args.real_domain.resolve()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    checks = {}
    models = {}
    with np.load(args.real_domain / "shards/shard_000000.npz", allow_pickle=False) as z:
        raw = {k: z[k] for k in z.files}
    norm = TactileNormalization(180, 320, 100)
    domain = AlignmentDomain(args.real_domain, args.real_domain / "manifest.json", norm, domain="real")
    checks["real_native_loader_50"] = domain.count == 50
    manifest = json.loads((args.real_domain / "manifest.json").read_text())
    for index, provenance in enumerate(manifest["provenance"]):
        p = Path(provenance["source"])
        assert hashlib.sha256(p.read_bytes()).hexdigest() == provenance["sha256"]
        with np.load(p, allow_pickle=False) as z:
            assert np.array_equal(raw["rgb"][index], z["warped_rgb"].transpose(2, 0, 1))
            assert np.array_equal(raw["depth_m"][index], (z["depth_sdk"][None] * .006).astype(np.float32))
    checks["sdk_rgb_depth_pixels_preserved_all_50"] = True
    inputs = {"rgb": torch.from_numpy(raw["rgb"][:2]).float() / 255,
              "depth": (torch.from_numpy(raw["depth_m"][:2]) / .003).clamp(0, 1),
              "marker": torch.from_numpy(raw["marker_2d"][:2]).clone(),
              "marker_valid_mask": torch.from_numpy(raw["marker_valid"][:2])}
    inputs["marker"][..., 0] /= 319
    inputs["marker"][..., 1] /= 179
    inputs["marker"][..., 2:4] /= 5
    for name in ("sim", "real"):
        source = torch.load(args.initialization / f"{name}_thumb_init.pt", map_location="cpu", weights_only=False)
        validate_thumb_checkpoint(source)
        model = build_tactile_encoder(source["network_type"], source["model_cfg"])
        model.load_state_dict(source["model_state"], strict=True)
        model.eval()
        seen_inputs, decoded_sizes = {}, {"rgb": [], "depth": []}
        handles = []
        for modality in ("rgb", "depth"):
            encoder, decoder = getattr(model, f"{modality}_encoder"), getattr(model, f"{modality}_decoder")
            handles.append(encoder.stem.register_forward_pre_hook(
                lambda _, values, key=modality: seen_inputs.update({key: values[0].detach().clone()})))
            for block in decoder.blocks:
                handles.append(block.register_forward_hook(
                    lambda _, values, result, key=modality: decoded_sizes[key].append(list(result.shape[-2:]))))
        with torch.inference_mode(), patch("torch.nn.functional.pad", side_effect=AssertionError("Input padding forbidden")):
            output = model(**inputs)
        for handle in handles:
            handle.remove()
        assert all(torch.equal(seen_inputs[k], inputs[k]) for k in seen_inputs)
        assert all(sizes == [[23, 40], [45, 80], [90, 160], [180, 320]] for sizes in decoded_sizes.values())
        checks[f"{name}_native_stem_pixels_no_pad_and_exact_decoder_pyramid"] = True
        assert output["rgb_recon"].shape == (2, 3, 180, 320)
        assert output["depth_recon"].shape == (2, 1, 180, 320)
        assert output["latent"].shape == (2, 256) and torch.isfinite(output["latent"]).all()
        assert torch.isfinite(output["rgb_recon"]).all() and torch.isfinite(output["depth_recon"]).all()
        checks[f"{name}_original_capacity_native_encode_decode"] = True
        models[name] = model
        del source, output
    aligner = TactileLatentAlignmentNetwork(models["sim"], models["real"],
                                           latent_dim=256, projection_dim=64)
    optimizer = torch.optim.Adam(aligner.trainable_parameters(), lr=1e-3)
    before = [p.detach().clone() for p in aligner.trainable_parameters()]
    outputs = aligner(inputs, inputs)
    loss = aligner.alignment_loss(outputs)
    assert torch.isfinite(loss) and outputs["z_real"].shape == (2, 64)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in aligner.trainable_parameters())
    optimizer.step()
    assert any(not torch.equal(a, p) for a, p in zip(before, aligner.trainable_parameters()))
    assert not any(p.grad is not None for m in models.values() for p in m.parameters())
    checks["original_clip_frozen_towers_projection_backward"] = True
    del models, aligner, optimizer
    for kind, cfg_cls in (("robust", CrossModalTactileNetworkCfg), ("tri_modal", TriModalCrossAutoencoderCfg)):
        try:
            cfg_cls(image_height=180)
        except ValueError:
            checks[f"{kind}_legacy_rejects_unflagged_180"] = True
        else:
            raise AssertionError("Unflagged legacy model accepted 180")
        cfg = cfg_cls(image_height=180, native_image_size=True, d_model=32, num_heads=4,
                      image_base_channels=8, decoder_base_channels=16, marker_transformer_layers=1)
        model = build_tactile_encoder(kind, asdict(cfg))
        out = model(**inputs)
        loss = out["rgb_recon"].mean() + out["depth_recon"].mean()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        checks[f"{kind}_native_decoder_backward"] = True
    cfg = CrossModalTactileNetworkCfg(image_height=180, native_image_size=True, d_model=32, num_heads=4,
        image_base_channels=8, decoder_base_channels=16, rgb_reference_residual=True,
        rgb_depth_spatial_skip=True, depth_rgb_spatial_skip=True)
    out = build_tactile_encoder("robust", asdict(cfg))(**inputs, rgb_reference=inputs["rgb"])
    assert out["rgb_recon"].shape == (2, 3, 180, 320)
    checks["native_reference_and_spatial_skip"] = True
    options = dict(in_channels=3, image_height=240, image_width=320, base_channels=8, d_model=32)
    image = torch.rand(1, 3, 240, 320)
    if args.baseline_dir is not None:
        # Load all saved encoder/decoder/network components together. Loading
        # only the old network class would accidentally use current components.
        package = types.ModuleType("_thumb_original")
        package.__path__ = [str(args.baseline_dir)]
        sys.modules[package.__name__] = package
        package_models = types.ModuleType("_thumb_original.models")
        package_models.__path__ = [str(args.baseline_dir), str(ROOT / "source/BrainCo_DexHand/BrainCo_DexHand/tactile_representation/models")]
        sys.modules[package_models.__name__] = package_models
        spec = importlib.util.spec_from_file_location("thumb_before_encoder", args.baseline_dir / "encoders.py")
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        original, current = old.ImageTokenEncoder(**options), ImageTokenEncoder(**options)
        current.load_state_dict(original.state_dict(), strict=True)
        with torch.no_grad():
            assert torch.equal(original(image), current(image))
        checks["240_encoder_exact_before_after"] = True
    for kind, filename, class_name, cfg_cls in (
        ("robust", "network.py", "RobustCrossModalTactileNetwork", CrossModalTactileNetworkCfg),
        ("tri_modal", "tri_modal_cross_autoencoder.py", "TriModalCrossAutoencoder", TriModalCrossAutoencoderCfg),
    ):
        cfg = cfg_cls(d_model=32, num_heads=4, image_base_channels=8, decoder_base_channels=16,
                      marker_transformer_layers=1, dropout=0)
        current = build_tactile_encoder(kind, asdict(cfg)).eval()
        if args.baseline_dir is not None:
            old = importlib.import_module("_thumb_original.models." + filename.removesuffix(".py"))
            original = getattr(old, class_name)(cfg).eval()
        else:
            original = build_tactile_encoder(kind, dict(asdict(cfg), native_image_size=True)).eval()
        current.load_state_dict(original.state_dict(), strict=True)
        legacy_inputs = dict(inputs, rgb=torch.rand(2, 3, 240, 320), depth=torch.rand(2, 1, 240, 320))
        with torch.no_grad():
            a, b = original(**legacy_inputs), current(**legacy_inputs)
        assert all(torch.equal(a[k], b[k]) for k in ("latent", "rgb_recon", "depth_recon", "marker_recon"))
        checks[f"{kind}_240_all_outputs_exact"] = True
    seen = []
    thumb = ImageTokenEncoder(**dict(options, image_height=180))
    handle = thumb.stem.register_forward_pre_hook(lambda _, values: seen.append(values[0].detach()))
    thumb(image[..., :180, :])
    handle.remove()
    assert seen[0].shape[-2:] == (180, 320)
    assert torch.equal(seen[0], image[..., :180, :])
    checks["native_no_input_resize_padding_or_crop"] = True
    probe = image[..., :180, :].clone().requires_grad_()
    tokens = thumb(probe)
    (tokens * torch.randn_like(tokens)).sum().backward()
    assert torch.isfinite(probe.grad).all()
    assert all(edge.abs().sum() > 0 for edge in (
        probe.grad[..., 0, :], probe.grad[..., -1, :],
        probe.grad[..., :, 0], probe.grad[..., :, -1]))
    checks["all_four_native_image_borders_influence_tokens"] = True
    obsolete = dict(model_cfg={"image_height": 180, "image_width": 320, "marker_count": 100,
                               "pad_to_stride": True}, normalization=asdict(norm))
    try:
        validate_thumb_checkpoint(obsolete)
    except ValueError:
        checks["obsolete_padded_checkpoint_refused"] = True
    else:
        raise AssertionError("Obsolete padded contract accepted")
    wrong = dict(inputs, rgb=torch.rand(2, 3, 240, 320))
    try:
        thumb(wrong["rgb"])
    except ValueError:
        checks["native_rejects_wrong_240_input"] = True
    else:
        raise AssertionError("Wrong input accepted")
    assert all(checks.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"passed": True, "checks": checks,
                                     "scope": "CPU wiring, not physical alignment or convergence"}, indent=2) + "\n")
    print(json.dumps({"passed": True, "checks": len(checks), "output": str(args.output)}))


if __name__ == "__main__":
    main()
