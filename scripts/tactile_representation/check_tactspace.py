#!/usr/bin/env python3
"""CPU contract tests, including bounded synthetic runs through the real CLIs.

Synthetic losses test wiring only, never representation quality or sim/real transfer.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/BrainCo_DexHand"))
sys.path.insert(0, str(ROOT / "scripts/tactile_representation"))
from BrainCo_DexHand.tactile_representation.models.tactspace import (
    TactSpaceCfg, TactSpaceLossCfg, TactSpaceRGBDepthMarker, batch_inputs,
    compute_tactspace_objective, representation_contract, validate_sim_training_manifest,
)
from BrainCo_DexHand.tactile_representation.policy import (
    FrozenTactilePolicyEncoder, build_tactile_encoder, checkpoint_model_type,
    load_tactile_checkpoint, make_tactspace_policy_bundle, load_sim_policy_encoder,
)
from BrainCo_DexHand.tactile_representation.config import CrossModalTactileNetworkCfg
from BrainCo_DexHand.tactile_representation.tri_modal_config import TriModalCrossAutoencoderCfg
from BrainCo_DexHand.tactile_representation.training import TactileNormalization
from BrainCo_DexHand.tactile_representation.training.contrastive import build_positive_mask, symmetric_masked_infonce_loss
import train_tri_modal_cross_autoencoder as trainer
from train_latent_alignment import _build_encoder


def small_cfg():
    return TactSpaceCfg(image_height=16, image_width=16, marker_count=4, d_model=16,
                       num_heads=4, image_base_channels=2, decoder_base_channels=16,
                       marker_transformer_layers=1, dropout=0)


def batch(size=3):
    marker = torch.rand(size, 4, 5)
    marker[..., 4] = 1
    return {"rgb": torch.rand(size, 3, 16, 16), "depth": torch.rand(size, 1, 16, 16),
            "marker": marker, "marker_valid": torch.ones(size, 4, dtype=torch.bool),
            "episode_id": torch.arange(size), "episode_step": torch.zeros(size, dtype=torch.long)}


def source(model):
    return {"network_type": "tactspace", "model_cfg": asdict(model.cfg),
            "model_state": model.state_dict(), "representation_contract": representation_contract(model.cfg),
            "normalization": asdict(TactileNormalization(16, 16, 4))}


def update(model, data, cfg=TactSpaceLossCfg()):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss, metrics = compute_tactspace_objective(model, data, cfg=cfg)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    model.record_trained_sources(metrics)
    return loss, metrics


def synthetic_dataset(root):
    rng = np.random.default_rng(7)
    count = 36
    arrays = {"rgb": rng.integers(0, 256, (count, 3, 16, 16), dtype=np.uint8),
              "depth_m": rng.random((count, 1, 16, 16), dtype=np.float32) * .003,
              "marker_2d": rng.random((count, 4, 5), dtype=np.float32),
              "marker_valid": np.ones((count, 4), dtype=bool),
              "episode_id": np.repeat(np.arange(9), 4), "episode_step": np.tile(np.arange(4), 9),
              "contact": np.tile(np.array([False, True, True, True]), 9)}
    arrays["marker_2d"][..., 4] = 1
    for key in ("env_id", "phase", "target_force_n", "presser_id", "sweep_stage"):
        arrays[key] = np.zeros(count, dtype=np.int64)
    plans = [dict(episode_id=i, sweep_stage="press", target_force_n=i + 1,
                  offset_u_m=0., offset_v_m=0., tilt_axis="x", tilt_deg=0.,
                  slide_axis="x", slide_distance_m=0., baseline_steps=1,
                  press_steps=1, slide_steps=1, hold_steps=1) for i in range(9)]
    manifest = dict(status="complete", sample_count=count, shards=[dict(file="samples.npz", sample_count=count)],
                    schema={k: dict(dtype=str(v.dtype), sample_shape=list(v.shape[1:])) for k, v in arrays.items()},
                    metadata=dict(episode_plans=plans, domain="sim", synthetic=True))
    root.mkdir()
    np.savez(root / "samples.npz", **arrays)
    (root / "manifest.json").write_text(json.dumps(manifest))


class TactSpaceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = TactSpaceRGBDepthMarker(small_cfg())
        self.data = batch()

    def test_default_and_known_legacy_models(self):
        args = trainer.parse_args([])
        self.assertEqual(args.model_type, "tri_modal")
        for kind, cfg in (("tri_modal", TriModalCrossAutoencoderCfg(**{k:v for k,v in asdict(small_cfg()).items()
                                                                         if k not in ("domains", "input_modalities")})),
                          ("robust", CrossModalTactileNetworkCfg(image_height=16, image_width=16, marker_count=4,
                              d_model=16, num_heads=4, image_base_channels=2, decoder_base_channels=16, marker_transformer_layers=1))):
            original = build_tactile_encoder(kind, asdict(cfg)).eval()
            old = {"model_cfg": asdict(cfg), "model_state": original.state_dict()}
            self.assertEqual(checkpoint_model_type(old), kind)
            loaded = _build_encoder(None, old)
            inputs = {k:v for k,v in self.data.items() if k in ("rgb", "depth", "marker")}
            torch.testing.assert_close(original.encode(**inputs), loaded.encode(**inputs), rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "requested"):
                _build_encoder("tactspace", old)
        with self.assertRaisesRegex(ValueError, "Unrecognized"):
            checkpoint_model_type({"model_cfg": {}, "model_state": {}})
        with self.assertRaisesRegex(ValueError, "Unknown checkpoint"):
            checkpoint_model_type({"network_type": "unrelated"})

    def test_single_multi_forward_and_mean(self):
        self.model.eval()
        inputs = batch_inputs(self.data)
        latents, valid = self.model.encode_modalities(**inputs)
        expected = torch.nn.functional.normalize(torch.stack(list(latents.values())).mean(0), dim=-1)
        torch.testing.assert_close(self.model.encode(**inputs), expected)
        for name in ("rgb", "depth", "marker"):
            output = self.model(**{name:self.data[name]})
            self.assertEqual(output["latent"].shape, (3, 16))
            torch.testing.assert_close(output["latent"], latents[name])
            self.assertEqual(output["rgb_recon"].shape, (3, 3, 16, 16))
            self.assertEqual(output["depth_recon"].shape, (3, 1, 16, 16))
            self.assertEqual(output["marker_recon"].shape, (3, 4, 2))

    def test_joint_gradients_and_updates(self):
        tracked = [self.model.encoders["sim"]["rgb"].stem[0].weight,
                   self.model.encoders["sim"]["depth"].stem[0].weight,
                   self.model.encoders["sim"]["marker"].motion_mlp[0].weight,
                   self.model.decoder_trunk[0].weight, self.model.image_head.output_head.weight,
                   self.model.marker_head.output_head[-1].weight]
        before = [p.detach().clone() for p in tracked]
        loss, metrics = update(self.model, self.data)
        self.assertTrue(torch.isfinite(loss))
        for key in ("loss/self_reconstruction", "loss/cross_reconstruction", "loss/alignment"):
            self.assertTrue(torch.isfinite(metrics[key]))
        self.assertEqual(int(metrics["count/valid_alignment_pairs"]), 9)
        for p, old in zip(tracked, before):
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(float(p.grad.abs().sum()), 0)
            self.assertFalse(torch.equal(p, old))

    def test_target_leakage(self):
        self.model.eval()
        original = self.model(**batch_inputs(self.data))["source_reconstructions"]["rgb"]
        changed = dict(self.data, depth=torch.randn_like(self.data["depth"]), marker=torch.randn_like(self.data["marker"]))
        outputs = [self.model(**batch_inputs(changed))["source_reconstructions"]["rgb"],
                   self.model(rgb=self.data["rgb"])["source_reconstructions"]["rgb"]]
        for output in outputs:
            for key in original:
                torch.testing.assert_close(original[key], output[key], rtol=0, atol=0)

    def test_missing_invalid_modalities_points_and_rows(self):
        self.model.eval()
        data = self.data
        data["marker_valid"][:, 1] = False
        before = compute_tactspace_objective(self.model, data, cfg=TactSpaceLossCfg())[0]
        data["marker"][:, 1] = float("nan")
        after = compute_tactspace_objective(self.model, data, cfg=TactSpaceLossCfg())[0]
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        after.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters()))
        data.pop("depth")
        _, metrics = compute_tactspace_objective(self.model, data, cfg=TactSpaceLossCfg())
        self.assertEqual(int(metrics["count/reconstruction_pairs"]), 4)
        data["marker_valid"].fill_(False)
        torch.testing.assert_close(self.model.encode(**batch_inputs(data)), self.model.encode(rgb=data["rgb"]))
        with self.assertRaisesRegex(ValueError, "No valid"):
            self.model.encode(marker=data["marker"], marker_valid_mask=data["marker_valid"])
        with self.assertRaisesRegex(ValueError, "No enabled"):
            self.model.encode()
        data["training_ready"] = torch.zeros(3, dtype=torch.bool)
        loss, metrics = compute_tactspace_objective(self.model, data, cfg=TactSpaceLossCfg())
        self.assertEqual(float(loss), 0)
        self.assertEqual(int(metrics["count/active_sources"]), 0)

    def test_pair_skips_and_multiple_positives(self):
        self.model.eval()
        for data in (batch(1), dict(self.data, pair_valid=torch.zeros(3, dtype=torch.bool)),
                     {k:v for k,v in self.data.items() if k not in ("episode_id", "episode_step")},
                     dict(self.data, episode_id=torch.zeros(3, dtype=torch.long))):
            loss, metrics = compute_tactspace_objective(self.model, data, cfg=TactSpaceLossCfg())
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(float(metrics["loss/alignment"]), 0)
            self.assertEqual(int(metrics["count/skipped_alignment_pairs"]), 3)
        self.data["episode_id"] = torch.tensor([7, 7, 8])
        z, _ = self.model.encode_modalities(**batch_inputs(self.data))
        positive = build_positive_mask(self.data["episode_id"], self.data["episode_id"])
        expected = torch.stack([symmetric_masked_infonce_loss(z[a], z[b], positive) for a,b in
                                (("rgb", "depth"), ("rgb", "marker"), ("depth", "marker"))]).mean()
        _, metrics = compute_tactspace_objective(self.model, self.data, cfg=TactSpaceLossCfg())
        torch.testing.assert_close(metrics["loss/alignment"], expected)
        self.assertEqual(int(positive.sum()), 5)

    def test_ablation_weights_and_domain_branches(self):
        for rec, align in ((1,0), (0,1), (0,0)):
            model = TactSpaceRGBDepthMarker(small_cfg())
            loss, metrics = compute_tactspace_objective(model, self.data, cfg=TactSpaceLossCfg(rec, align))
            loss.backward()
            if not rec:
                self.assertTrue(all(p.grad is None for p in model.decoder_trunk.parameters()))
            if rec == align == 0:
                self.assertEqual(float(loss), 0)
                self.assertEqual(float(metrics["count/active_sources"]), 0)
        both = TactSpaceRGBDepthMarker(replace(small_cfg(), domains=("sim", "real")))
        self.assertIsNot(both.encoders["sim"]["rgb"], both.encoders["real"]["rgb"])
        update(both, self.data)
        with self.assertRaisesRegex(ValueError, "untrained real"):
            make_tactspace_policy_bundle(source(both), domain="real", source_checkpoint_sha256="a" * 64)

    def test_saved_exported_frozen_features(self):
        update(self.model, self.data)
        self.model.eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "joint.pt"
            torch.save(source(self.model), path)
            restored = _build_encoder(None, load_tactile_checkpoint(path))
            torch.testing.assert_close(self.model.encode(**batch_inputs(self.data)), restored.encode(**batch_inputs(self.data)), rtol=0, atol=0)
            frozen = load_sim_policy_encoder(path, chunk_size=2)
            inputs = {"rgb": (self.data["rgb"][:1] * 255).repeat(5,1,1,1).unsqueeze(0),
                      "depth_m": (self.data["depth"][:1] * .003).repeat(5,1,1,1).unsqueeze(0),
                      "marker": self.data["marker"][:1].repeat(5,1,1).unsqueeze(0).clone()}
            inputs["marker"][..., :2] *= 15
            inputs["marker"][..., 2:4] *= 5
            for names in (("rgb",), ("depth_m",), ("marker",), tuple(inputs)):
                selected = {k:inputs[k] for k in names}
                normalized = {"depth" if k == "depth_m" else k:self.data["depth" if k == "depth_m" else k][:1] for k in names}
                expected = self.model.encode(**normalized).expand(5,-1)[None]
                torch.testing.assert_close(frozen(**selected), expected, rtol=2e-5, atol=2e-6)
            valid = torch.ones(1, 5, 3, dtype=torch.bool)
            valid[..., 1:] = False
            torch.testing.assert_close(frozen(**inputs, modality_valid=valid), frozen(rgb=inputs["rgb"]))
            with self.assertRaisesRegex(ValueError, "No valid"):
                frozen(**inputs, modality_valid=torch.zeros_like(valid))
            self.assertFalse(frozen.training)
            self.assertTrue(all(not p.requires_grad for p in frozen.parameters()))
            contract = frozen.observation_contract(history_length=4)
            self.assertEqual(contract["observation_dim"], (152 + 5 * 16) * 4)
            self.assertEqual(contract["representation_contract"], representation_contract(small_cfg()))

    def test_equal_modality_weighting_and_missing_target(self):
        model = TactSpaceRGBDepthMarker(replace(small_cfg(), input_modalities=("rgb",)))
        data = batch()
        data["rgb"].fill_(1)
        data["depth"].fill_(2)
        data["marker"][..., 2:4] = 3
        model.decode = lambda latent: {
            "rgb_recon": torch.zeros_like(data["rgb"]),
            "depth_recon": torch.zeros_like(data["depth"]),
            "marker_recon": torch.zeros_like(data["marker"][..., 2:4]),
        }
        loss, _ = compute_tactspace_objective(model, data, cfg=TactSpaceLossCfg(alignment_weight=0))
        self.assertAlmostEqual(float(loss), 14 / 3, places=5)
        data.pop("depth")
        model.decode = lambda latent: {
            "rgb_recon": torch.zeros_like(data["rgb"]),
            "marker_recon": torch.zeros_like(data["marker"][..., 2:4]),
        }
        loss, _ = compute_tactspace_objective(model, data, cfg=TactSpaceLossCfg(alignment_weight=0))
        self.assertAlmostEqual(float(loss), 5, places=5)

    def test_empty_updates_do_not_change_or_certify_weights(self):
        args = trainer.parse_args(["--model-type", "tactspace", "--grad-accum-steps", "1"])
        before = {k:v.clone() for k,v in self.model.state_dict().items()}
        data = dict(self.data, training_ready=torch.zeros(3, dtype=torch.bool))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1.)
        metrics, steps = trainer.train_epoch(self.model, [data], optimizer=optimizer,
            scaler=trainer._make_scaler(False), device=torch.device("cpu"),
            normalization=TactileNormalization(16,16,4), objective_cfg=TactSpaceLossCfg(),
            args=args, amp_enabled=False, seed=7, epoch=0)
        self.assertEqual(steps, 0)
        self.assertEqual(metrics["count/skipped_empty_updates"], 1)
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "untrained sim"):
            make_tactspace_policy_bundle(source(self.model), source_checkpoint_sha256="a" * 64)

    def test_data_gate(self):
        for manifest in ({}, {"training_ready":False}, {"metadata":{"domain":"real"}},
                         {"schema":{"pair_valid":{}}}, {"schema":{"training_ready":{}}}):
            with self.assertRaises(ValueError):
                validate_sim_training_manifest(manifest)

    def test_real_cli_smoke_resume_export_and_conflicts(self):
        with tempfile.TemporaryDirectory(prefix="tactspace_test_") as tmp:
            root = Path(tmp)
            synthetic_dataset(root / "data")
            def run(script, *args, ok=True):
                result = subprocess.run([sys.executable, str(ROOT / "scripts/tactile_representation" / script), *map(str,args)],
                                        cwd=ROOT, capture_output=True, text=True, timeout=90)
                if ok:
                    self.assertEqual(result.returncode, 0, result.stdout[-2000:] + result.stderr[-3000:])
                else:
                    self.assertNotEqual(result.returncode, 0)
                return result
            common = ["--dataset",root/"data", "--device","cpu", "--epochs","1", "--batch-size","3", "--grad-accum-steps","1",
                      "--num-workers","0", "--max-train-batches","2", "--max-val-batches","1", "--max-test-batches","1",
                      "--d-model","16", "--num-heads","4", "--image-base-channels","2", "--decoder-base-channels","16",
                      "--marker-transformer-layers","1", "--dropout","0", "--no-amp", "--no-balance-contact"]
            script = "train_tri_modal_cross_autoencoder.py"
            run(script, *common, "--output",root/"old")  # no model selector
            self.assertEqual(checkpoint_model_type(load_tactile_checkpoint(root/"old/last.pt")), "tri_modal")
            run(script, *common, "--model-type","tactspace", "--output",root/"new")
            payload = load_tactile_checkpoint(root/"new/last.pt")
            self.assertEqual(payload["global_step"], 2)
            self.assertEqual(payload["model_cfg"]["d_model"], 16)
            self.assertTrue(bool((payload["model_state"]["trained_source_steps"] > 0).all()))
            history = json.loads((root/"new/history.jsonl").read_text().splitlines()[0])
            self.assertGreater(history["train"]["count/valid_alignment_pairs"], 0)
            run(script, "--dataset",root/"data", "--resume",root/"new/last.pt", "--output",root/"new",
                "--device","cpu", "--epochs","2", "--batch-size","3", "--grad-accum-steps","1", "--num-workers","0",
                "--max-train-batches","1", "--max-val-batches","1", "--max-test-batches","1")
            self.assertEqual(load_tactile_checkpoint(root/"new/last.pt")["global_step"], 3)
            error = run(script, "--dataset",root/"data", "--resume",root/"new/last.pt", "--d-model","32", ok=False)
            self.assertIn("conflicts with checkpoint", error.stderr)
            run("export_tactile_encoders.py", "--sim-checkpoint",root/"new/last.pt", "--unaligned-sim", "--output-dir",root/"exports")
            exported = root/"exports/sim_policy_encoder_tactspace.pt"
            frozen = FrozenTactilePolicyEncoder(exported)
            direct = load_sim_policy_encoder(root/"new/last.pt")
            raw = torch.rand(2, 5, 3, 16, 16) * 255
            torch.testing.assert_close(frozen(rgb=raw), direct(rgb=raw), rtol=0, atol=0)
            no_overwrite = run("export_tactile_encoders.py", "--sim-checkpoint",root/"new/last.pt",
                               "--unaligned-sim", "--output-dir",root/"exports", ok=False)
            self.assertIn("Refusing to overwrite", no_overwrite.stderr)
            run(script, *common, "--model-type","tactspace", "--input-modalities","marker", "--output",root/"single")
            single = load_tactile_checkpoint(root/"single/last.pt")
            self.assertEqual(tuple(single["model_cfg"]["input_modalities"]), ("marker",))
            single_frozen = load_sim_policy_encoder(root/"single/last.pt")
            markers = torch.rand(1,5,4,5)
            markers[...,4] = 1
            self.assertEqual(single_frozen(marker=markers).shape, (1,5,16))
            error = run("export_tactile_encoders.py", "--sim-checkpoint",root/"new/last.pt", "--sim-model-type","robust",
                        "--unaligned-sim", "--output-dir",root/"bad", ok=False)
            self.assertIn("requested", error.stderr)
            error = run("train_latent_alignment.py", "--sim-checkpoint",root/"new/last.pt", "--real-checkpoint",root/"new/last.pt",
                        "--device","cpu", ok=False)
            self.assertIn("cannot certify physical", error.stderr)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
