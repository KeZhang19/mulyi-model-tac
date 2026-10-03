#!/usr/bin/env python3
"""Two real Gloo workers: global promotion, bank provenance and resume state."""
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace as NS

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/RotateBulbAssemblyCurriculum"
core = runpy.run_path(str(CONFIG / "core.py"))
distributed = runpy.run_path(str(CONFIG / "distributed.py"))


class Runtime:
    def __init__(self, rank):
        self.env = NS(cfg=NS(seed=42 + rank))
        self.contract = {"schema": 2, "seed": 42 + rank, "physics": "same"}
        self.cfg = core["CurriculumSettings"](min_episodes=2)
        self.scheduler = core["StageScheduler"](self.cfg)
        self.bank = core["StateBank"](8, self.contract)
        self.rng = torch.Generator().manual_seed(42 + rank)
        self.episodes, self.transitions = 10 + rank, 100 + rank

    def state_dict(self):
        return {"schema": 1, "settings": core["settings_dict"](self.cfg),
                "scheduler": self.scheduler.state_dict(), "bank": self.bank.state_dict(),
                "rng": self.rng.get_state(), "episodes": self.episodes, "transitions": self.transitions}


def worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2)
    try:
        runtime = Runtime(rank)
        sync = distributed["DistributedCurriculum"](runtime)
        for _ in range(2):
            runtime.scheduler.observe(0, 0, True)
        runtime.bank.add(2, {"q": torch.tensor([rank], dtype=torch.float32)},
                         {"source_stage": 1, "physical_success": True, "seed": 42 + rank})
        assert not runtime.scheduler.ready and not runtime.bank.samples[2]
        saved = sync.synchronize()
        assert runtime.scheduler.ready
        assert len(runtime.bank.samples[2]) == 2
        assert [int(x["state"]["q"][0]) for x in runtime.bank.samples[2]] == [0, 1]
        assert [x["provenance"]["rank"] for x in runtime.bank.samples[2]] == [0, 1]
        assert saved["episodes"] == 21 and saved["transitions"] == 201
        for selected in (0, 1):
            own = distributed["restore_worker_state"](saved, seed=42 + selected, world_size=2, rank=selected)
            assert own["episodes"] == 10 + selected and own["transitions"] == 100 + selected
            assert own["bank"]["contract"]["seed"] == 42 + selected
            expected_rng = torch.Generator().manual_seed(42 + selected).get_state()
            assert torch.equal(own["rng"], expected_rng)
        try:
            distributed["restore_worker_state"](saved, seed=42, world_size=1, rank=0)
        except ValueError:
            pass
        else:
            raise AssertionError("Changed worker count was silently accepted")
        # The evaluator may change sampling seed without mutating saved state.
        assert distributed["restore_worker_state"](saved, seed=401, world_size=1, rank=0, evaluation=True) is saved
        sync.synchronize()
        assert len(runtime.bank.samples[2]) == 2  # no duplicate prefix on an empty boundary
        # Exercise the runner's all-rank early stop. Only rank zero calls log
        # and periodic save, matching RSL-RL; no collective may live there.
        module_name = "BrainCo_DexHand.tasks.manager_based.dexsuite.config.RotateBulbAssemblyCurriculum.distributed"
        sys.modules[module_name] = NS(DistributedCurriculum=distributed["DistributedCurriculum"])
        integration = runpy.run_path(str(ROOT / "scripts/rsl_rl/bulb_curriculum_training.py"))
        rt = Runtime(rank)
        class Runner:
            is_distributed = True
            gpu_global_rank = rank
            current_learning_iteration = 0
            writer = None
            log_dir = str(Path(rendezvous).parent)

            def __init__(self):
                self.env = NS(set_iteration=lambda iteration: None)
                self.alg = NS(update=lambda: {})

            def save(self, path, infos=None):
                torch.save({"infos": infos}, path)

            def log(self, locs):
                pass

            def learn(self, num_learning_iterations, init_at_random_ep_len=False):
                for iteration in range(num_learning_iterations):
                    self.env.set_iteration(iteration)
                    for _ in range(2):
                        rt.scheduler.observe(0, 0, True)
                    self.alg.update()
                    self.current_learning_iteration = iteration
                    if rank == 0:
                        self.log({"it": iteration})
                raise AssertionError("Global promotion failed to stop all workers")
        runner = Runner()
        integration["install_curriculum_runner"](runner, NS(_bulb_curriculum=rt))
        runner.learn(4)
        dist.barrier()
        checkpoint = torch.load(Path(rendezvous).parent / "model_0.pt", weights_only=False)
        assert checkpoint["infos"]["bulb_curriculum"]["scheduler"]["ready"]
        assert len(checkpoint["infos"]["bulb_curriculum"]["distributed"]["ranks"]) == 2
        # Different physical contracts must be rejected by every worker.
        mismatched = Runtime(rank)
        mismatched.contract["physics"] = str(rank)
        try:
            distributed["DistributedCurriculum"](mismatched)
        except ValueError:
            pass
        else:
            raise AssertionError("Mismatched physics accepted")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(worker, args=(str(Path(directory) / "rendezvous"),), nprocs=2, join=True)
    print("Two-rank promotion, prefix merge, RNG/counters, resume guards and no-duplicate checks passed")
