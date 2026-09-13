#!/usr/bin/env python3
"""Instantiate both locked USD assets through Isaac Lab's RigidObject API."""

from __future__ import annotations

from pathlib import Path

from isaaclab.app import AppLauncher


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def main():
    simulation_app = AppLauncher(headless=True).app
    try:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObject, RigidObjectCfg

        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device="cpu"))
        specifications = (
            (
                "lightbulb_with_socket",
                PACKAGE_ROOT / "lightbulb_with_socket" / "lightbulb_with_socket.usd",
                "/World/LightbulbWithSocket",
                (0.0, 0.0, 0.5),
            ),
            (
                "hiveboard_lamp",
                PACKAGE_ROOT / "hiveboard_lamp" / "hiveboard_lamp_locked.usd",
                "/World/HiveboardLamp",
                (0.4, 0.0, 0.5),
            ),
        )

        objects = []
        for name, usd_path, prim_path, position in specifications:
            print(f"creating {name}", flush=True)
            cfg = RigidObjectCfg(
                prim_path=prim_path,
                spawn=sim_utils.UsdFileCfg(usd_path=str(usd_path)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=position),
            )
            objects.append((name, RigidObject(cfg)))

        print("resetting simulation", flush=True)
        sim.reset()
        print("simulation reset complete", flush=True)
        for _, obj in objects:
            obj.reset()
        for _ in range(2):
            for _, obj in objects:
                obj.write_data_to_sim()
            sim.step()
            for _, obj in objects:
                obj.update(sim.get_physics_dt())

        for name, obj in objects:
            if obj.num_instances != 1:
                raise AssertionError(f"{name}: expected one instance, got {obj.num_instances}")
            print(f"spawned {name}: body_names={obj.body_names}", flush=True)
    finally:
        # This is a disposable smoke process with no render products to flush.
        # Immediate shutdown also avoids slow renderer/PhysX teardown on some
        # headless Isaac Sim installations.
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    main()
