#!/usr/bin/env python3
"""Visual Tag-frame calibration collector; offline mode never starts Isaac or ROS."""
import argparse
import json
from pathlib import Path
import time

from stage_config import COORDINATES, ROS_WORKSPACE, StageCoordinates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'config/simulation/visual_config.json')
    p.add_argument('--coordinates', type=Path, default=COORDINATES)
    p.add_argument('--ros-workspace', type=Path, default=ROS_WORKSPACE)
    p.add_argument('--output', type=Path, help='New session directory; existing directories refused')
    p.add_argument('--plan-only', action='store_true', help='Offline config/coordinate summary, no output')
    p.add_argument('--offline', action='store_true', help='Finite CPU capture without Isaac')
    p.add_argument('--episodes', type=int, default=1, help='Finite offline/autostart prefix')
    p.add_argument('--head', choices=('spherical', 'triangle', 'conical'))
    p.add_argument('--finger', choices=('index', 'middle', 'ring', 'little', 'thumb'))
    p.add_argument('--segment', choices=('DIP', 'MCP', 'PIP'))
    p.add_argument('--autostart', action='store_true', help='Collect finite prefix on viewer startup')
    p.add_argument('--viewer-frames', type=int, default=0, help='0 = interactive; positive = bounded native check')
    p.add_argument('--headless', action='store_true')
    p.add_argument('--capture', action='store_true', help='Save native RTX image after a bounded run')
    args = p.parse_args()
    if args.episodes < 1 or args.viewer_frames < 0:
        p.error('episodes must be positive; viewer-frames must be nonnegative')
    if args.headless and not args.offline and args.viewer_frames == 0:
        p.error('headless native runs require --viewer-frames')
    coordinates = StageCoordinates(args.coordinates, args.ros_workspace)
    config = json.loads(args.config.read_text())
    for key in ('head', 'finger', 'segment'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    from protocol import episode_plans, validate_config
    validate_config(config)
    if set(config['protocols']) & {'twist', 'roll'}:
        p.error('Only XYZ protocols are supported; calibrated turntable transforms are missing')
    if args.plan_only:
        print(json.dumps(dict(coordinates=coordinates.snapshot(),
                              episode_count=len(episode_plans(config)), head=config['head'],
                              finger=config['finger'], segment=config['segment'],
                              simulation='prescribed_motion; no physical calibration'), indent=2, ensure_ascii=False))
        return 0
    if args.output is None:
        p.error('--output is required; use a new directory')
    if args.output.exists():
        p.error('Output directory exists; choose a new session path')
    app = None
    session = None
    error = None
    try:
        if not args.offline:
            from isaaclab.app import AppLauncher
            app = AppLauncher(headless=args.headless, enable_cameras=args.capture).app
        import torch
        torch.set_num_threads(1)
        from visual_session import VisualSession, finite_capture
        from visual_scene import export_visual_scene
        session = VisualSession(config, coordinates, args.output)
        from geometry import fingerprint
        session.manifest['input_config_source'] = fingerprint({args.config.resolve()})
        session.persist()
        if args.offline:
            finite_capture(session, args.episodes)
            arrays = session.last_arrays
            export_visual_scene(session.output / 'scene.usda', session, arrays['T_tag_flange'])
            from visual_scene import channel_panel
            from PIL import Image
            peak = int(arrays['total_normal_load_n'].argmax())
            session.replay(peak)
            panel, trace = channel_panel(session.data, session.force_history,
                                          config['max_depth_m'], config['pressure_saturation_pa'])
            Image.fromarray(panel).save(session.output / 'channels.png')
            Image.fromarray(trace).save(session.output / 'force_trace.png')
        else:
            run_viewer(app, session, args)
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if session:
            session.close(error)
        if app:
            app.close()
    print(json.dumps(dict(output=str(session.output), complete=session.manifest['complete'],
                          episodes=session.reports), indent=2))
    return 0 if session.manifest['complete'] or not session.reports else 2


def run_viewer(app, session, args):
    import numpy as np
    import omni.usd
    import omni.ui as ui
    from pxr import Gf, UsdGeom
    from visual_scene import export_visual_scene
    from visual_ui import CollectorPanel
    from protocol import episode_plans
    stage_path = session.output / 'scene.usda'
    export_visual_scene(stage_path, session)
    if not omni.usd.get_context().open_stage(str(stage_path)):
        raise RuntimeError('Could not open calibration scene')
    for _ in range(20):
        app.update()
    stage = omni.usd.get_context().get_stage()
    tool = UsdGeom.Xform(stage.GetPrimAtPath('/World/Tool'))
    attr = tool.GetPrim().GetAttribute('xformOp:transform')
    attr.Clear()
    panel = CollectorPanel(session)
    viewport_window = ui.Workspace.get_window('Viewport')
    if viewport_window:
        panel.window.dock_in(viewport_window, ui.DockPosition.RIGHT, .4)
    from omni.kit.viewport.utility import get_active_viewport
    viewport = get_active_viewport()
    viewport.camera_path = '/World/Overview'
    plans = list(episode_plans(session.config)[:args.episodes]) if args.autostart else []
    annotator = None
    if args.capture:
        import omni.replicator.core as rep
        product = rep.create.render_product('/World/Overview', (1280, 800))
        annotator = rep.AnnotatorRegistry.get_annotator('rgb')
        annotator.attach(product)
    frame = 0
    next_step = time.monotonic()
    pose_error = 0.
    while app.is_running() and (args.viewer_frames == 0 or frame < args.viewer_frames):
        if plans and not session.queue and not session.recording:
            if session.reports and not session.reports[-1]['completed']:
                plans.clear()
            else:
                session.start(plans.pop(0))
        if time.monotonic() >= next_step or args.headless:
            session.step()
            next_step = time.monotonic() + 1 / session.config['sample_hz']
            panel.update()
        matrix = session.data['T_tag_flange'] if 'T_tag_flange' in session.data else session.data['T_world_flange']
        attr.Set(Gf.Matrix4d(*matrix.T.ravel().tolist()))
        pose_error = max(pose_error, float(np.abs(np.asarray(tool.GetLocalTransformation()).T - matrix).max()))
        app.update()
        frame += 1
    # Paused/closed or bounded mid-cycle runs keep their samples and failure reason.
    if session.recording:
        session.stop('viewer_frame_budget' if args.viewer_frames else 'application_closed')
    if annotator:
        if session.last_arrays and len(session.last_arrays['frame_id']):
            session.replay(int(session.last_arrays['total_normal_load_n'].argmax()))
            attr.Set(Gf.Matrix4d(*session.data['T_world_flange'].T.ravel().tolist()))
            panel.update()
        for _ in range(30):
            app.update()
        data = annotator.get_data()
        if data is None or not data.size:
            raise RuntimeError('Native renderer returned no image')
        from PIL import Image
        Image.fromarray(data[..., :3]).save(session.output / 'isaac_scene.png')
        panel.save_images(session.output)
    topology = [str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)
                and not UsdGeom.Mesh.ValidateTopology(UsdGeom.Mesh(prim).GetFaceVertexIndicesAttr().Get(),
                    UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get(),
                    len(UsdGeom.Mesh(prim).GetPointsAttr().Get()))[0]]
    from visual_session import write_json
    report = dict(stage_opened=True, frame='tag_center', viewer_frames=frame,
                  invalid_mesh_topology=topology, pose_max_abs_error=pose_error,
                  panel_created=True, panel_updates=panel.updates,
                  native_capture=(annotator is not None), dynamics_validated=False,
                  passed=not topology and pose_error < 1e-7 and panel.updates > 0)
    write_json(session.output / 'isaac_verification.json', report)
    if not report['passed']:
        raise RuntimeError('Native scene verification failed')


if __name__ == '__main__':
    raise SystemExit(main())
