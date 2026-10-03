#!/usr/bin/env python3
"""Open/replay a generated calibration stage in Isaac Sim; bounded by default.

This viewer does not run a rigid contact solver or produce force ground truth.
"""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path)
    p.add_argument('--frames',type=int,default=120,help='Bounded viewer updates')
    p.add_argument('--capture',action='store_true',help='Save a native RTX image at peak model load')
    # Parse help before importing the application launcher.
    if '--help' in __import__('sys').argv:
        p.print_help();print('Also accepts Isaac AppLauncher --headless and --device.');return
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(p)
    args=p.parse_args()
    if args.frames<1:
        p.error('--frames must be positive')
    if args.capture:
        args.enable_cameras=True
    report_path=args.folder.resolve()/'isaac_verification.json'
    if report_path.exists():
        p.error('Use a new run directory; native verification output already exists')
    app=AppLauncher(args).app
    try:
        import numpy as np
        import omni.usd
        import omni.timeline
        from pxr import Usd, UsdGeom
        folder=args.folder.resolve()
        if not omni.usd.get_context().open_stage(str(folder/'scene.usda')):
            raise RuntimeError('Could not open generated stage')
        for _ in range(20):app.update()
        stage=omni.usd.get_context().get_stage()
        meshes=[UsdGeom.Mesh(x) for x in stage.Traverse() if x.IsA(UsdGeom.Mesh)]
        topology=[]
        for mesh in meshes:
            result=UsdGeom.Mesh.ValidateTopology(mesh.GetFaceVertexIndicesAttr().Get(),mesh.GetFaceVertexCountsAttr().Get(),len(mesh.GetPointsAttr().Get()))
            if not result[0]:topology.append(str(mesh.GetPath()))
        with np.load(folder/'episode_00000.npz',allow_pickle=False) as a:
            matrices=a['T_world_flange'];peak=int(np.argmax(a['total_normal_load_n']))
        tool=stage.GetPrimAtPath('/World/Tool')
        max_error=0.
        for i,matrix in enumerate(matrices):
            actual=np.asarray(UsdGeom.XformCache(Usd.TimeCode(i)).GetLocalToWorldTransform(tool)).T
            max_error=max(max_error,float(np.abs(actual-matrix).max()))
        timeline=omni.timeline.get_timeline_interface()
        annotator=None
        if args.capture:
            import omni.replicator.core as rep
            product=rep.create.render_product('/World/Overview',(1280,800))
            annotator=rep.AnnotatorRegistry.get_annotator('rgb');annotator.attach(product)
        for i in range(args.frames):
            timeline.set_current_time((peak if args.capture else i%len(matrices))/stage.GetTimeCodesPerSecond())
            app.update()
        capture_ok=False
        if annotator is not None:
            data=annotator.get_data()
            if data is None or not data.size:
                raise RuntimeError('Renderer returned no RGB')
            from PIL import Image
            Image.fromarray(data[...,:3]).save(folder/'isaac_scene.png');capture_ok=True
        report=dict(stage_opened=True,meshes=len(meshes),invalid_mesh_topology=topology,
                    animation_pose_max_abs_error=max_error,viewer_frames=args.frames,capture=capture_ok,
                    dynamics_validated=False,passed=len(meshes)>50 and not topology and max_error<1e-7)
        (folder/'isaac_verification.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
        if not report['passed']:raise RuntimeError('Native stage verification failed')
    except BaseException as error:
        report_path.write_text(json.dumps(dict(passed=False,error=f'{type(error).__name__}: {error}'),indent=2)+'\n')
        raise
    finally:
        app.close()


if __name__=='__main__':main()
