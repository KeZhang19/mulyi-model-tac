#!/usr/bin/env python3
"""Capture per-camera native RGB/depth/marker with persistent unloaded references.

This is a source dataset for calibrated sim/real conversion, not a training manifest.
No motor commands, pressure reads, or simulated positive pairs are generated.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import uuid

HERE = Path(__file__).resolve().parent


def collect_real_trimodal():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('baseline', 'loaded'), required=True)
    parser.add_argument('--confirm-unloaded', action='store_true')
    parser.add_argument('--reference-session', type=Path)
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--serial', action='append', help='Repeat to select cameras; default all five')
    parser.add_argument('--finger-map', type=Path, help='Confirmed JSON mapping camera serial to finger name')
    args = parser.parse_args()
    if not 2 <= args.frames <= 60:
        parser.error('frames must be 2..60')
    if args.phase == 'baseline' and (not args.confirm_unloaded or args.reference_session):
        parser.error('baseline requires --confirm-unloaded and creates its own reference')
    if args.phase == 'loaded' and (args.reference_session is None or args.confirm_unloaded):
        parser.error('loaded requires --reference-session and cannot confirm unloaded')
    reference = None
    if args.reference_session:
        args.reference_session = args.reference_session.resolve()
        reference = json.loads((args.reference_session / 'session.json').read_text())
        if (reference['phase'] != 'baseline' or reference['reference_state'] != 'unloaded-confirmed'
                or reference['status'] != 'complete'):
            parser.error('reference must be a completed confirmed unloaded session')
    mapping = json.loads(args.finger_map.read_text()) if args.finger_map else {}
    fingers = {'little', 'ring', 'middle', 'index', 'thumb'}
    if not isinstance(mapping, dict) or any(v not in fingers for v in mapping.values()) or len(set(mapping.values())) != len(mapping):
        parser.error('finger-map must assign distinct valid finger names to serials')
    devices = sorted(Path('/dev/v4l/by-id').glob('*ViTai-GFBC*-video-index0'))
    cameras = {p.name.split('ViTai-GFBC_', 1)[1].rsplit('-video-index', 1)[0]: p for p in devices}
    selected = args.serial or list(cameras)
    if len(set(selected)) != len(selected) or not selected or any(s not in cameras for s in selected):
        parser.error('requested camera missing or duplicated')
    if not args.serial and len(selected) != 5:
        parser.error('expected five cameras; use --serial for an explicitly selected subset')
    if reference:
        for serial in selected:
            record = next((c for c in reference['cameras'] if c['serial'] == serial), None)
            if record is None:
                parser.error(f'reference missing serial {serial}')
            path = args.reference_session / 'raw' / serial / 'frames.npz'
            if hashlib.sha256(path.read_bytes()).hexdigest() != record['raw_sha256']:
                parser.error(f'reference content changed: {serial}')
            old_finger = record['physical_finger']
            if old_finger and serial in mapping and mapping[serial] != old_finger:
                parser.error(f'reference finger mapping conflict: {serial}')
            if old_finger:
                mapping.setdefault(serial, old_finger)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    raw = root / 'raw'
    raw.mkdir()
    session_id = str(uuid.uuid4())
    session = dict(schema='revo3_native_trimodal_source_v1', session_id=session_id,
                   status='collecting', phase=args.phase, reference_state='unloaded-confirmed',
                   reference_confirmation='user confirmed unloaded before baseline capture',
                   reference_session_id=reference['session_id'] if reference else session_id,
                   reference_session=str(args.reference_session or root),
                   started_utc=datetime.now(timezone.utc).isoformat(), cameras=[], sample_count=0,
                   training_ready=False, pair_valid=False, physical_calibration_validated=False,
                   sampling='sequential cameras; modalities derive from the same RGB frame',
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   conversion_blockers=['depth scale to metres unknown', 'physical finger/ROI mapping needs verification',
                                        '400 SDK flow points are not the encoder 100-point geometry',
                                        'no corresponding simulation states or canonical pair keys'],
                   marker_layout='row-major SDK grid; [x0,y0,dx,dy,finite]; finite is not tracking confidence',
                   units={'depth_sdk':'SDK native, uncalibrated', 'marker_2d':'SDK image pixels'},
                   paired_training_key='episode_id/episode_step assigned only after physical sim/real matching')
    def save():
        (root / 'session.json').write_text(json.dumps(session, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    save()
    try:
        raw_report = {'cameras': []}
        for serial in selected:
            cmd = [sys.executable, str(HERE/'probe_revo3_readonly.py'), '--output', str(raw),
                   '--worker', 'camera', '--camera', str(cameras[serial]), '--frames', str(args.frames)]
            with (raw/f'{serial}.log').open('w') as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=30, check=True)
            camera = json.loads((raw/serial/'camera.json').read_text())
            raw_report['cameras'].append(camera)
            (raw/'report.json').write_text(json.dumps(raw_report, indent=2)+'\n')
            if not camera['passed']:
                raise RuntimeError(f'camera capture failed: {serial}: {camera["errors"]}')
        cmd = [sys.executable, str(HERE/'probe_revo3_trimodal.py'), '--capture', str(raw),
               '--output', str(root/'derived'), '--reference-state', 'unloaded-confirmed',
               '--expected-cameras', str(len(selected))]
        if reference:
            cmd.extend(['--reference-capture', str(args.reference_session/'raw')])
        with (root/'inference.log').open('w') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=300, check=True)
        import numpy as np
        import cv2
        for serial in selected:
            folder = root/'derived'/serial
            info = json.loads((folder/'inference.json').read_text())
            with np.load(folder/'trimodal.npz', allow_pickle=False) as z:
                origin = z['marker_origin'].reshape(len(z['rgb']), -1, 2).astype(np.float32)
                offset = z['marker_offset'].reshape(origin.shape).astype(np.float32)
                finite = np.isfinite(origin).all(-1) & np.isfinite(offset).all(-1)
                n = len(origin)
                # Preserve SDK resolution, coordinate system and units. No depth_m alias.
                packed = dict(rgb=z['rgb'].transpose(0,3,1,2),
                              rgb_warped=z['warped_bgr'][..., ::-1].transpose(0,3,1,2),
                              depth_sdk=z['depth'][:,None].astype(np.float32),
                              marker_2d=np.concatenate([origin, offset, finite[...,None]], axis=-1).astype(np.float32),
                              marker_finite=finite, source_frame_index=z['source_frame_index'],
                              source_episode_id=np.full(n, uuid.uuid4().int & ((1<<63)-1), dtype=np.int64),
                              episode_step=np.arange(n, dtype=np.int64), pair_valid=np.zeros(n,dtype=bool),
                              unloaded_confirmed=np.full(n,args.phase=='baseline',dtype=bool))
                for key in z.files:
                    if key.startswith('host_read_'):
                        packed[key]=z[key]
                np.savez_compressed(folder/'samples.npz', **packed)
            with np.load(folder/'reference.npz', allow_pickle=False) as z:
                if not cv2.imwrite(str(folder/'reference_rgb.png'), cv2.cvtColor(z['rgb'],cv2.COLOR_RGB2BGR)):
                    raise RuntimeError('reference PNG write failed')
            record = dict(serial=serial, physical_finger=mapping.get(serial), sample_count=n,
                          file=str((folder/'samples.npz').relative_to(root)),
                          raw_sha256=info['source_sha256'], reference_sha256=info['reference_source_sha256'],
                          sample_sha256=hashlib.sha256((folder/'samples.npz').read_bytes()).hexdigest(),
                          schema={k:dict(dtype=str(v.dtype),sample_shape=list(v.shape[1:])) for k,v in packed.items()})
            session['cameras'].append(record)
            session['sample_count']+=n
            save()
        session['status']='complete'
    except Exception as exc:
        session['status']='failed'
        session['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        session['finished_utc']=datetime.now(timezone.utc).isoformat()
        save()
    print(f'Saved {session["sample_count"]} native trimodal samples: {root}', flush=True)


if __name__ == '__main__':
    collect_real_trimodal()
