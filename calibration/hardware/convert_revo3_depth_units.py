#!/usr/bin/env python3
"""Verify this SDK offline and derive metre depth without changing source captures.

Uses recorded RGB only; never opens a camera. The native sensor constructors are
in offline mode. Replays every source sample before accepting the SDK unit chain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

SDK_ROOT = Path(__file__).resolve().parents[1] / 'brainco-revo3-sdk/runtime/python'
METRES_PER_SDK_UNIT = 1.2 / 200.0


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def depth_to_metres(depth):
    depth = np.asarray(depth)
    if depth.dtype.kind != 'f' or not np.isfinite(depth).all():
        raise ValueError('Expected finite floating-point SDK depth')
    if np.any(depth < 0) or np.any(depth > 1):
        raise ValueError('SDK depth outside verified [0,1] contract')
    return depth.astype(np.float32) * np.float32(METRES_PER_SDK_UNIT)


def verify_sdk_depth_units(source, session):
    import cv2
    sys.path.insert(0, str(SDK_ROOT))
    from pyvitaisdk import VTSensor, VTSensorType, VTSDataType as D
    from pyvitaisdk.vts.core import GFBCI, GFBCT
    from pyvitaisdk.vts.core.base.profile import VideoStreamFmt
    from vitai_depth.reconstruct3d import Postprocessor
    import vitai_depth.reconstruct3d as depth_module
    import pyvitaisdk.vts.core.base.gf_sensor_base as sensor_module

    evidence = dict(sdk_binary_sha256={str(Path(m.__file__).relative_to(SDK_ROOT)): sha256(m.__file__)
                                     for m in (depth_module, sensor_module)}, cameras=[])
    for camera in session['cameras']:
        sn = camera['serial']
        folder = source / 'derived' / sn
        info = json.loads((folder / 'inference.json').read_text())
        product = info['product']
        if product not in {'GFBCI', 'GFBCT'}:
            raise ValueError(f'Unsupported product: {product}')
        prototype = VTSensor(sensor_type=VTSensorType[product])
        sensor = None
        try:
            cfg = prototype._config
            if sha256(cfg.model_path) != info['model_sha256']:
                raise ValueError(f'Model changed since capture: {sn}')
            stream = info['profile']['stream_format']
            sensor = {'GFBCI': GFBCI, 'GFBCT': GFBCT}[product](
                model_path=cfg.model_path, marker_size=list(cfg.marker_size),
                marker_offsets=list(cfg.marker_offsets), output_format=cfg.output_format,
                stream_format=VideoStreamFmt(**stream))
            recon = sensor._reconstruction._implementation
            post = recon.postprocessor
            if not np.allclose([recon._gel_width, recon._gel_height, recon._gel_thickness],
                               [.025, .025, .004], rtol=0, atol=1e-12):
                raise ValueError('Unverified gel profile')
            h, w = post.h, post.w
            # Known metric surface, zero boundaries. The SDK decodes normals
            # from [0,1], with opposite image-y convention. Finite differences
            # and FFT boundary discretisation introduce a small numerical error.
            x = np.arange(w) * recon._gel_width / w
            y = np.arange(h) * recon._gel_height / h
            xx, yy = np.meshgrid(x, y)
            lx, ly = x[-1], y[-1]
            synthetic = []
            for amplitude in [.0005, .001, .002]:
                z = amplitude * np.sin(np.pi*xx/lx) * np.sin(np.pi*yy/ly)
                gx = amplitude*np.pi/lx*np.cos(np.pi*xx/lx)*np.sin(np.pi*yy/ly)
                gy = amplitude*np.pi/ly*np.sin(np.pi*xx/lx)*np.cos(np.pi*yy/ly)
                normal = np.stack([-gx, gy, np.ones_like(gx)]).astype(np.float32)
                normal /= np.linalg.norm(normal, axis=0, keepdims=True)
                out = Postprocessor(h, w, .025, .025).process((normal+1)/2)
                relative_error = float(np.max(np.abs(out/200-z))/amplitude)
                if relative_error > .04:
                    raise ValueError(f'Postprocessor metric scale failed: {relative_error}')
                synthetic.append(dict(amplitude_m=amplitude, max_relative_error=relative_error))
            raw_path = source / 'raw' / sn / 'frames.npz'
            sample_path = source / camera['file']
            if sha256(raw_path) != camera['raw_sha256'] or sha256(sample_path) != camera['sample_sha256']:
                raise ValueError(f'Source hash mismatch: {sn}')
            with np.load(raw_path, allow_pickle=False) as z:
                rgb = z['rgb'].copy()
            with np.load(folder/'reference.npz', allow_pickle=False) as z:
                reference = z['rgb'].copy()
            with np.load(sample_path, allow_pickle=False) as z:
                saved = z['depth_sdk'].copy()
                indices = z['source_frame_index'].copy()
            sensor.calibrate(cv2.cvtColor(reference, cv2.COLOR_RGB2BGR))
            sensor.enable_lowcost_mode(False)
            max_replay_error = 0.
            for i, index in enumerate(indices):
                actual = sensor.collect_sensor_data(D.DEPTH_MAP,
                    frame=cv2.cvtColor(rgb[index], cv2.COLOR_RGB2BGR))[D.DEPTH_MAP]
                error = float(np.max(np.abs(actual-saved[i, 0])))
                max_replay_error = max(max_replay_error, error)
                if not np.allclose(actual, saved[i, 0], rtol=1e-5, atol=1e-7):
                    raise ValueError(f'Current SDK does not reproduce captured depth: {sn}/{index}')
            # Verify final API normalization and clipping independently of model.
            class ConstantPostprocessor:
                value = .2
                def process(self, *args, **kwargs):
                    return np.full((h, w), self.value, dtype=np.float32)
            constant = ConstantPostprocessor()
            recon.postprocessor = constant
            api_checks = []
            for value in [.1, .2, .6, 1.2, 2.4]:
                constant.value = value
                actual = sensor.collect_sensor_data(D.DEPTH_MAP,
                    frame=cv2.cvtColor(rgb[indices[0]], cv2.COLOR_RGB2BGR))[D.DEPTH_MAP]
                expected = min(value, 1.2)/1.2
                if not np.allclose(actual, expected, atol=1e-6, rtol=0):
                    raise ValueError('Final API normalization changed')
                api_checks.append(dict(post_value=value, api_value=float(actual.mean())))
            evidence['cameras'].append(dict(serial=sn, product=product,
                model_sha256=info['model_sha256'], synthetic_surfaces=synthetic,
                api_normalization=api_checks, replay_samples=len(indices),
                replay_max_absolute_error=max_replay_error))
        finally:
            if sensor is not None:
                sensor.release()
            prototype.release()
    evidence.update(passed=True, post_units_per_metre=200, api_normalizer=1.2,
                    metres_per_sdk_unit=METRES_PER_SDK_UNIT,
                    saturation_m=.006, physical_accuracy_validated=False)
    return evidence


def convert_revo3_depth_units():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        parser.error('Output must be a new directory')
    session_path = source/'session.json'
    session = json.loads(session_path.read_text())
    if session.get('schema') != 'revo3_native_trimodal_source_v1' or session.get('status') != 'complete':
        parser.error('Expected completed native trimodal source session')
    evidence = verify_sdk_depth_units(source, session)
    output.mkdir(parents=True, exist_ok=False)
    converted = dict(schema='revo3_metric_depth_derivative_v1', status='converting',
        source_session=str(source), source_session_sha256=sha256(session_path),
        conversion_script_sha256=sha256(__file__), units={'depth_sdk':'normalized SDK depth',
        'depth_m':'m', 'marker_2d':'SDK image pixels'}, metres_per_sdk_unit=METRES_PER_SDK_UNIT,
        training_ready=False, pair_valid=False, physical_calibration_validated=False,
        remaining_blockers=['finger/ROI mapping', 'marker geometry/validity adaptation',
                            'sim/real pairing and training manifest'], cameras=[], sample_count=0)
    for camera in session['cameras']:
        path = source/camera['file']
        with np.load(path, allow_pickle=False) as z:
            data = {k:z[k] for k in z.files}
        data['depth_m'] = depth_to_metres(data['depth_sdk'])
        data['depth_saturated'] = data['depth_sdk'] >= 1
        folder = output/camera['serial']
        folder.mkdir()
        dest = folder/'samples.npz'
        np.savez_compressed(dest, **data)
        with np.load(dest, allow_pickle=False) as z:
            for k, value in data.items():
                if not np.array_equal(z[k], value):
                    raise ValueError(f'Roundtrip failed: {k}')
        if sha256(path) != camera['sample_sha256']:
            raise ValueError('Source changed during conversion')
        converted['cameras'].append(dict(serial=camera['serial'],
            physical_finger=camera['physical_finger'], file=str(dest.relative_to(output)),
            source_sha256=camera['sample_sha256'], sha256=sha256(dest),
            sample_count=len(data['depth_m']), depth_m_min=float(data['depth_m'].min()),
            depth_m_max=float(data['depth_m'].max()), saturated_pixels=int(data['depth_saturated'].sum()),
            schema={k:dict(dtype=str(v.dtype),sample_shape=list(v.shape[1:])) for k,v in data.items()}))
        converted['sample_count'] += len(data['depth_m'])
    converted['status'] = 'complete'
    for name, value in [('unit_verification.json', evidence), ('session.json', converted)]:
        (output/name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(f'Converted {converted["sample_count"]} samples to metres: {output}')


if __name__ == '__main__':
    convert_revo3_depth_units()
