"""Tag-based scene and image panels for the prescribed-motion viewer."""
import itertools
import re

import numpy as np
from PIL import Image, ImageDraw

from scene import export_scene


def export_visual_scene(path, session, matrices=None):
    matrices = np.asarray([session.data['T_world_flange']]) if matrices is None else matrices
    export_scene(path, session.bench, matrices)
    text = path.read_text()
    # Fit the complete assembly, looking at its centre in the current Tag frame.
    visible_bounds = [mesh.bounds for mesh in session.bench.hand.values()]
    for mesh in session.bench.tool.values():
        moved = mesh.copy(); moved.apply_transform(matrices[0])
        visible_bounds.append(moved.bounds)
    lo = np.asarray(visible_bounds)[:, 0].min(0)
    hi = np.asarray(visible_bounds)[:, 1].max(0)
    centre = (lo + hi) / 2
    radius = np.linalg.norm(hi - lo) / 2
    direction = np.array([1., -1.15, 1.])
    eye = centre + direction / np.linalg.norm(direction) * max(.5, radius * 4.6)
    backward = (eye - centre) / np.linalg.norm(eye - centre)
    right = np.cross([0, 0, 1], backward); right /= np.linalg.norm(right)
    camera = np.eye(4)
    camera[:3, :3] = np.stack((right, np.cross(backward, right), backward), axis=1)
    camera[:3, 3] = eye
    matrix_text = '(' + ', '.join('(' + ', '.join(f'{v:.9g}' for v in row) + ')' for row in camera.T) + ')'
    text = re.sub(r'    def Camera "Overview" \{.*?\n    \}', f'''    def Camera "Overview" {{
        matrix4d xformOp:transform = {matrix_text}
        uniform token[] xformOpOrder = ["xformOp:transform"]
        float focalLength = 24
        float2 clippingRange = (0.001, 100)
    }}''', text, flags=re.S)
    text = text.replace('float inputs:intensity = 1500', 'float inputs:intensity = 700')
    # Explicit visual materials; does not alter collision/contact parameters.
    for name, material in [('tag_center', 'Fixture'), ('connection_flange', 'Tool'), ('press_head', 'Tool')]:
        text = text.replace(f'def Mesh "{name}" {{', f'def Mesh "{name}" {{\n            rel material:binding = </World/Materials/{material}>')
    bounds = session.coordinates.tag_bounds_m
    corners = list(itertools.product(*bounds))
    edges = [(a, b) for i, a in enumerate(corners) for b in corners[i + 1:]
             if sum(abs(x - y) > 1e-10 for x, y in zip(a, b)) == 1]
    def points(values):
        return ', '.join('(' + ', '.join(f'{float(x):.9g}' for x in v) + ')' for v in values)
    additions = f'''def BasisCurves "ConfiguredWorkspace" {{
        uniform token type = "linear"
        uniform token wrap = "nonperiodic"
        int[] curveVertexCounts = [{','.join(['2'] * len(edges))}]
        point3f[] points = [{points([p for edge in edges for p in edge])}]
        float[] widths = [0.0004]
        color3f[] primvars:displayColor = [(0.22, 0.48, 0.66)]
        custom string meaning = "Configured software travel envelope; NOT mechanical gantry geometry"
    }}
    def Points "SelectedSensor" {{
        point3f[] points = [{points(session.bench.surface[session.bench.valid])}]
        float[] widths = [0.00035]
        uniform token widths:interpolation = "constant"
        color3f[] primvars:displayColor = [(0.97, 0.76, 0.12)]
    }}
'''
    additions += '''def Scope "Materials" {
        def Material "Fixture" {
            token outputs:surface.connect = </World/Materials/Fixture/Shader.outputs:surface>
            def Shader "Shader" {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.07, 0.38, 0.34)
                float inputs:roughness = 0.65
                token outputs:surface
            }
        }
        def Material "Tool" {
            token outputs:surface.connect = </World/Materials/Tool/Shader.outputs:surface>
            def Shader "Shader" {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.72, 0.36, 0.07)
                float inputs:roughness = 0.5
                token outputs:surface
            }
        }
    }
'''
    for axis, color in enumerate([(1, .15, .15), (.15, 1, .2), (.2, .4, 1)]):
        endpoint = np.zeros(3); endpoint[axis] = .03
        additions += f'''def BasisCurves "TagAxis_{axis}" {{
            uniform token type = "linear"
            int[] curveVertexCounts = [2]
            point3f[] points = [(0,0,0), {points([endpoint])}]
            float[] widths = [0.001]
            color3f[] primvars:displayColor = [{points([color])}]
        }}
'''
    path.write_text(text[:-2] + additions + '}\n')


def channel_panel(data, history, max_depth_m, saturation_pa):
    """Fixed scales; no framewise autoscale that can conceal saturation."""
    from matplotlib import colormaps
    panel = Image.new('RGBA', (640, 410), '#131e2a')
    draw = ImageDraw.Draw(panel)
    w, h = data['depth_m'].shape[1], data['depth_m'].shape[0]
    for key, title, maximum, xy in [
        ('rgb', 'Synthetic RGB / uncalibrated', 255, (8, 24)),
        ('depth_m', f'Depth: 0 - {max_depth_m * 1000:.2f} mm', max_depth_m, (328, 24)),
        ('pressure_pa', f'Model pressure: 0 - {saturation_pa / 1000:.0f} kPa', saturation_pa, (8, 222)),
    ]:
        arr = data[key]
        rgb = arr if key == 'rgb' else (colormaps['viridis'](np.clip(arr / maximum, 0, 1))[..., :3] * 255).astype(np.uint8)
        image = Image.fromarray(rgb).transpose(Image.Transpose.FLIP_TOP_BOTTOM).resize((300, 164), Image.Resampling.NEAREST)
        panel.paste(image, xy)
        draw.text((xy[0], xy[1] - 18), title, fill='#e4ecf5')
    draw.text((328, 204), 'Marker flow / virtual pixels', fill='#e4ecf5')
    draw.rectangle((328, 222, 628, 386), fill='#203343')
    for u, v, du, dv, valid in data['marker']:
        if valid:
            x, y = 328 + u / w * 300, 386 - v / h * 164
            x1, y1 = x + du / w * 300, y - dv / h * 164
            draw.line((x, y, x1, y1), fill='#60e3b0', width=2)
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill='#f5cf5b')
    draw.text((8, 392), 'RGB / depth / marker are virtual sensor outputs. Pressure is a provisional model.', fill='#a6b8cc')
    trace = Image.new('RGBA', (640, 130), '#131e2a')
    d = ImageDraw.Draw(trace)
    values = history[-1800:]
    peak = max(max(values, default=0), 1)
    d.text((8, 4), f'Model normal load (N) | peak {max(values, default=0):.4f}', fill='#e4ecf5')
    d.line((10, 110, 630, 110), fill='#526779')
    if len(values) > 1:
        d.line([(10 + i / (len(values) - 1) * 620, 110 - v / peak * 80)
                for i, v in enumerate(values)], fill='#5ad6ad', width=2)
    return np.asarray(panel), np.asarray(trace)
