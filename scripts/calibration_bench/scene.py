"""Portable USDA scene with the hand fixture and the animated press head only.

No collision/rigid-body schema is authored: this is explicitly the viewer for
the prescribed-motion soft-contact model, not a dynamics stage.
"""
import numpy as np


def export_scene(path, bench, matrices):
    def vec(value):
        return '(' + ', '.join(f'{float(v):.9g}' for v in value) + ')'
    def matrix(value):
        return '(' + ', '.join(vec(row) for row in value.T) + ')'
    def mesh(name, data, color):
        points = ','.join(vec(v) for v in data.vertices)
        indices = ','.join(str(int(v)) for v in data.faces.ravel())
        return f'''def Mesh "{name}" {{
            point3f[] points = [{points}]
            int[] faceVertexCounts = [{','.join(['3'] * len(data.faces))}]
            int[] faceVertexIndices = [{indices}]
            uniform token subdivisionScheme = "none"
            color3f[] primvars:displayColor = [{vec(color)}]
        }}
'''
    with path.open('x') as f:
        f.write(f'''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
    startTimeCode = 0
    endTimeCode = {len(matrices)-1}
    timeCodesPerSecond = {bench.cfg['sample_hz']}
)
def Xform "World" {{
    custom string simulationMode = "prescribed_motion_geometry_hydroshear_no_rigid_dynamics"
    def DomeLight "Light" {{
        float inputs:intensity = 1500
    }}
    def Camera "Overview" {{
        double3 xformOp:translate = (0.44, -0.36, 0.34)
        float3 xformOp:rotateXYZ = (65, 0, 35)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
        float focalLength = 35
        float2 clippingRange = (0.001, 100)
    }}
''')
        f.write('def Xform "HandFixture" {\n')
        for name, data in bench.hand.items():
            color = (.12, .72, .72) if name == 'lower_mount' else (.12, .16, .22) if 'rubber' in name else (.85, .87, .90)
            f.write(mesh(name, data, color))
        f.write('}\ndef Xform "Tool" {\n')
        f.write(f'matrix4d xformOp:transform = {matrix(matrices[0])}\n')
        f.write('matrix4d xformOp:transform.timeSamples = {\n')
        f.write(',\n'.join(f'{i}: {matrix(m)}' for i, m in enumerate(matrices)))
        f.write('\n}\nuniform token[] xformOpOrder = ["xformOp:transform"]\n')
        for name, data in bench.tool.items():
            f.write(mesh(name, data, (.78, .80, .82)))
        f.write('}\n}\n')


def plot_episode(path, bench, arrays):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    idx = int(np.argmax(arrays['total_normal_load_n']))
    fig = plt.figure(figsize=(15, 8), constrained_layout=True)
    grid = fig.add_gridspec(2, 4)
    ax = fig.add_subplot(grid[:, :2], projection='3d')
    for name, mesh in bench.hand.items():
        faces = mesh.triangles
        color = '#198f94' if name == 'lower_mount' else '#374958' if 'rubber' in name else '#b3bec9'
        ax.add_collection3d(Poly3DCollection(faces, facecolor=color, edgecolor='none'))
    for mesh in bench.tool.values():
        moved = mesh.copy(); moved.apply_transform(arrays['T_world_flange'][idx])
        ax.add_collection3d(Poly3DCollection(moved.triangles[::max(1,len(moved.faces)//1800)], facecolor='#dc8c31', edgecolor='none'))
    ax.set(xlim=(0,.23), ylim=(-.01,.25), zlim=(0,.16), xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)',
           title='V4 fixture + Revo3 + real tool meshes\nPrescribed-motion simulation')
    ax.set_box_aspect((.23,.26,.16)); ax.view_init(48, -62)
    for cell, key, title, scale in [(grid[0,2], 'depth_m','Normal-ray depth (mm)',1000),
                                   (grid[0,3], 'pressure_pa','Model pressure (kPa)',.001)]:
        panel=fig.add_subplot(cell); im=panel.imshow(arrays[key][idx]*scale,origin='lower',cmap='viridis')
        panel.set_title(title); fig.colorbar(im,ax=panel,shrink=.75)
    panel=fig.add_subplot(grid[1,2]); a=arrays['marker'][idx]; valid=a[:,4]>0
    panel.quiver(a[valid,0],a[valid,1],a[valid,2],a[valid,3],angles='xy',scale_units='xy',scale=1)
    panel.set_title('HydroShear marker flow (virtual px)'); panel.set_aspect('equal')
    panel=fig.add_subplot(grid[1,3]); panel.plot(arrays['time_s'],arrays['total_normal_load_n'])
    panel.set(xlabel='Simulation time (s)',ylabel='Model normal load (N)',title='Full load / unload trace')
    fig.savefig(path,dpi=140); plt.close(fig)
