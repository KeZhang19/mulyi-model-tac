"""Native Isaac controls, live channels, load trace and saved-episode scrubber."""
import numpy as np
import omni.ui as ui

from visual_scene import channel_panel


class CollectorPanel:
    def __init__(self, session):
        self.session = session
        self.updates = 0
        self.syncing = False
        self.protocols = session.config['protocols']
        self.images = [ui.ByteImageProvider(), ui.ByteImageProvider()]
        self.window = ui.Window('Calibration collection | Tag frame', width=680, height=1050)
        self.xyz = []
        with self.window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=5, height=0):
                    ui.Label('Revo touch hand / M6x16 fixture / XYZ press head', height=24)
                    ui.Label('Simulation only | configured ROS limits | nominal tip TCP', height=22)
                    ui.Label(f'{session.config["finger"]} {session.config["segment"]} / {session.config["head"]}', height=22)
                    self.status = ui.Label(session.message, height=45, word_wrap=True)
                    ui.Separator(height=4)
                    ui.Label('Target device display (mm)', height=20)
                    for axis, value, bounds in zip('XYZ', session.machine_mm, session.coordinates.bounds_mm):
                        with ui.HStack(height=24):
                            ui.Label(f'{axis} [{bounds[0]:g}, {bounds[1]:g}]', width=165)
                            drag = ui.FloatDrag(min=bounds[0], max=bounds[1], step=.1)
                            drag.model.set_value(float(value))
                            self.xyz.append(drag.model)
                    with ui.HStack(height=30):
                        ui.Button('Move to XYZ', clicked_fn=lambda: self.action(self.move))
                        ui.Button('Sensor clearance / retract', clicked_fn=lambda: self.action(self.clearance))
                    self.position = ui.Label('', height=44, word_wrap=True)
                    self.tag = ui.Label('', height=44, word_wrap=True)
                    ui.Label('Collect a complete baseline / load / hold / unload / release', height=24)
                    with ui.HStack(height=26):
                        self.protocol = ui.ComboBox(0, *self.protocols)
                        ui.Label('Depth (mm)', width=95)
                        self.depth = ui.FloatDrag(min=.01, max=session.config['max_depth_m'] * 1000, step=.05, width=100)
                        self.depth.model.set_value(session.config['depths_m'][0] * 1000)
                    with ui.HStack(height=32):
                        ui.Button('Start episode', clicked_fn=lambda: self.action(self.start))
                        ui.Button('Pause / resume', clicked_fn=self.pause)
                        ui.Button('Stop + save interrupted', clicked_fn=lambda: session.stop())
                    ui.Label('Pause freezes simulation time. Stop preserves partial/fault data.', height=24)
                    ui.ImageWithProvider(self.images[0], width=640, height=410,
                                         fill_policy=ui.IwpFillPolicy.IWP_STRETCH)
                    ui.ImageWithProvider(self.images[1], width=640, height=130,
                                         fill_policy=ui.IwpFillPolicy.IWP_STRETCH)
                    ui.Label('Saved episode replay frame (no new samples)', height=22)
                    self.scrub = ui.IntSlider(min=0, max=0)
                    self.scrub.model.add_value_changed_fn(self.scrub_changed)
                    ui.Button('Replay peak load', height=28, clicked_fn=lambda: self.action(self.peak))
                    self.saved = ui.Label('', height=55, word_wrap=True)
                    ui.Label('Workspace wireframe = software limits. Yellow dots = selected sensor.', height=24)
                    ui.Label('Pressure / RGB uncalibrated; containment checks are sampled.', height=24)
                    ui.Label(str(session.output), height=45, word_wrap=True)
        self.update()

    def action(self, callback):
        try:
            callback()
        except Exception as error:
            self.session.message = f'Rejected: {error}'
        self.update()

    def move(self):
        self.session.move(np.array([m.as_float for m in self.xyz]))

    def clearance(self):
        s = self.session
        target = s.coordinates.tag_to_machine(s.zero + s.bench.tip + [0, 0, s.config['clearance_m']])
        s.move(target)
        for model, value in zip(self.xyz, target):
            model.set_value(float(value))

    def start(self):
        s = self.session
        protocol = self.protocols[self.protocol.model.get_item_value_model().as_int]
        s.start(dict(protocol=protocol, depth_m=self.depth.model.as_float / 1000,
                     offset_xy_m=[0, 0], group_id=-1, split='unassigned', repeat=0,
                     capability='xyz', selection='interactive; no automatic training partition'))

    def pause(self):
        self.session.paused = not self.session.paused
        self.session.message = 'Paused (simulation clock frozen)' if self.session.paused else 'Resumed'
        self.update()

    def scrub_changed(self, model):
        if not self.syncing:
            self.action(lambda: self.session.replay(model.as_int))

    def peak(self):
        a = self.session.last_arrays
        if not a or not len(a['frame_id']):
            raise ValueError('No saved episode yet')
        self.session.replay(int(a['total_normal_load_n'].argmax()))

    def update(self):
        self.updates += 1
        s = self.session
        self.status.text = s.message
        xyz = s.machine_mm
        self.position.text = f'Current device XYZ (mm): {xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}'
        tag = s.coordinates.machine_to_tag(xyz) * 1000
        self.tag.text = f'Tag tip XYZ (mm): {tag[0]:.4f}, {tag[1]:.4f}, {tag[2]:.4f}'
        panels = channel_panel(s.data, s.force_history, s.config['max_depth_m'], s.config['pressure_saturation_pa'])
        for provider, image in zip(self.images, panels):
            provider.set_bytes_data(image.ravel().tolist(), [image.shape[1], image.shape[0]])
        self.syncing = True
        self.scrub.max = len(s.last_arrays['frame_id']) - 1 if s.last_arrays else 0
        self.syncing = False
        self.saved.text = f'Episodes saved: {len(s.reports)} | completed: {sum(r["completed"] for r in s.reports)}\n' + str(s.output)

    def save_images(self, folder):
        from PIL import Image
        panel, trace = channel_panel(self.session.data, self.session.force_history,
                                      self.session.config['max_depth_m'], self.session.config['pressure_saturation_pa'])
        Image.fromarray(panel).save(folder / 'channels.png')
        Image.fromarray(trace).save(folder / 'force_trace.png')
