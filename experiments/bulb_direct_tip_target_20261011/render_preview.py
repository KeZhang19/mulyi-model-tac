"""Plot direct targets and draw the production HUD from manufactured CPU states.

This is a coordinate/annotation demonstration, not a trained-policy rollout.
"""
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, Rectangle
from PIL import Image, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("checks", ROOT / "scripts/diagnostics/check_bulb_grasp_curriculum.py")
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)
progress_module = checks.load("bulb_check.progress", checks.TASK / "progress.py")
font = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams["font.family"] = font.get_name()
plt.rcParams["axes.unicode_minus"] = False

states = []
for depth, radial, tilt in ((.004, .0012, 20.), (.0055, .00025, 20.)):
    env, command = checks.fixture(1)
    command.stage_two[:] = True
    checks.place_inserted(env, command, depths=[depth], radial=[radial], tilts=[tilt])
    command.advance()
    states.append((env, command, progress_module.stage_progress_snapshot(command)))

fig, axes = plt.subplots(1, 2, figsize=(13, 8), facecolor="#f3f6fa")
fig.suptitle("v9：二、三阶段直接使用螺纹底端坐标", fontsize=23, fontproperties=font, y=.96)
fig.text(.5, .91, "构造几何状态示意 · 未运行物理或策略 · 图中完成状态由生产判据计算", ha="center",
         fontsize=12, fontproperties=font, color="#596579")
records = []
for ax, (env, command, state), title in zip(axes, states, ("浅插未完成", "浅插完成，进入旋入")):
    ax.set_facecolor("#ffffff")
    ax.set_xlim(-45, 70)
    ax.set_ylim(-55, 115)
    ax.set_aspect("equal")
    ax.axvline(0, color="#b0bccb", linestyle="--", linewidth=1)
    ax.axhline(0, color="#5d6d81", linewidth=1)
    ax.add_patch(Rectangle((-14, -44.7), 28, 44.7, fill=False, linewidth=2, edgecolor="#78899e"))
    ax.text(-40, 2, "孔口 z=0", fontsize=10, fontproperties=font)
    mouth = command.socket.data.root_pos_w.clone()
    mouth[:, 2] += command.cfg.socket_mouth_x_m
    tip = checks.curriculum.bulb_tip_position_w(command.object, command.scales, command.cfg)[0]
    center = command.object.data.root_pos_w[0]
    tip_relative = (tip - mouth[0]).numpy() * 1000
    center_relative = (center - mouth[0]).numpy() * 1000
    # This plot shows radial magnitude horizontally; it is not world X.
    radial_mm = (tip_relative[:2] ** 2).sum() ** .5
    center_radial = (center_relative[:2] ** 2).sum() ** .5
    ax.plot([radial_mm, center_radial], [tip_relative[2], center_relative[2]], color="#728399", linewidth=12,
            alpha=.55, solid_capstyle="round")
    ax.add_patch(Circle((center_radial, center_relative[2]), 19, color="#a4bfdf", alpha=.65))
    ax.scatter([0], [-5.5], s=130, color="#f1a52b", zorder=6)
    ax.scatter([radial_mm], [tip_relative[2]], s=65, color="#db4958", zorder=7)
    ax.annotate("实际底端", (radial_mm, tip_relative[2]), (22, -15),
                arrowprops={"arrowstyle": "->", "color": "#db4958"}, fontsize=11, fontproperties=font)
    ax.annotate("浅插目标：孔内 5.5 mm", (0, -5.5), (-40, -39),
                arrowprops={"arrowstyle": "->", "color": "#b97c15"}, fontsize=10, fontproperties=font)
    if state["stage"] == 3:
        goal_depth = command.cfg.socket_mouth_x_m - (command.pose_command_w[0, 2] - command.socket.data.root_pos_w[0, 2]).item()
        ax.scatter([0], [-goal_depth * 1000], s=160, facecolors="none", edgecolors="#6d49b5", linewidths=2, zorder=8)
        ax.text(-40, -50, f"旋入当前底端目标：{goal_depth * 1000:.2f} mm", fontsize=10, fontproperties=font, color="#6d49b5")
    ax.set_title(title, fontsize=16, fontproperties=font, color="#168257" if state["completed"][1] else "#b34747")
    ax.set_xlabel("横向径向距离（mm，示意）", fontproperties=font)
    ax.set_ylabel("相对孔口高度（mm）", fontproperties=font)
    labels = [("阶段一", state["completed"][0]), ("阶段二", state["completed"][1]), ("阶段三", state["completed"][2])]
    completion = "  |  ".join(f"{label}：{'完成' if done else '未完成'}" for label, done in labels)
    ax.text(.02, .94, completion, transform=ax.transAxes, fontsize=11, fontproperties=font)
    for i, check in enumerate(state["shallow_checks"]):
        name = ("插入深度", "底端横向偏差", "倾角")[i]
        line = f"{'通过' if check['passed'] else '未通过'}  {name}：{check['value']:.2f} / {check['requirement']}"
        ax.text(.02, .89 - i * .05, line, transform=ax.transAxes, fontsize=11, fontproperties=font,
                color="#168257" if check["passed"] else "#b34747")
    records.append(state)
fig.text(.5, .055, "一阶段仍追踪中心到孔口上方 15 cm；二阶段不再换算中心目标。三阶段沿原螺距继续推进底端。",
         ha="center", fontsize=12, fontproperties=font)
fig.subplots_adjust(top=.84, bottom=.13, wspace=.24)
fig.savefig(OUT / "direct_tip_completion.png", dpi=150)
plt.close(fig)

# Blank preview surface exercises the exact production HUD, clearly labeled.
frame = Image.new("RGB", (1280, 420), (43, 53, 68))
hud_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
progress_module.draw_stage_progress(frame, records[0], "V9 HUD PREVIEW | Manufactured state; no physics or policy", hud_font)
frame.save(OUT / "hud_preview.png")
(OUT / "preview_states.json").write_text(json.dumps(records, indent=2) + "\n")
print("preview PNGs and state JSON saved")
