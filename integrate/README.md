# Integrated Tactile Runner

`integrate/run_integrated_tactile.py` 现在服务两类演示：

| 模式 | 入口 | 作用 |
| --- | --- | --- |
| pressure pad WarpSDF | `--robot-urdf` + `--pressure-layout-urdf` | 读取 URDF pressure pad/taxel layout，输出压阻 force map。 |
| fingertip TacMap demo | `--finger middle`，不传 pressure URDF | 复现 second branch 的 TacMap/FOTS/TacEx/TacSL/HydroShear 指尖展示。 |

这两类不要混在同一个命令里。pressure pad run 会关闭 TacMap/FOTS/TacEx/TacSL/HydroShear；指尖展示不要传 `--pressure-layout-urdf`。

## 环境

```bash
cd ~/BC_Tactile_Lab
export REPO_ROOT="$PWD"
export ISAACLAB_SH="$HOME/IsaacLab/isaaclab.sh"
export VIRTUAL_ENV="$HOME/env_isaaclab"
export PATH="$VIRTUAL_ENV/bin:$PATH"
pip install -e "$REPO_ROOT/source/BrainCo_DexHand"
```

## Pressure Pad: WarpSDF

使用带 `<pressure_pad>` 的 URDF，让中指 PIP pressure pad 按压圆柱：

```bash
TERM=xterm "$ISAACLAB_SH" -p "$REPO_ROOT/integrate/run_integrated_tactile.py" \
  --mode press \
  --robot-urdf 'assets/revo21_right_touch/urdf/revo21_dv2_urdf_right-touch.SLDASM.urdf' \
  --pressure-layout-urdf 'assets/revo21_right_touch/urdf/revo21_dv2_urdf_right-touch.SLDASM.urdf' \
  --pressure-layout-link right_midpip_roll_touch_link \
  --press-motion-actor finger \
  --press-finger-joint right_midmcp_roll_joint \
  --press-finger-end-rad 0.2 \
  --presser cylinder_D4 \
  --num-rows 16 \
  --num-cols 24 \
  --point-distance 0.00075 \
  --force-scale 6 \
  --press-debug-log outputs/pressure_gui_demo/right_midpip_highres_debug.jsonl \
  --press-debug-log-every 5
```

相关参数：

- `--pressure-layout-link`：选择 URDF 里的 pressure pad link。
- `--num-rows --num-cols --point-distance`：临时覆盖 grid 分辨率和 taxel pitch。
- `--show-sample-points --sample-point-radius`：只在临时检查 taxel frame 时打开；它显示的是 `/Visuals/WarpSdfTactile` debug marker，不是最终 pressure map。
- `--force-scale`：只影响可视化颜色缩放。
- `--pressure-contact-model surface_gap`：默认 pressure pad 后端；沿 taxel normal 测表面间隙，不要求刚体明显穿模。
- `--pressure-response-model auto`：默认在 `surface_gap` 下使用 gap fraction 标定，不走 Kelvin-Voigt；如需旧行为，用 `--pressure-response-model penetration_kv`。
- `--mesh-shell-thickness --pressure-gain --pressure-gamma --pressure-max-force`：surface-gap 标定；`--pressure-stiffness --pressure-damping` 只对 `penetration_kv` 有意义。
- `--save-pressure-trace --pressure-trace-dir outputs/pressure_traces`：保存 raw pressure trace。

pressure pad 运行时不要加：

```text
--enable_fots
--enable_tacex_rgb
--enable_tacsl_shear
--enable_hydroshear_marker
--pressure-view-source normal_ray
```

### 压台同步标定：整手沿指定轴运动

真实压台是固定行程时，球头 GUI 标定见下一节：默认移动球头，沿当前 touch link 里最接近 world `-Z` 的局部轴下压。通用 `--press-motion-actor hand` 仍保留给非球头实验。

```bash
TERM=xterm "$ISAACLAB_SH" -p "$REPO_ROOT/integrate/run_integrated_tactile.py" \
  --mode press \
  --robot-urdf 'assets/revo21_right_touch/urdf/revo21_dv2_urdf_right-touch.SLDASM.urdf' \
  --pressure-layout-urdf 'assets/revo21_right_touch/urdf/revo21_dv2_urdf_right-touch.SLDASM.urdf' \
  --pressure-layout-link right_midpip_roll_touch_link \
  --press-motion-actor hand \
  --press-start-offset 0.010 \
  --press-distance 0.005 \
  --press-steps 300 \
  --presser cylinder_D4 \
  --disable-presser-collision \
  --save-pressure-trace \
  --pressure-trace-dir outputs/pressure_traces/right_midpip_hand_normal_calib
```

Pressure pad / WarpSDF run 只需要 presser 的可见 mesh 做 SDF 查询，不需要它参与 PhysX 求解。默认 pressure-pad run 会关闭 presser collision；如果要做 sparse PhysX contact sanity check，再传 `--enable-presser-collision --enable-physx-contact-map`。

通常不要传 `--press-hand-axis-link/--press-hand-axis`。只有非球头实验确实需要其它方向时，才显式覆盖运动轴。

### 球头压台标定

当前球头压台推荐用这组 flag：

```bash
TERM=xterm "$ISAACLAB_SH" -p "$REPO_ROOT/integrate/run_integrated_tactile.py" \
  --mode press \
  --robot-urdf 'assets/revo21_right_touch/urdf/revo21_dv2_urdf_right-touch.SLDASM.urdf' \
  --pressure-layout-urdf 'assets/revo21_right_touch/urdf/revo21_dv2_urdf_right-touch.SLDASM.urdf' \
  --pressure-layout-link right_midpip_roll_touch_link \
  --robot-world-pos 0 0 0.03 \
  --robot-world-quat-wxyz 0.7071068 0 -0.7071068 0 \
  --presser ball_probe \
  --press-motion-actor object \
  --disable-presser-collision \
  --press-setup-before-motion \
  --show-pressure-pad-centers \
  --print-press-coordinates \
  --press-coordinate-log-every 10 \
  --press-contact-search-distance 0.02 \
  --press-indent-depth 0.003
```

关键语义：

- `--pressure-layout-link LINK`：本次运行唯一 active pressure pad link；换 pad 需要改这个 flag 后重启。
- `--press-setup-before-motion`：setup 阶段检查球头位置，Enter 后捕获当前位置作为运动起点。
- `--press-motion-actor object`：移动球头；pressure-pad 默认按压方向是 `ball_probe.STL` local `-Y`。
- `--press-distance D`：从 Enter 捕获的位置开始，沿 ball probe local `-Y` 移动 D 米；不要同时传 `--press-end-offset`。
- `--press-indent-depth D`：先沿 ball probe local `-Y` 接近 pad，第一次 `pressure_penetration_map.max()` 超过 `--press-contact-threshold` 后，再沿同一方向压入 D 米。
- `--press-contact-search-distance D`：配合 `--press-indent-depth` 使用，表示 contact onset 前沿 ball probe local `-Y` 最多搜索多远；不传时使用 `--press-distance` 对应行程。
- `--print-press-coordinates --press-coordinate-log-every N`：打印 `[PRESS_COORD]`，用 `ball_mesh_minus_center` 看球头和 pad center 的偏差。
- `--disable-presser-collision`：pressure pad WarpSDF 推荐保持关闭，避免 PhysX contact 把球头顶飞。
- 不要加 `--press-object-control manual_gui`；那是只拖球看图的模式，Enter 后不会自动下压。

通常不用改的覆盖项：

- `--presser-world-quat-wxyz W X Y Z`：覆盖球头初始 spawn 姿态。
- `--press-center-offset-l DX DY DZ`：在 active link 局部坐标里微调按压中心。

维护边界：

- 球头压台标定 helper 在 `integrate/pressure_calibration_setup.py`。
- `run_integrated_tactile.py` 只调用这些 helper；CLI、env 创建和主循环暂时不拆。

## Fingertip Demo: second Branch View

复现 second 里的中指按压 `square_4`，并显示 marker motion：

```bash
TERM=xterm "$ISAACLAB_SH" -p "$REPO_ROOT/integrate/run_integrated_tactile.py" \
  --mode press \
  --finger middle \
  --presser square_4 \
  --press-start-offset 0.035 \
  --press-end-offset 0.024 \
  --press-steps 400 \
  --press-slide-distance 0.004 \
  --press-slide-steps 300 \
  --press-slide-axis +y \
  --tacmap-ray-mode link_surface \
  --enable_fots \
  --fots-track-contact-center \
  --fots-depth-background \
  --enable_tacex_rgb \
  --enable_tacsl_shear \
  --enable_hydroshear_marker \
  --hydroshear-object-sample-mode poisson \
  --hydroshear-poisson-radius 0.00075 \
  --hydroshear-poisson-initial-count 5000 \
  --hydroshear-debug-visuals \
  --lock-press-finger-joints
```

图像顺序：

```text
WarpSDF | TacMap | FOTS marker/shear overlay | TacEx RGB | TacSL SDF force-field | HydroShear original marker motion | modified HydroShear marker motion
```

五根手指视触觉 + `index` 压敏片额外 presser 示例：

```bash
python integrate/run_integrated_tactile.py \
  --mode press \
  --fingers middle,index,ring,pinky,thumb \
  --presser square_4 \
  --press-start-offset 0.035 \
  --press-end-offset 0.0255 \
  --press-steps 200 \
  --press-slide-distance 0.000 \
  --press-slide-steps 200 \
  --press-motion-frame link_surface \
  --press-slide-axis +u \
  --tacmap-ray-mode link_surface \
  --enable_fots \
  --fots-track-contact-center \
  --fots-depth-background \
  --enable_tacex_rgb \
  --enable_tacsl_shear \
  --enable_hydroshear_marker \
  --hydroshear-debug-visuals \
  --hydroshear-object-sample-mode poisson \
  --hydroshear-poisson-radius 0.00075 \
  --hydroshear-poisson-initial-count 5000 \
  --focus-finger index \
  --no-set-viewport-camera \
  --enable-focus-pressure-pad-presser \
  --pressure-pad-press-start-offset 0.025 \
  --pressure-pad-press-end-offset 0.017 \
  --pressure-pad-press-steps 100
```

核心参数：

- `--tacmap-ray-mode link_surface`：second 的新版 TacMap。用 touch-link 表面网格做 depth reference。
- `--press-start-offset --press-end-offset`：压头起止距离，单位是米。
- `--press-slide-distance --press-slide-axis`：压入后横向滑动距离和方向。
- `--fots-view` 默认是 `markers`；如果改成 `both`，会额外显示一张 FOTS flow pane。
- `--fots-depth-background`：FOTS marker 图背景使用 TacMap depth。
- `--enable_tacex_rgb`：打开 TacEx/GPU-Taxim RGB。
- `--enable_tacsl_shear`：打开 TacSL-style normal/shear force-field。
- `--hydroshear-debug-visuals`：把 HydroShear sample/marker/normal debug 点打到 USD stage，同时保留 7 张图窗口。
- `--hydroshear-debug-only`：只看 HydroShear 内部 debug 图时再用；它会替换主窗口，不再显示 7 张图。
- `--lock-press-finger-joints`：锁住被压手指关节，减少接触后漂移。

## URDF Pressure Pad Contract

runner 当前支持的最小 pressure pad 声明：

```xml
<pressure_pad rows="4" cols="8" taxel_count="32"
              row_distance="0.0035" col_distance="0.002357142857142857"
              normal_axis="2" normal_sign="1"
              origin_semantics="pad_surface"
              stiffness="5000" damping="0" max_force="10"
              gain="1" bias="0" gamma="1" threshold="0" />
```

含义：

- `rows/cols`：taxel matrix 的行列数。
- `point_distance` 或 `row_distance/col_distance`：相邻 taxel 中心距离，单位米。
- `normal_axis/normal_sign`：pad 外法线在 link local frame 中的方向。
- `origin_semantics="pad_surface"`：pad origin 在 pressure-sensitive 表面。
- `stiffness/damping/gain/bias/gamma/max_force/threshold`：penetration 到 force map 的标定参数。

检查 URDF：

```bash
python scripts/force_map/inspect_pressure_urdf.py \
  assets/revo21_right_touch/urdf/revo21_dv2_urdf_right-touch.SLDASM.urdf \
  --require-pressure-layouts
```

## 输出

- GUI/force-map panel：实时看 pressure map 或 six-pane tactile demo。
- `outputs/pressure_gui_demo/*.jsonl`：press debug log。
- `outputs/pressure_traces/*.npz`：`--save-pressure-trace` 保存的 raw trace。

更完整的 pressure trace、GT 分级和标定说明见 `docs/pressure_sdf_tactile.md`。
