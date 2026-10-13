"""Per-frame task completion data and the recording HUD (no Kit imports)."""
import math


def stage_progress_snapshot(command, index=0):
    """Separate latched stage completion from instantaneous geometric checks."""
    metrics, cfg = command.metrics, command.cfg
    number = lambda name: float(metrics[name][index].item())
    passed = lambda name: bool(metrics[name][index].item())
    stage = int(number("stage"))
    depth = number("insertion_depth_m")
    radial = number("insertion_tip_radial_error_m")
    tilt = number("insertion_tilt_deg")
    checks = [
        {"label": "Depth", "value": depth * 1000,
         "requirement": f"{cfg.insertion_depth_range_m[0] * 1000:g}..{cfg.insertion_depth_range_m[1] * 1000:g} mm",
         "passed": passed("insertion_depth_ok")},
        {"label": "Tip radial", "value": radial * 1000,
         "requirement": f"<= {cfg.shallow_insertion_radial_tolerance_m * 1000:g} mm",
         "passed": passed("insertion_tip_radial_ok")},
        {"label": "Tilt", "value": tilt,
         "requirement": f"< {cfg.shallow_insertion_tilt_tolerance_deg:g} deg",
         "passed": passed("shallow_tilt_ok")},
    ]
    final_depth_error = abs(depth - number("screw_depth_target_m"))
    final_depth_tolerance = cfg.screw_final_depth_tolerance_m * float(command.scales[index].item())
    final_checks = [
        {"label": "Net turns", "value": number("screw_angular_turns"), "requirement": f">= {cfg.screw_target_turns:g}",
         "passed": bool(metrics["screw_angular_turns"][index] >= cfg.screw_target_turns - 1e-5)},
        {"label": "Final depth error", "value": final_depth_error * 1000,
         "requirement": f"<= {final_depth_tolerance * 1000:g} mm",
         "passed": bool((metrics["insertion_depth_m"][index] - metrics["screw_depth_target_m"][index]).abs()
                        <= cfg.screw_final_depth_tolerance_m * command.scales[index])},
        {"label": "Tip/lip radial", "value": number("insertion_radial_error_m") * 1000,
         "requirement": f"<= {cfg.insertion_radial_tolerance_m * 1000:g} mm",
         "passed": bool(metrics["insertion_radial_error_m"][index] <= cfg.insertion_radial_tolerance_m)},
        {"label": "Final tilt", "value": tilt, "requirement": f"<= {cfg.insertion_tilt_tolerance_deg:g} deg",
         "passed": passed("screw_tilt_ok")},
        {"label": "Relative speed", "value": number("insertion_speed_m_s"),
         "requirement": f"<= {cfg.insertion_speed_limit_m_s:g} m/s",
         "passed": bool(metrics["insertion_speed_m_s"][index] <= cfg.insertion_speed_limit_m_s)},
        {"label": "Relative angular speed", "value": number("insertion_angular_speed_rad_s"),
         "requirement": f"<= {cfg.insertion_angular_speed_limit_rad_s:g} rad/s",
         "passed": bool(metrics["insertion_angular_speed_rad_s"][index] <= cfg.insertion_angular_speed_limit_rad_s)},
        {"label": "Stable hold", "value": number("screw_hold_s"), "requirement": f">= {cfg.insertion_hold_s:g} s",
         "passed": int(command.screw_hold_steps[index].item()) >= command.required_hold_steps},
        {"label": "Elapsed", "value": number("stage3_elapsed_s"), "requirement": f"<= {cfg.stage3_duration_s:g} s",
         "passed": int(command.stage_three_steps[index].item()) <= math.ceil(cfg.stage3_duration_s / command._env.step_dt)},
        {"label": "Valid helix", "value": int(command.screw.valid[index].item()), "requirement": "1",
         "passed": bool(command.screw.valid[index].item())},
    ]
    return {
        "stage": stage,
        "completed": [passed("stage1_success"), passed("stage2_success"), bool(number("success"))],
        "position_reference": "root" if stage == 1 else "thread_tip",
        "target_world_m": command.pose_command_w[index, :3].tolist(),
        "tracked_world_m": command.tracked_position_w[index].tolist(),
        "position_error_cm": number("position_error") * 100,
        "shallow_checks": checks,
        "final_checks": final_checks,
        "screw_turns": number("screw_turns"),
        "screw_hold_s": number("screw_hold_s"),
        "success": bool(number("success")),
    }


def draw_stage_progress(frame, progress, title, font):
    """Draw completion and individual shallow-entry checks on a rendered frame."""
    from PIL import ImageDraw

    draw = ImageDraw.Draw(frame)
    green, red, muted = (70, 220, 160), (255, 110, 110), (170, 180, 195)
    row_height = max(24, getattr(font, "size", 22) + 8)
    x, y = 25, 20
    checks = progress["final_checks"] if progress["stage"] == 3 else progress["shallow_checks"]
    draw.rectangle((12, 12, frame.width - 12, 20 + (6 + len(checks)) * row_height), fill=(15, 20, 27))

    def row(text, color="white"):
        nonlocal y
        draw.text((x, y), text, font=font, fill=color)
        y += row_height

    row(title)
    labels = ("1 Grasp/lift", "2 Tip entry", "3 Screw")
    statuses = ["DONE" if done else ("ACTIVE" if progress["stage"] == i + 1 else "PENDING")
                for i, done in enumerate(progress["completed"])]
    row(" | ".join(f"{label}: {status}" for label, status in zip(labels, statuses)))
    row(f"Reference: {progress['position_reference']} | Position error: {progress['position_error_cm']:.2f} cm")
    for check in checks:
        status = "PASS" if check["passed"] else "FAIL"
        row(f"[{status}] {check['label']}: {check['value']:.4g} ({check['requirement']})",
            (green if check["passed"] else red) if progress["stage"] >= 2 else muted)
    if progress["stage"] == 1:
        row("Stage 1: root distance < 5 cm; entry checks become active in stage 2")
    elif progress["stage"] == 2:
        row("Stage 2: all three checks in this frame | No hold/speed gate")
    else:
        row(f"Turns: {progress['screw_turns']:.2f}/3 | Stable hold: {progress['screw_hold_s']:.2f}s | Success: {progress['success']}")
    xyz = lambda point: ", ".join(f"{value:.4f}" for value in point)
    row(f"Actual {progress['position_reference']} XYZ: ({xyz(progress['tracked_world_m'])}) m", red)
    row(f"Target XYZ: ({xyz(progress['target_world_m'])}) m | Orange=goal, red=actual, green=passed", (255, 195, 80))
    return frame
