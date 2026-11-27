"""Measure contact and cube motion without adding a policy input or changing control."""

from functools import wraps
import json
import math
import os
import time


def record_trial(run):
    @wraps(run)
    def wrapped(env, trial, seed, launch, config, multi_turn_prompt, output):
        from capx.envs.simulators.tron2_cube_lift import _TableContactFeedback

        low = env.low_level_env
        tracker = low._table_contact_feedback
        measurements = _TableContactFeedback(low.model)
        measurements.summary = {}
        first_xyz = None
        close_time = None
        max_square_displacement = 0.0
        samples = contact_samples = 0
        reset = low.reset
        set_gripper = low.set_gripper
        sample = tracker.sample

        def record_reset(*args, **kwargs):
            nonlocal first_xyz
            result = reset(*args, **kwargs)
            with low._physics_lock:
                first_xyz = low._cube_init_world.tolist()
            return result

        def record_gripper(side, open_fraction, *args, **kwargs):
            nonlocal close_time
            if side == "left" and open_fraction < 0.5:
                with low._physics_lock:
                    if close_time is None:
                        close_time = float(low._motor_data.time)
            return set_gripper(side, open_fraction, *args, **kwargs)

        def record_sample(data):
            nonlocal max_square_displacement, samples, contact_samples
            sample(data)
            if first_xyz is None:
                return
            measurements.sample(data)
            samples += 1
            contact_samples += int(measurements.loaded)
            if close_time is None:
                x, y, z = data.xpos[low._cube_id]
                squared = (x - first_xyz[0]) ** 2 + (y - first_xyz[1]) ** 2 + (z - first_xyz[2]) ** 2
                max_square_displacement = max(max_square_displacement, squared)

        with low._physics_lock:
            low.reset = record_reset
            low.set_gripper = record_gripper
            tracker.sample = record_sample
        started = time.monotonic()
        try:
            row = run(env, trial, seed, launch, config, multi_turn_prompt, output)
        finally:
            with low._physics_lock:
                low.reset = reset
                low.set_gripper = set_gripper
                tracker.sample = sample
                cube_xyz = low._motor_data.xpos[low._cube_id].tolist()
                loads = measurements.summary.copy()

        dt = float(low.model.opt.timestep)
        stats = {
            "prompt_variant": os.environ["CAPX_PROMPT_VARIANT"],
            "episode_wall_sec": time.monotonic() - started,
            "initial_cube_world_m": first_xyz,
            "final_cube_world_m": cube_xyz if first_xyz is not None else None,
            "final_lift_cm": 100 * (cube_xyz[2] - first_xyz[2]) if first_xyz is not None else None,
            "table_peak_normal_n": max((peak for peak, _ in loads.values()), default=0.0),
            "table_contact_duration_sim_s": contact_samples * dt,
            "table_links": {
                low.model.body(body).name: {"peak_normal_n": peak, "loaded_sim_s": count * dt}
                for body, (peak, count) in loads.items()
            },
            "first_close_sim_s": close_time,
            "max_cube_displacement_preclose_mm": 1000 * math.sqrt(max_square_displacement),
            "physics_samples": samples,
        }
        (output / "physical_metrics.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        row.update(stats)
        return row

    return wrapped
