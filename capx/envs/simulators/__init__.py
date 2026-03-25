# import all environments here to register them!
from capx.envs.base import list_envs, register_env


from .franka_real import FrankaRealLowLevel
register_env("franka_real_low_level", FrankaRealLowLevel)

# NOTE: Can only have one of Robosuite or LIBERO installed at a time!
# Using Robosuite run: uv sync --extra robosuite
try:
    from .robosuite_cube_lift import FrankaRobosuiteCubeLiftLowLevel
    from .robosuite_cubes import FrankaRobosuiteCubesLowLevel
    from .robosuite_cubes_restack import FrankaRobosuiteCubesRestackLowLevel
    from .robosuite_spill_wipe import FrankaRobosuiteSpillWipeLowLevel
    from .robosuite_handover import RobosuiteHandoverEnv
    from .robosuite_two_arm_lift import RobosuiteTwoArmLiftEnv
    from .robosuite_nut_assembly import FrankaRobosuiteNutAssembly
    from .robosuite_nut_assembly import FrankaRobosuiteNutAssemblyVisual

    register_env("franka_robosuite_cube_lift_low_level", FrankaRobosuiteCubeLiftLowLevel)
    register_env("franka_robosuite_cubes_low_level", FrankaRobosuiteCubesLowLevel)
    register_env("franka_robosuite_cubes_restack_low_level", FrankaRobosuiteCubesRestackLowLevel)
    register_env("franka_robosuite_spill_wipe_low_level", FrankaRobosuiteSpillWipeLowLevel)


    register_env("franka_robosuite_nut_assembly_low_level", FrankaRobosuiteNutAssembly)
    register_env("franka_robosuite_nut_assembly_low_level_visual", FrankaRobosuiteNutAssemblyVisual)

    register_env("two_arm_handover_robosuite", RobosuiteHandoverEnv)
    register_env("two_arm_lift_robosuite", RobosuiteTwoArmLiftEnv)
except Exception:
    print("Robosuite not installed!")

# NOTE: Can only have one of LIBERO or Robosuite installed at a time!
# Using LIBERO run: uv sync --extra libero --extra contactgraspnet
try:
    from .libero import FrankaLiberoOpenMicrowave, FrankaLiberoPickPlace, FrankaLiberoPickAlphabetSoup, FrankaLiberoTask

    register_env("franka_libero_pick_place_low_level", FrankaLiberoPickPlace)
    register_env("franka_libero_open_microwave_low_level", FrankaLiberoOpenMicrowave)
    register_env("franka_libero_pick_alphabet_soup_low_level", FrankaLiberoPickAlphabetSoup)

    # Register all LIBERO suites with all task indices for easy YAML access.
    # Usage in YAML: low_level: franka_libero_<suite>_<task_id>_low_level
    # e.g. franka_libero_libero_spatial_2_low_level
    import functools
    for _suite in ["libero_10", "libero_90", "libero_object", "libero_spatial", "libero_goal"]:
        try:
            from libero import benchmark as _bm
            _bd = _bm.get_benchmark_dict()
            _n = _bd[_suite]().n_tasks
        except Exception:
            _n = 10
        for _tid in range(_n):
            _name = f"franka_libero_{_suite}_{_tid}_low_level"
            register_env(_name, functools.partial(FrankaLiberoTask, suite_name=_suite, task_id=_tid))
except Exception:
    print("LIBERO not installed!")

try:
    from .r1pro_b1k import R1ProBehaviourLowLevel
    register_env("r1pro_b1k_low_level", R1ProBehaviourLowLevel)
except Exception:
    print("R1Pro not installed!")
