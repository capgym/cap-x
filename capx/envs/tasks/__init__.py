from .base import (
    CodeExecEnvConfig,
    CodeExecutionEnvBase,
    get_config,
    get_exec_env,
    list_configs,
    list_exec_envs,
    register_config,
    register_exec_env,
)
from .franka.franka_cube_restack import FrankaRestackCodeEnv
from .franka.franka_lift import FrankaLiftCodeEnv
from .franka.franka_nut_assembly import FrankaNutAssemblyCodeEnv
from .franka.franka_pick_place import FrankaPickPlaceCodeEnv
from .franka.franka_spill_wipe import FrankaSpillWipeCodeEnv
from .franka.two_arm_handover import TwoArmHandoverCodeEnv
from .franka.two_arm_lift import TwoArmLiftCodeEnv

register_exec_env("franka_real_code_env", FrankaPickPlaceCodeEnv)
register_config(
    "franka_real_code_env",
    CodeExecEnvConfig(
        low_level="franka_real_low_level",
        apis=["FrankaControlApi"],
    ),
)
register_exec_env("franka_robosuite_spill_wipe_code_env", FrankaSpillWipeCodeEnv)
register_config(
    "franka_robosuite_spill_wipe_code_env",
    CodeExecEnvConfig(
        low_level="franka_robosuite_spill_wipe_low_level",
        apis=["FrankaControlSpillWipePrivilegedApi"],
    ),
)

register_exec_env("franka_pick_place_code_env", FrankaPickPlaceCodeEnv)
register_config(
    "franka_pick_place_code_env",
    CodeExecEnvConfig(
        low_level="franka_cubes_low_level",
        apis=["FrankaControlPrivilegedApi"],
    ),
)
register_exec_env("franka_robosuite_pick_place_code_env", FrankaPickPlaceCodeEnv)
register_config(
    "franka_robosuite_pick_place_code_env",
    CodeExecEnvConfig(
        low_level="franka_robosuite_cubes_low_level",
        apis=["FrankaControlPrivilegedApi"],
    ),
)
register_exec_env("franka_nut_assembly_code_env", FrankaNutAssemblyCodeEnv)
register_config(
    "franka_nut_assembly_code_env",
    CodeExecEnvConfig(
        low_level="franka_robosuite_nut_assembly_low_level",
        apis=["FrankaControlNutAssemblyPrivilegedApi"],
        privileged=True,
    ),
)

register_exec_env("franka_nut_assembly_code_env_visual", FrankaNutAssemblyCodeEnv)
register_config(
    "franka_nut_assembly_code_env_visual",
    CodeExecEnvConfig(
        low_level="franka_robosuite_nut_assembly_low_level_visual",
        apis=["FrankaControlNutAssemblyVisualApi"],
        privileged=False,
    ),
)
register_exec_env("franka_pick_place_multi_code_env", FrankaPickPlaceCodeEnv)
register_config(
    "franka_pick_place_multi_code_env",
    CodeExecEnvConfig(
        low_level="franka_cubes_low_level",
        apis=["FrankaControlMultiPrivilegedApi"],
    ),
)

register_exec_env("franka_lift_code_env", FrankaLiftCodeEnv)
register_config(
    "franka_lift_code_env",
    CodeExecEnvConfig(
        low_level="franka_robosuite_cube_lift_low_level",
        apis=["FrankaControlPrivilegedApi"],
    ),
)
register_exec_env("two_arm_handover_code_env", TwoArmHandoverCodeEnv)
register_config(
    "two_arm_handover_code_env",
    CodeExecEnvConfig(
        low_level="two_arm_handover_robosuite",
        apis=["FrankaHandoverApi"],
    ),
)

# from .franka.franka_libero_pick_place import FrankaLiberoPickPlaceCodeEnv
# from .franka.franka_libero_open_microwave import FrankaLiberoOpenMicrowaveCodeEnv
# from .franka.franka_libero_pick_alphabet_soup import FrankaLiberoPickAlphabetSoupCodeEnv
from .franka.franka_libero_env import FrankaLiberoCodeEnv
register_exec_env("franka_libero_code_env", FrankaLiberoCodeEnv)
register_config(
    "franka_libero_code_env",
    CodeExecEnvConfig(
        low_level="franka_libero_low_level",
        apis=["FrankaLiberoApi"],
    ),
)

# # Non-privileged (perception-based) Libero environments
# register_exec_env("franka_libero_pick_place_code_env", FrankaLiberoPickPlaceCodeEnv)
# register_config(
#     "franka_libero_pick_place_code_env",
#     CodeExecEnvConfig(
#         low_level="franka_libero_pick_place_low_level",
#         apis=["FrankaLiberoApi"],
#         privileged=False,
#     ),
# )

# register_exec_env("franka_libero_open_microwave_code_env", FrankaLiberoOpenMicrowaveCodeEnv)
# register_config(
#     "franka_libero_open_microwave_code_env",
#     CodeExecEnvConfig(
#         low_level="franka_libero_open_microwave_low_level",
#         apis=["FrankaLiberoApi"],
#         privileged=False,
#     ),
# )

# register_exec_env("franka_libero_pick_alphabet_soup_code_env", FrankaLiberoPickAlphabetSoupCodeEnv)
# register_config(
#     "franka_libero_pick_alphabet_soup_code_env",
#     CodeExecEnvConfig(
#         low_level="franka_libero_pick_alphabet_soup_low_level",
#         apis=["FrankaLiberoApi"],
#         privileged=False,
#     ),
# )

# # Privileged Libero environments
# register_exec_env("franka_libero_pick_place_code_env_privileged", FrankaLiberoPickPlaceCodeEnv)
# register_config(
#     "franka_libero_pick_place_code_env_privileged",
#     CodeExecEnvConfig(
#         low_level="franka_libero_pick_place_low_level",
#         apis=["FrankaLiberoPrivilegedApi"],
#         privileged=True,
#     ),
# )

# register_exec_env("franka_libero_open_microwave_code_env_privileged", FrankaLiberoOpenMicrowaveCodeEnv)
# register_config(
#     "franka_libero_open_microwave_code_env_privileged",
#     CodeExecEnvConfig(
#         low_level="franka_libero_open_microwave_low_level",
#         apis=["FrankaLiberoPrivilegedApi"],
#         privileged=True,
#     ),
# )
register_exec_env("franka_restack_code_env", FrankaRestackCodeEnv)
register_config(
    "franka_restack_code_env",
    CodeExecEnvConfig(
        low_level="franka_robosuite_cubes_restack_low_level",
        apis=["FrankaControlPrivilegedApi"],
    ),
)

from .r1pro.r1pro_pickup_radio import R1ProRadioCodeEnv
register_exec_env("r1pro_radio_code_env", R1ProRadioCodeEnv)
register_config(
    "r1pro_radio_code_env",
    CodeExecEnvConfig(
        low_level="r1pro_b1k_low_level",
        apis=["R1ProControlApi"],
    ),
)
from .r1pro.r1pro_pickup_trash import R1ProTrashCodeEnv
register_exec_env("r1pro_trash_code_env", R1ProTrashCodeEnv)
register_config(
    "r1pro_trash_code_env",
    CodeExecEnvConfig(
        low_level="r1pro_b1k_low_level",
        apis=["R1ProControlApi"],
    ),
)
