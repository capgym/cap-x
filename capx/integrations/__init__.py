from .base_api import list_apis, register_api
from .franka.control import FrankaControlApi
from .franka.control_privileged import FrankaControlPrivilegedApi
from .franka.control_reduced import FrankaControlApiReduced
from .franka.control_reduced_skill_library import FrankaControlApiReducedSkillLibrary
from .franka.control_reduced_exampleless import FrankaControlApiReducedExampleless
from .franka.nut_assembly_privileged import FrankaControlNutAssemblyPrivilegedApi
from .franka.nut_assembly_visual import FrankaControlNutAssemblyVisualApi
from .franka.spill_wipe import FrankaControlSpillWipeApi
from .franka.spill_wipe_privileged import FrankaControlSpillWipePrivilegedApi
from .franka.handover_privileged import FrankaHandoverPrivilegedApi
from .franka.handover import FrankaHandoverApi
from .franka.handover_reduced import FrankaHandoverApiReduced
from .franka.handover_reduced_exampleless import FrankaHandoverApiReducedExampleless
from .franka.two_arm_lift import FrankaTwoArmLiftApi
from .franka.two_arm_lift_privileged import FrankaTwoArmLiftPrivilegedApi
from .g1.control import G1LeftRealControlApi, G1RealControlApi
from .g1.vision import G1CameraApi
try:
    from .franka.libero import FrankaLiberoApi
    from .franka.libero_privileged import FrankaLiberoPrivilegedApi
    from .franka.libero_reduced import FrankaLiberoApiReduced
    from .franka.libero_reduced_skill_library import FrankaLiberoApiReducedSkillLibrary
    _libero_available = True
except ImportError:
    _libero_available = False
    print("LIBERO not installed, skipping LIBERO APIs")

register_api("FrankaControlPrivilegedApi", FrankaControlPrivilegedApi)
register_api("FrankaControlApi", lambda env: FrankaControlApi(env, use_sam3=True))
register_api("FrankaControlApiReduced", FrankaControlApiReduced)
register_api(
    "FrankaControlApiReducedBimanual", lambda env: FrankaControlApiReduced(env, bimanual=True)
)
register_api(
    "FrankaControlApiReducedExamplelessBimanual", lambda env: FrankaControlApiReducedExampleless(env, bimanual=True)
)
register_api(
    "FrankaControlApiReducedBimanualHandover", lambda env: FrankaControlApiReduced(env, bimanual=True, is_handover=True)
)
register_api(
    "FrankaControlApiReducedExamplelessBimanualHandover", lambda env: FrankaControlApiReducedExampleless(env, bimanual=True, is_handover=True)
)
register_api(
    "FrankaControlApiReducedSpillWipe",
    lambda env: FrankaControlApiReduced(env, tcp_offset=[0.0, 0.0, -0.0158]),
)
register_api("FrankaControlApiReducedExampleless", FrankaControlApiReducedExampleless)

register_api("FrankaControlApiReducedSkillLibrary", FrankaControlApiReducedSkillLibrary)
register_api(
    "FrankaControlApiReducedSkillLibraryBimanual",
    lambda env: FrankaControlApiReducedSkillLibrary(env, bimanual=True),
)
register_api(
    "FrankaControlApiReducedSkillLibrarySpillWipe",
    lambda env: FrankaControlApiReducedSkillLibrary(env, tcp_offset=[0.0, 0.0, -0.0158]),
)
register_api(
    "FrankaControlApiReducedSkillLibraryBimanualHandover",
    lambda env: FrankaControlApiReducedSkillLibrary(env, bimanual=True, is_handover=True)
)

# For spill wipe environment, we use a custom tcp offset of -0.0158m since panda end effector has been modified by robosuite to have the sponge attachement
register_api(
    "FrankaControlSpillWipeApi",
    lambda env: FrankaControlSpillWipeApi(env, tcp_offset=[0.0, 0.0, -0.0158], use_sam3=True),
)
register_api(
    "FrankaControlSpillWipeApiReduced",
    lambda env: FrankaControlApiReduced(
        env, tcp_offset=[0.0, 0.0, -0.0158], is_spill_wipe=True
    ),
)
register_api(
    "FrankaControlSpillWipePrivilegedApi",
    lambda env: FrankaControlSpillWipePrivilegedApi(env, tcp_offset=[0.0, 0.0, -0.0158]),
)
register_api(
    "FrankaControlSpillWipeApiReducedExampleless",
    lambda env: FrankaControlApiReducedExampleless(
        env, tcp_offset=[0.0, 0.0, -0.0158], is_spill_wipe=True
    ),
)

register_api("FrankaHandoverPrivilegedApi", FrankaHandoverPrivilegedApi)
register_api("FrankaHandoverApi", FrankaHandoverApi)
register_api("FrankaHandoverApiReduced", FrankaHandoverApiReduced)
register_api("FrankaHandoverApiReducedExampleless", FrankaHandoverApiReducedExampleless)

register_api("FrankaTwoArmLiftApi", FrankaTwoArmLiftApi)
register_api("FrankaTwoArmLiftPrivilegedApi", FrankaTwoArmLiftPrivilegedApi)
register_api(
    "FrankaTwoArmLiftApiReduced", 
    lambda env: FrankaControlApiReduced(env, bimanual=True, use_sam3=False),
)
register_api(
    "FrankaTwoArmLiftApiReducedExampleless",
    lambda env: FrankaControlApiReducedExampleless(env, bimanual=True, use_sam3=False),
)

register_api("FrankaControlNutAssemblyPrivilegedApi", FrankaControlNutAssemblyPrivilegedApi)
register_api("FrankaControlNutAssemblyVisualApi", FrankaControlNutAssemblyVisualApi)
register_api(
    "FrankaControlNutAssemblyApiReduced",
    lambda env: FrankaControlApiReduced(env, is_peg_assembly=True),
)
register_api(
    "FrankaControlNutAssemblyApiReducedExampleless",
    lambda env: FrankaControlApiReducedExampleless(env, is_peg_assembly=True),
)
# Register multi-turn variant with multi_turn=True
register_api(
    "FrankaControlMultiPrivilegedApi",
    lambda env: FrankaControlPrivilegedApi(env, multi_turn=True),
)

register_api("FrankaRealReducedSkillLibraryControlApi", lambda env: FrankaControlApiReducedSkillLibrary(env, tcp_offset=[0.0, 0.0, -0.157], real = True))
register_api("FrankaRealControlApi", lambda env: FrankaControlApi(env, tcp_offset=[0.0, 0.0, -0.157], real = True))
register_api("G1RealControlApi", lambda env: G1RealControlApi(env, use_sam3=True))
register_api("G1LeftRealControlApi", lambda env: G1LeftRealControlApi(env, use_sam3=True))
register_api("G1CameraApi", lambda env: G1CameraApi(env))

try:
    from .r1pro.control import R1ProControlApi
    register_api("R1ProControlApi", lambda env: R1ProControlApi(env, use_sam3=True))
except ImportError:
    print("R1Pro not installed, skipping R1Pro APIs")

if _libero_available:
    register_api("FrankaLiberoPrivilegedApi", FrankaLiberoPrivilegedApi)
    register_api("FrankaLiberoApi", lambda env: FrankaLiberoApi(env, use_sam3=True))
    register_api("FrankaLiberoApiReduced", FrankaLiberoApiReduced)
    register_api("FrankaLiberoApiReducedSkillLibrary", FrankaLiberoApiReducedSkillLibrary)
