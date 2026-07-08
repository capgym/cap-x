from __future__ import annotations

import os
import copy
import contextlib
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import pyroki as pk  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pk = None  # type: ignore[assignment]

try:
    from robot_descriptions.loaders.yourdfpy import load_robot_description  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    load_robot_description = None  # type: ignore[assignment]

"""Per-process cached PyRoKI context to avoid repeated JIT recompiles.

This module constructs and memoizes the heavy PyRoKI objects (robot model and
collision model). Reusing the same Python objects across environment instances
within a worker process prevents JAX/XLA from retracing and recompiling the IK
solver on every env spawn, which can exhaust memory.
"""


@dataclass(frozen=True)
class PyrokiContext:
    """Holds shared PyRoKI objects that should be reused within a process."""

    robot: Any
    robot_coll: Any
    target_link_name: str


@lru_cache(maxsize=8)
def get_pyroki_context(
    robot_urdf_or_name: str = "panda_description",
    *,
    target_link_name: str = "panda_hand",
    root_link_name: str | None = None,
) -> PyrokiContext:
    """Return a cached PyRoKI context for the given robot and target link.

    Args:
        robot_urdf_or_name: Filesystem path to a URDF, or a name that
            `robot_descriptions` can resolve (e.g., "panda_description").
        target_link_name: End-effector link name.

    Returns:
        PyrokiContext with constructed `pk.Robot` and `pk.collision.RobotCollision`.

    Notes:
        - The returned objects are cached per-process; callers must not mutate
          them in ways that affect JAX-pytree structure.
    """
    if pk is None:
        raise RuntimeError("pyroki not installed; install with robotics extras")
    if load_robot_description is None:
        raise RuntimeError("robot_descriptions not available; install robotics extras")

    if os.path.exists(robot_urdf_or_name):
        import yourdfpy as urdfpy  # type: ignore

        if root_link_name is None:
            urdf = urdfpy.URDF.load(robot_urdf_or_name)
        else:
            urdf = _load_reduced_chain_urdf(
                robot_urdf_or_name,
                root_link_name=root_link_name,
                target_link_name=target_link_name,
            )
    else:
        if root_link_name is not None:
            raise ValueError("root_link_name is only supported for local URDF paths.")
        urdf = load_robot_description(robot_urdf_or_name)

    robot = pk.Robot.from_urdf(urdf)
    robot_coll = pk.collision.RobotCollision.from_urdf(urdf)
    return PyrokiContext(robot=robot, robot_coll=robot_coll, target_link_name=target_link_name)


def _load_reduced_chain_urdf(
    robot_urdf_path: str,
    *,
    root_link_name: str,
    target_link_name: str,
) -> Any:
    import yourdfpy as urdfpy  # type: ignore

    source_path = Path(robot_urdf_path).expanduser().resolve()
    source_tree = ET.parse(source_path)
    source_root = source_tree.getroot()

    link_by_name = {
        link.attrib["name"]: link
        for link in source_root.findall("link")
        if "name" in link.attrib
    }
    joint_by_child = {}
    for joint in source_root.findall("joint"):
        child = joint.find("child")
        if child is not None and "link" in child.attrib:
            joint_by_child[child.attrib["link"]] = joint

    if root_link_name not in link_by_name:
        raise ValueError(f"root_link_name {root_link_name!r} is not a link in {source_path}.")
    if target_link_name not in link_by_name:
        raise ValueError(f"target_link_name {target_link_name!r} is not a link in {source_path}.")

    chain_joints = []
    current_link = target_link_name
    while current_link != root_link_name:
        joint = joint_by_child.get(current_link)
        if joint is None:
            raise ValueError(
                f"Could not trace URDF chain from target {target_link_name!r} "
                f"back to root {root_link_name!r}."
            )
        parent = joint.find("parent")
        if parent is None or "link" not in parent.attrib:
            raise ValueError(f"Joint {joint.attrib.get('name', '<unnamed>')} has no parent link.")
        parent_link = parent.attrib["link"]
        chain_joints.append(joint)
        current_link = parent_link
    chain_joints.reverse()

    reduced_root = ET.Element("robot", {"name": f"{source_root.attrib.get('name', 'robot')}_reduced"})
    for material in source_root.findall("material"):
        reduced_root.append(copy.deepcopy(material))
    for link_name in _ordered_chain_links(root_link_name, chain_joints, target_link_name):
        link = copy.deepcopy(link_by_name[link_name])
        _absolutize_mesh_filenames(link, source_path.parent)
        reduced_root.append(link)
    for joint in chain_joints:
        joint_copy = copy.deepcopy(joint)
        _absolutize_mesh_filenames(joint_copy, source_path.parent)
        reduced_root.append(joint_copy)

    xml_text = ET.tostring(reduced_root, encoding="unicode")
    with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as f:
        f.write(xml_text)
        temp_path = f.name
    try:
        return urdfpy.URDF.load(temp_path)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temp_path)


def _ordered_chain_links(root_link_name: str, chain_joints: list[ET.Element], target_link_name: str) -> list[str]:
    names = [root_link_name]
    for joint in chain_joints:
        child = joint.find("child")
        if child is not None and "link" in child.attrib:
            names.append(child.attrib["link"])
    if names[-1] != target_link_name:
        raise ValueError(f"Reduced URDF chain does not end at {target_link_name!r}.")
    return names


def _absolutize_mesh_filenames(element: ET.Element, urdf_dir: Path) -> None:
    assets_dir = Path(__file__).resolve().parents[2] / "envs" / "assets" / "g1_description"
    for mesh in element.findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if not filename or filename.startswith("package://") or Path(filename).is_absolute():
            continue
        candidate = (urdf_dir / filename).resolve()
        if not candidate.exists():
            asset_candidate = (assets_dir / filename).resolve()
            if asset_candidate.exists():
                candidate = asset_candidate
        mesh.attrib["filename"] = str(candidate)


__all__ = ["PyrokiContext", "get_pyroki_context"]
