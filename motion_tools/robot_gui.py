from numpy.typing import ArrayLike
import pinocchio as pin
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from importlib import resources
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
import urchin as urdfpy
import spatialmath as sm
import spatialmath.base as smb
import warnings


warnings.filterwarnings(
    "ignore", message=".*angle.*outside limits.*", category=UserWarning
)


class ReRunRobot:
    def __init__(
        self,
        urdf_path: str,
        name="",
        target_frame: str = "world",
    ):
        self.source_urdf_path = urdf_path
        self.target_frame = target_frame
        self.name = name
        self.link_name_map: dict[str, str] = {}
        self.joint_name_map: dict[str, str] = {}
        self.inverse_link_name_map: dict[str, str] = {}
        self.inverse_joint_name_map: dict[str, str] = {}
        self.urdf_path = urdf_path

        if self.name:
            (
                self.urdf_path,
                self.link_name_map,
                self.joint_name_map,
            ) = self._namespace_urdf(self.source_urdf_path, self.name)
            self.inverse_link_name_map = {
                namespaced: original
                for original, namespaced in self.link_name_map.items()
            }
            self.inverse_joint_name_map = {
                namespaced: original
                for original, namespaced in self.joint_name_map.items()
            }

        rr.log_file_from_path(self.urdf_path)
        self.target_frame = target_frame
        self.tree = rr.urdf.UrdfTree.from_file_path(self.urdf_path)
        # rr.send_blueprint(get_blueprint(target_frame))
        self.robot = urdfpy.URDF.load(self.urdf_path)
        self.n_revolute_joints = self._get_total_joints()
        self.joint_names = self._get_joint_names()
        self.resolution_order = self._build_resolution_order()
        self.limits = self._build_limits()

    def add_copy(self, name: str, target_frame: str | None = None) -> "ReRunRobot":
        return type(self)(
            self.source_urdf_path,
            name=name,
            target_frame=self.target_frame if target_frame is None else target_frame,
        )

    @staticmethod
    def _namespace_token(prefix: str, name: str) -> str:
        return f"{prefix}_{name}"

    def _resolve_link_name(self, name: str) -> str:
        return self.link_name_map.get(name, name)

    def _resolve_joint_name(self, name: str) -> str:
        return self.joint_name_map.get(name, name)

    def _input_joint_name(self, name: str) -> str:
        return self.inverse_joint_name_map.get(name, name)

    def _get_total_joints(self):
        n_revolute_joints = 0
        for joint in self.tree.joints():
            if joint.joint_type == "revolute":
                n_revolute_joints += 1
        return n_revolute_joints

    def _get_joint_names(self):
        joint_names = []
        for joint, urdf_joint in zip(self.tree.joints(), self.robot.joints):
            if joint.joint_type == "revolute" and urdf_joint.mimic is None:
                joint_names.append(joint.name)
        return joint_names

    def _build_resolution_order(self):
        """Returns joint names in dependency order (actuated first, then mimics)."""
        mimic_map = {j.name: j.mimic for j in self.robot.joints if j.mimic is not None}
        resolved = list(self.joint_names)  # actuated joints are already resolved
        resolved_set = set(resolved)
        order = []  # list of (joint_name, mimic_or_None)

        for name in resolved:
            order.append((name, None))

        unresolved = [(name, mimic_map[name]) for name in mimic_map]
        for _ in range(len(unresolved) + 1):
            next_unresolved = []
            for name, mimic in unresolved:
                if mimic.joint in resolved_set:
                    order.append((name, mimic))
                    resolved_set.add(name)
                else:
                    next_unresolved.append((name, mimic))
            unresolved = next_unresolved
            if not unresolved:
                break

        return order

    def _build_limits(self):
        limits = {}
        for joint, urdf_joint in zip(self.tree.joints(), self.robot.joints):
            if joint.joint_type == "revolute":
                limits[joint.name] = urdf_joint.limit
        return limits

    def apply_color(self, color: ArrayLike):
        for joint in self.tree.joints():
            link = self.tree.get_joint_child(joint)
            visual_path = self.tree.get_visual_geometry_paths(link)
            for v_path in visual_path:
                rr.log(str(v_path), rr.Asset3D.from_fields(albedo_factor=color))

    def log(self, joint_pos):
        """Log from a positional array (existing behaviour)."""
        pos_dic = {name: pos for name, pos in zip(self.joint_names, joint_pos)}
        self._log_from_dict(pos_dic)

    def log_from_dict(self, joint_pos: dict[str, float]):
        """Log from a {urdf_joint_name: position} dict (e.g. from hand_state_to_urdf_map)."""
        # Only keep keys that are actuated joints this robot knows about
        pos_dic = {
            name: joint_pos[key]
            for name in self.joint_names
            for key in (name, self._input_joint_name(name))
            if key in joint_pos
        }
        self._log_from_dict(pos_dic)

    def _log_from_dict(self, pos_dic: dict[str, float]):
        """Shared implementation: resolve mimics then log transforms."""
        for joint_name, mimic in self.resolution_order:
            if mimic is not None:
                pos_dic[joint_name] = (
                    mimic.multiplier * pos_dic[mimic.joint] + mimic.offset
                )
                pos_dic[joint_name] = np.clip(
                    pos_dic[joint_name],
                    self.limits[joint_name].lower,
                    self.limits[joint_name].upper,
                )

        for joint, urdf_joint in zip(self.tree.joints(), self.robot.joints):
            if joint.joint_type == "revolute":
                angle = pos_dic[joint.name]
                transform = joint.compute_transform(angle)
                rr.log("transforms", transform)

    def log_transform_named_frames(
        self,
        entity_path: str,
        pos: ArrayLike,
        quat_xyzw: ArrayLike,
        parent_frame: str,
        child_frame: str,
    ) -> None:
        """
        Logs a transform with explicit named frames like ros TF.
        """
        pos = np.asarray(pos, dtype=float).reshape(3)
        quat_xyzw = np.asarray(quat_xyzw, dtype=float).reshape(4)

        rr.log(
            entity_path,
            rr.Transform3D(
                translation=pos,
                rotation=rr.Quaternion(xyzw=quat_xyzw),
                parent_frame=self._resolve_link_name(parent_frame),
                child_frame=self._resolve_link_name(child_frame),
            ),
        )

    def log_pin_transform(self, name: str, pose: pin.SE3, parent_frame="pelvis"):
        pos = pose.translation
        quat_xyzw = pin.Quaternion(pose.rotation).coeffs()
        self.log_transform(
            self._resolve_link_name(name),
            pos,
            quat_xyzw,
            parent_frame=self._resolve_link_name(parent_frame),
            child_frame=self._resolve_link_name(name),
        )

    def log_se3_transform(self, name: str, pose: sm.SE3):
        pos = pose.t
        quat_xyzw = smb.r2q(pose.R, order="xyzs")
        self.log_transform(
            self._resolve_link_name(name),
            pos,
            quat_xyzw,
            parent_frame=self._resolve_link_name("pelvis"),
            child_frame=self._resolve_link_name(name),
        )

    @staticmethod
    def log_transform(
        entity_path: str,
        pos: ArrayLike,
        quat_xyzw: ArrayLike,
        parent_frame: str,
        child_frame: str,
    ) -> None:
        pos = np.asarray(pos, dtype=float).reshape(3)
        quat_xyzw = np.asarray(quat_xyzw, dtype=float).reshape(4)

        rr.log(
            entity_path,
            rr.Transform3D(
                translation=pos,
                rotation=rr.Quaternion(xyzw=quat_xyzw),
                parent_frame=parent_frame,
                child_frame=child_frame,
            ),
            rr.TransformAxes3D(axis_length=0.1),
        )

    @staticmethod
    def _resolve_package_uris_in_urdf(
        urdf_path: str, package_roots: dict[str, str]
    ) -> str:
        original_urdf_text = Path(urdf_path).read_text()
        urdf_text = original_urdf_text

        for package_name, package_root in package_roots.items():
            package_prefix = f"package://{package_name}/"
            resolved_prefix = Path(package_root).resolve().as_posix() + "/"
            urdf_text = urdf_text.replace(package_prefix, resolved_prefix)

        if urdf_text == original_urdf_text:
            return urdf_path

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".urdf",
            prefix=f"{Path(urdf_path).stem}_resolved_",
            delete=False,
        ) as resolved_urdf:
            resolved_urdf.write(urdf_text)
            return resolved_urdf.name

    @classmethod
    def _namespace_urdf(
        cls, urdf_path: str, prefix: str
    ) -> tuple[str, dict[str, str], dict[str, str]]:
        root = ET.parse(urdf_path).getroot()
        urdf_dir = Path(urdf_path).resolve().parent

        link_name_map = {}
        joint_name_map = {}

        for link in root.findall("link"):
            original_name = link.attrib["name"]
            namespaced_name = cls._namespace_token(prefix, original_name)
            link_name_map[original_name] = namespaced_name
            link.attrib["name"] = namespaced_name

        for joint in root.findall("joint"):
            original_name = joint.attrib["name"]
            namespaced_name = cls._namespace_token(prefix, original_name)
            joint_name_map[original_name] = namespaced_name
            joint.attrib["name"] = namespaced_name

            parent = joint.find("parent")
            if parent is not None and "link" in parent.attrib:
                parent.attrib["link"] = link_name_map[parent.attrib["link"]]

            child = joint.find("child")
            if child is not None and "link" in child.attrib:
                child.attrib["link"] = link_name_map[child.attrib["link"]]

            mimic = joint.find("mimic")
            if mimic is not None and "joint" in mimic.attrib:
                mimic.attrib["joint"] = joint_name_map.get(
                    mimic.attrib["joint"],
                    cls._namespace_token(prefix, mimic.attrib["joint"]),
                )

        for mesh in root.findall(".//mesh"):
            filename = mesh.attrib.get("filename")
            if not filename:
                continue

            mesh_path = Path(filename)
            if "://" in filename or mesh_path.is_absolute():
                continue

            mesh.attrib["filename"] = str((urdf_dir / mesh_path).resolve())

        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".urdf",
            prefix=f"{Path(urdf_path).stem}_{prefix}_",
            delete=False,
        ) as namespaced_urdf:
            ET.ElementTree(root).write(namespaced_urdf, encoding="utf-8")
            return namespaced_urdf.name, link_name_map, joint_name_map

    @classmethod
    def g1(cls, name="", target_frame="world"):
        with resources.as_file(
            resources.files("motion_tools.assets") / "g1_29dof_no_hands.urdf"
        ) as p:
            return cls(str(p), name=name, target_frame=target_frame)

    @classmethod
    def g1_debug(cls, name="", target_frame="world"):
        cls.log_transform(
            "transforms",
            [0, 0, 0],
            [0, 0, 0, 1],
            parent_frame=target_frame,
            child_frame=cls._namespace_token(name, "debug_pelvis")
            if name
            else "debug_pelvis",
        )
        with resources.as_file(
            resources.files("motion_tools.assets") / "g1_29dof_no_hands_debug.urdf"
        ) as p:
            robot = cls(str(p), name=name, target_frame=target_frame)

        robot.apply_color([1.0, 0.0, 0.0, 0.5])
        return robot

    @classmethod
    def left_dfq_hand(cls, name="", target_frame="world"):
        with resources.as_file(
            resources.files("motion_tools.assets") / "DFQ_left_hand.urdf"
        ) as p:
            return cls(str(p), name=name, target_frame=target_frame)

    @classmethod
    def right_dfq_hand(cls, name="", target_frame="world"):
        with resources.as_file(
            resources.files("motion_tools.assets") / "DFQ_right_hand.urdf"
        ) as p:
            return cls(str(p), name=name, target_frame=target_frame)

    @classmethod
    def left_ftp_hand(cls, name="", target_frame="world"):
        with resources.as_file(
            resources.files("motion_tools.assets") / "FTP_left_hand.urdf"
        ) as p:
            return cls(str(p), name=name, target_frame=target_frame)

    @classmethod
    def right_ftp_hand(cls, name="", target_frame="world"):
        with resources.as_file(
            resources.files("motion_tools.assets") / "FTP_right_hand.urdf"
        ) as p:
            return cls(str(p), name=name, target_frame=target_frame)
    
    @classmethod
    def panda(cls, name="", target_frame="world"):
        try:
            from robot_descriptions.panda_description import (
                REPOSITORY_PATH,
                URDF_PATH,
            )
        except ImportError as exc:
            raise ImportError(
                "ReRunRobot.panda requires the `robot_descriptions` package "
                "with `robot_descriptions.panda_description` available."
            ) from exc

        resolved_urdf_path = cls._resolve_package_uris_in_urdf(
            str(URDF_PATH),
            {"example-robot-data": str(REPOSITORY_PATH)},
        )
        return cls(resolved_urdf_path, name=name, target_frame=target_frame)


def get_blueprint(target_frame: str) -> rrb.Blueprint:
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    spatial_information=rrb.SpatialInformation(
                        target_frame=target_frame
                    ),
                    contents=["-/cameras/**", "-/plots/**", "/**"],
                ),
                rrb.Spatial2DView(contents="cameras/**"),
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(
                    contents="state/**"  # , axis_y=rrb.ScalarAxis(range=(0, 1))
                ),
                rrb.TimeSeriesView(contents="costs/**"),
                rrb.TimeSeriesView(contents="costs_norm/**"),
            ),
        ),
        collapse_panels=True,
    )
    return blueprint


if __name__ == "__main__":
    rr.init("robot")
    rr.spawn()
    blueprint = rrb.Spatial3DView(
        spatial_information=rrb.SpatialInformation(target_frame="pelvis")
    )
    rr.send_blueprint(blueprint)
    robot = ReRunRobot("assets/g1_29dof_no_hands.urdf")
