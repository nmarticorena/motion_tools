from numpy.typing import ArrayLike
import pinocchio as pin
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from importlib import resources
from pathlib import Path
import tempfile
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
        rec: rr.RecordingStream,
        urdf_path: str,
        name="",
        target_frame: str = "world",
    ):
        rec.log_file_from_path(urdf_path, entity_path_prefix=name)
        self.rec = rec
        self.urdf_path = urdf_path
        self.tree = rr.urdf.UrdfTree.from_file_path(urdf_path)
        self.rec.send_blueprint(get_blueprint(target_frame))
        self.robot = urdfpy.URDF.load(self.urdf_path)
        self.n_revolute_joints = self._get_total_joints()
        self.joint_names = self._get_joint_names()
        self.resolution_order = self._build_resolution_order()
        self.limits = self._build_limits()
        self.name = name

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
            clean_visual_path = [self.name + "/" + str(v) for v in visual_path]
            for v_path in clean_visual_path:
                self.rec.log(v_path, rr.Asset3D.from_fields(albedo_factor=color))

    def log(self, joint_pos):
        """Log from a positional array (existing behaviour)."""
        pos_dic = {name: pos for name, pos in zip(self.joint_names, joint_pos)}
        self._log_from_dict(pos_dic)

    def log_from_dict(self, joint_pos: dict[str, float]):
        """Log from a {urdf_joint_name: position} dict (e.g. from hand_state_to_urdf_map)."""
        # Only keep keys that are actuated joints this robot knows about
        pos_dic = {
            name: joint_pos[name] for name in self.joint_names if name in joint_pos
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
                self.rec.log("transforms", transform)

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

        self.rec.log(
            entity_path,
            rr.Transform3D(
                translation=pos,
                rotation=rr.Quaternion(xyzw=quat_xyzw),
                parent_frame=parent_frame,
                child_frame=child_frame,
            ),
        )

    def log_pin_transform(self, name: str, pose: pin.SE3, parent_frame="pelvis"):
        pos = pose.translation
        quat_xyzw = pin.Quaternion(pose.rotation).coeffs()
        self.log_transform(
            self.rec, name, pos, quat_xyzw, parent_frame=parent_frame, child_frame=name
        )

    def log_se3_transform(self, name: str, pose: sm.SE3):
        pos = pose.t
        quat_xyzw = smb.r2q(pose.R, order="xyzs")
        self.log_transform(
            self.rec, name, pos, quat_xyzw, parent_frame="pelvis", child_frame=name
        )

    @staticmethod
    def log_transform(
        rec: rr.RecordingStream,
        entity_path: str,
        pos: ArrayLike,
        quat_xyzw: ArrayLike,
        parent_frame: str,
        child_frame: str,
    ) -> None:
        pos = np.asarray(pos, dtype=float).reshape(3)
        quat_xyzw = np.asarray(quat_xyzw, dtype=float).reshape(4)

        rec.log(
            entity_path,
            rr.Transform3D(
                translation=pos,
                rotation=rr.Quaternion(xyzw=quat_xyzw),
                parent_frame=parent_frame,
                child_frame=child_frame,
            ),
            rr.TransformAxes3D(axis_length=0.1),
        )

    @classmethod
    def g1(cls, rec: rr.RecordingStream, name="", target_frame="world"):
        with resources.as_file(
            resources.files("motion_tools.assets") / "g1_29dof_no_hands.urdf"
        ) as p:
            return cls(rec, str(p), name=name, target_frame=target_frame)

    @classmethod
    def g1_debug(cls, rec: rr.RecordingStream, name="", target_frame="world"):
        cls.log_transform(
            rec,
            "transforms",
            [0, 0, 0],
            [0, 0, 0, 1],
            parent_frame=target_frame,
            child_frame="debug_pelvis",
        )
        with resources.as_file(
            resources.files("motion_tools.assets") / "g1_29dof_no_hands_debug.urdf"
        ) as p:
            robot = cls(rec, str(p), name=name, target_frame=target_frame)

        robot.apply_color([1.0, 0.0, 0.0, 0.5])
        return robot

    @classmethod
    def left_dfq_hand(cls, rec: rr.RecordingStream, name="", target_frame="world"):
        with resources.as_file(
            resources.files("motion_tools.assets") / "DFQ_left_hand.urdf"
        ) as p:
            return cls(rec, str(p), name=name, target_frame=target_frame)

    @classmethod
    def right_dfq_hand(cls, rec: rr.RecordingStream, name="", target_frame="world"):
        with resources.as_file(
            resources.files("motion_tools.assets") / "DFQ_right_hand.urdf"
        ) as p:
            return cls(rec, str(p), name=name, target_frame=target_frame)

    @classmethod
    def left_ftp_hand(cls, rec: rr.RecordingStream, name="", target_frame="world"):
        with resources.as_file(
            resources.files("motion_tools.assets") / "FTP_left_hand.urdf"
        ) as p:
            return cls(rec, str(p), name=name, target_frame=target_frame)

    @classmethod
    def right_ftp_hand(cls, rec: rr.RecordingStream, name="", target_frame="world"):
        with resources.as_file(
            resources.files("motion_tools.assets") / "FTP_right_hand.urdf"
        ) as p:
            return cls(rec, str(p), name=name, target_frame=target_frame)
    
    @classmethod
    def panda(cls, rec: rr.RecordingStream, name="", target_frame="world"):
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
        return cls(rec, resolved_urdf_path, name=name, target_frame=target_frame)


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
    rec = rr.RecordingStream("robot")

    rec.spawn()
    blueprint = rrb.Spatial3DView(
        spatial_information=rrb.SpatialInformation(target_frame="pelvis")
    )
    rec.send_blueprint(blueprint)
    robot = ReRunRobot(rec, "assets/g1_29dof_no_hands.urdf")
    rec2 = rr.RecordingStream("robot_2")
    rec2.spawn()

    robot = ReRunRobot(rec2, "assets/g1_29dof_no_hands.urdf")
