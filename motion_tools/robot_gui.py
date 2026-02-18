from numpy.typing import ArrayLike
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from importlib import resources
import urdfpy


class ReRunRobot:
    def __init__(
        self,
        rec: rr.RecordingStream,
        urdf_path: str,
        name="",
        target_frame: str = "world",
    ):
        rec.log_file_from_path(urdf_path)
        self.rec = rec
        self.urdf_path = urdf_path
        self.tree = rr.urdf.UrdfTree.from_file_path(urdf_path, entity_path_prefix=name)
        self.rec.send_blueprint(get_blueprint(target_frame))
        self.rec.log("/", rr.CoordinateFrame(target_frame), static=True)
        self.robot = urdfpy.URDF.load(self.urdf_path)
        self.n_revolute_joints = self._get_total_joints()
        self.joint_names = self._get_joint_names()

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

    def log(self, joint_pos):
        pos_dic = {name: pos for name, pos in zip(self.joint_names, joint_pos)}
        revolute_joints_idx = 0
        for joint, urdf_joint in zip(self.tree.joints(), self.robot.joints):
            assert joint.name == urdf_joint.name
            if joint.joint_type == "revolute":
                if joint.name in self.robot.actuated_joint_names:
                    angle = pos_dic[joint.name]
                else:
                    angle = (
                        urdf_joint.mimic.multiplier * pos_dic[urdf_joint.mimic.joint]
                        + urdf_joint.mimic.offset
                    )
                revolute_joints_idx += 1
                transform = joint.compute_transform(angle)
                self.rec.log("transforms", transform)

    def log_transform_named_frames(
        self,
        entity_path: str,
        pos: ArrayLike,
        quat_xyzw: ArrayLike,
        *,
        parent_frame: str,
        child_frame: str,
    ) -> None:
        """
        Logs a transform with explicit named frames (like ROS TF, but you can log it anywhere).
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

    @classmethod
    def g1(cls, rec: rr.RecordingStream, name="", target_frame="world"):
        with resources.as_file(
            resources.files("motion_tools.assets") / "g1_29dof_no_hands.urdf"
        ) as p:
            return cls(rec, str(p), name=name)

    @classmethod
    def left_dfq_hand(cls, rec: rr.RecordingStream, name="", target_frame="world"):
        with resources.as_file(
            resources.files("motion_tools.assets") / "DFQ_left_hand.urdf"
        ) as p:
            return cls(rec, str(p), name=name, target_frame=target_frame)


def get_blueprint(target_frame: str) -> rrb.Blueprint:
    blueprint = rrb.Spatial3DView(
        spatial_information=rrb.SpatialInformation(target_frame=target_frame)
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
