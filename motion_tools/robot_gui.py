from numpy.typing import ArrayLike
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

class ReRunRobot():
    def __init__(self,rec:rr.RecordingStream, urdf_path:str, name = ""):
        rec.log_file_from_path(urdf_path)
        self.rec = rec
        self.tree = rr.urdf.UrdfTree.from_file_path(urdf_path, entity_path_prefix=name)
        self.rec.send_blueprint(get_blueprint())
        self.rec.log("/", rr.CoordinateFrame("world"), static=True)

    def log(self, joint_pos):
        revolute_joints_idx = 0
        for joint in self.tree.joints():
            if joint.joint_type == "revolute":
                angle = joint_pos[revolute_joints_idx]
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
        Logs a transform with explicit named frames (like ROS TF, but you can log it anywhere). :contentReference[oaicite:4]{index=4}
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

def get_blueprint():
    blueprint = rrb.Spatial3DView(
        spatial_information=rrb.SpatialInformation(
            target_frame="world"
        )
    )
    return blueprint


if __name__ == "__main__":
    rec = rr.RecordingStream("robot")

    rec.spawn()
    blueprint = rrb.Spatial3DView(
        spatial_information=rrb.SpatialInformation(
            target_frame="pelvis"
        )
    )
    rec.send_blueprint(blueprint)
    robot = ReRunRobot(rec, "assets/g1_29dof_no_hands.urdf")
    rec2 = rr.RecordingStream("robot_2")
    rec2.spawn()

    robot = ReRunRobot(rec2, "assets/g1_29dof_no_hands.urdf")
