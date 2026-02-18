from motion_tools.robot_gui import ReRunRobot
import rerun as rr
import numpy as np


if __name__ == "__main__":
    rec = rr.RecordingStream("robot")
    rec.spawn()
    hand = ReRunRobot.left_dfq_hand(rec, target_frame="left_wrist_yaw_link")

    import urdfpy

    robot = urdfpy.URDF.load(hand.urdf_path)
    actued_joints = robot.actuated_joints

    for i in np.linspace(0, 1, 1000):
        hand.log(i * np.ones(50))
