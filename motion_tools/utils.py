import pathlib
from motion_tools.robot_gui import ReRunRobot
from motion_tools.motion import Motion
import rerun as rr

import motion_tools

PATH = pathlib.Path(motion_tools.__file__).parent.parent


def iter_files(folder: pathlib.Path, pattern: str):
    yield from sorted(folder.rglob(pattern))


def play_one_file(
    robot: ReRunRobot, rec: rr.RecordingStream, motion: Motion, time: float = 0
) -> float:
    fps = float(motion.fps)
    dt = 1.0 / fps

    # Reset timeline if desired
    t = time

    dof_pos = motion.dof_pos
    root_pos = motion.root_pos
    root_rot = motion.root_rot

    # Basic sanity checks
    n = min(len(dof_pos), len(root_pos), len(root_rot))

    for i in range(n):
        rec.set_time("time", duration=t)
        t += dt

        q = dof_pos[i]
        pos = root_pos[i]
        rot = root_rot[i]

        robot.log(q)

        # Root pose (world -> pelvis). Use a separate entity path (e.g. "tf").
        robot.log_transform_named_frames(
            "transforms",
            pos,
            rot,
            parent_frame="world",
            child_frame="pelvis",
        )
    return t


def play_folder(
    path: str,
    robot_path: str = str(PATH / "assets/g1_29dof_no_hands.urdf"),
    pattern: str = "*.pkl",
    flatten: bool = False,
    **kwargs,
):
    if pathlib.Path(path).is_dir():
        folder = pathlib.Path(path).expanduser().resolve()
        files = list(iter_files(folder, pattern))
        if not files:
            raise SystemExit(f"No files matching *.pkl in {folder}")

        print(f"Found {len(files)} files matching pattern in {folder}:")
    else:
        files = [path]

    print(files)
    rec = rr.RecordingStream(path)
    rec.set_time("time", duration=0)
    t = 0

    for f in files:
        motion = Motion.load(f, **kwargs)
        if not flatten:
            rec = rr.RecordingStream(str(f))
            rec.set_time("time", duration=0)
            t = 0
        rec.spawn()
        robot = ReRunRobot(rec, robot_path)
        t = play_one_file(robot, rec, motion, t)
