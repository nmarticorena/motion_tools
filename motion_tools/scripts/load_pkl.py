import pickle as pkl
import tyro
from motion_tools.robot_gui import ReRunRobot
import rerun as rr

import motion_tools
import pathlib

PATH = pathlib.Path(motion_tools.__file__).parent.parent

def iter_files(folder: pathlib.Path, pattern: str):
    yield from sorted(folder.rglob(pattern))


def play_one_file(robot: ReRunRobot, rec: rr.RecordingStream, pkl_path: pathlib.Path):
    with open(pkl_path, "rb") as f:
        data = pkl.load(f)

    fps = float(data["fps"])
    dt = 1.0 / fps

    # Reset timeline if desired
    t = 0.0

    dof_pos = data["dof_pos"]
    root_pos = data["root_pos"]
    root_rot = data["root_rot"]

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

def main(
    folder_path:str,
    /,
    robot_path:str = str(PATH/"assets/g1_29dof_no_hands.urdf")
):
    if  pathlib.Path(folder_path).is_dir():
        folder = pathlib.Path(folder_path).expanduser().resolve()
        files = list(iter_files(folder, '*.pkl'))
        if not files:
            raise SystemExit(f"No files matching *.pkl in {folder}")

        print(f"Found {len(files)} files matching pattern in {folder}:")
    else:
        files = [folder_path]

    print(files)
    rec = rr.RecordingStream("motion")
    rec.set_time("time",duration = 0)
    rec.spawn()

    robot = ReRunRobot(rec, robot_path)

    for f in files:
        play_one_file(robot, rec, f)

def cli():
    args = tyro.cli(main)

if __name__ == "__main__":
    cli()

