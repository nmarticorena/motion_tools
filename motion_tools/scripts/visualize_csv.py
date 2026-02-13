import tyro
from motion_tools.utils import play_folder
from dataclasses import dataclass
from typing import Optional


@dataclass
class PlayFolderCsvArgs:
    path: str
    robot_path: Optional[str] = None
    pattern: str = "*.csv"
    fps: float = 30.0
    flatten: bool = False


def cli():
    args = tyro.cli(PlayFolderCsvArgs)
    kwargs = dict(
        path=args.path,
        pattern=args.pattern,
        flatten=args.flatten,
        fps=args.fps,
    )

    if args.robot_path is not None:
        kwargs["robot_path"] = args.robot_path

    play_folder(**kwargs)


if __name__ == "__main__":
    cli()
