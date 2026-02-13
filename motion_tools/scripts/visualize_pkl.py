import tyro
from motion_tools.utils import play_folder
from dataclasses import dataclass
from typing import Optional


@dataclass
class PlayFolderPklArgs:
    path: str
    robot_path: Optional[str] = None
    pattern: str = "*.pkl"
    flatten: bool = False


def cli():
    args = tyro.cli(PlayFolderPklArgs)
    kwargs = dict(
        path=args.path,
        pattern=args.pattern,
        flatten=args.flatten,
    )

    if args.robot_path is not None:
        kwargs["robot_path"] = args.robot_path

    play_folder(**kwargs)


if __name__ == "__main__":
    cli()
