from dataclasses import dataclass
import pickle
from pathlib import Path
from numpy.typing import NDArray
import numpy as np


@dataclass
class Motion:
    """base class for motion data"""

    dof_pos: NDArray
    fps: float
    root_pos: NDArray
    root_rot: NDArray

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        fps: float,
        delimiter: str = ",",
    ) -> "Motion":
        """Load from a CSV following the mjlab format.

        Output format: [x, y, z, qx, qy, qz, qw, joint1, joint2, ...]
        Assumes CSV contains joint columns (and possibly others).
        """
        path = Path(path)

        data = np.genfromtxt(
            path,
            delimiter=delimiter,
            dtype=np.float32,
        )
        if data.ndim == 1:
            data = data[None, :]  # single row

        pos = data[:, :3]
        rot = data[:, 3:7]
        dof_pos = data[:, 7:]

        return cls(dof_pos=dof_pos, fps=fps, root_pos=pos, root_rot=rot)

    @classmethod
    def from_pkl(
        cls,
        path: str | Path,
    ) -> "Motion":
        """Load from a fkl from GMR"""
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)

        dof_pos = data["dof_pos"]
        fps = data["fps"]
        root_pos = data["root_pos"]
        root_rot = data["root_rot"]
        return cls(dof_pos=dof_pos, fps=fps, root_pos=root_pos, root_rot=root_rot)

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "Motion":
        path = Path(path)
        suf = path.suffix.lower()
        if suf == ".csv":
            return cls.from_csv(path, **kwargs)
        if suf in (".pkl", ".pickle"):
            return cls.from_pkl(path)
        raise ValueError(f"Unsupported format: {suf}")
