"""Readers for the raw trip files.

File formats (checked by hand, see outputs/eda):
  <id>.track  lon  lat  t_seconds          (float, whitespace separated, no header)
  <id>.nodes  lon  lat                     (row index = node id)
  <id>.arcs   from_node  to_node           (row index = arc id, directed)
  <id>.route  arc_id                       (ordered reference route, arc level)
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import config


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: np.ndarray            # (N, 3) lon, lat, t
    nodes: np.ndarray            # (M, 2) lon, lat
    arcs: np.ndarray             # (A, 2) from, to   (int64)
    route: Optional[np.ndarray]  # (R,) arc ids, None when not supplied

    @property
    def lon(self):
        return self.track[:, 0]

    @property
    def lat(self):
        return self.track[:, 1]

    @property
    def t(self):
        return self.track[:, 2]


def _read_table(path: Path, dtype) -> np.ndarray:
    # pandas' C parser is a lot faster than np.loadtxt on the 100k+ line node files
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="c", dtype=dtype)
    return df.to_numpy()


def partition_dir(partition: str) -> Path:
    if partition not in config.PARTITIONS:
        raise KeyError(f"unknown partition {partition!r}")
    return config.PARTITIONS[partition]


def list_trips(partition: str) -> list[str]:
    root = partition_dir(partition)
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def trip_files(partition: str, trip_id: str) -> dict[str, Path]:
    base = partition_dir(partition) / trip_id
    return {ext: base / f"{trip_id}.{ext}" for ext in ("track", "nodes", "arcs", "route")}


def files_in(folder: Path, trip_id: str) -> dict[str, Path]:
    base = Path(folder) / trip_id
    return {ext: base / f"{trip_id}.{ext}" for ext in ("track", "nodes", "arcs", "route")}


def load_trip_dir(folder: Path, trip_id: str, with_route: bool = False, partition: str = "external") -> Trip:
    """Load a trip from any folder laid out like the handout data (<folder>/<id>/<id>.<ext>).

    with_route defaults to False on purpose: the prediction path (API) never
    needs the reference route, it is read separately and only for scoring.
    """
    files = files_in(folder, trip_id)
    track = _read_table(files["track"], np.float64)
    nodes = _read_table(files["nodes"], np.float64)
    arcs = _read_table(files["arcs"], np.int64)
    route = read_route(folder, trip_id) if with_route else None
    return Trip(trip_id, partition, track, nodes, arcs, route)


def read_route(folder: Path, trip_id: str):
    path = files_in(folder, trip_id)["route"]
    return _read_table(path, np.int64).ravel() if path.exists() else None


def list_trips_in(folder: Path) -> list[str]:
    folder = Path(folder)
    return sorted(p.name for p in folder.iterdir() if p.is_dir() and (p / f"{p.name}.track").exists())


def load_trip(partition: str, trip_id: str, with_route: bool = True) -> Trip:
    return load_trip_dir(partition_dir(partition), trip_id, with_route=with_route, partition=partition)


def iter_trips(partition: str, with_route: bool = True):
    for tid in list_trips(partition):
        yield load_trip(partition, tid, with_route=with_route)
