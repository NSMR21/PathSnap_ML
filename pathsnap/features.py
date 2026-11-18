"""Feature engineering.

One row per (GPS fix, candidate arc). The classifier answers "is this the arc
the vehicle was on at this fix?" and its probability is later reused as the
HMM emission score, so every feature must be computable without the route.

Feature groups
--------------
distance   how far the fix is from the arc, absolutely and relative to the
           other candidates of the same fix
heading    does the direction of travel agree with the arc direction?
           (the only way to tell the two directions of a two-way street apart)
motion     speed / turning / sampling gaps of the fix itself - tells the model
           when the heading or distance evidence can be trusted
topology   arc length, node degrees, two-way flag
support    do the neighbouring fixes also sit close to this arc? A single noisy
           fix may be nearest to a side street, its neighbours usually are not.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from . import config
from .candidates import CandidateSet, generate_candidates
from .geo import LocalProjector, bearing_gap, compass_bearing, point_to_segment
from .labels import align_fixes_to_route
from .network import RoadNetwork
from .trip_io import Trip

FEATURE_COLUMNS = [
    # distance
    "dist_m", "dist_rank", "dist_gap_m", "dist_ratio", "n_cands",
    # heading
    "head_cos", "head_gap_deg", "align_score", "align_rank",
    # motion
    "speed_mps", "turn_deg", "dt_prev_s", "jump_mps",
    # topology / position on the arc
    "arc_len_m", "proj_frac", "at_arc_end", "tail_in_deg", "head_out_deg", "two_way",
    # neighbourhood support
    "support_m", "support_gap_m",
]


@dataclass
class TripTables:
    candidates: pd.DataFrame   # one row per fix x candidate
    fixes: pd.DataFrame        # one row per fix
    network: RoadNetwork
    projector: LocalProjector


def motion_profile(x, y, t, half_window=config.HEADING_HALF_WINDOW) -> dict:
    """Heading / speed from a centred difference over +-half_window fixes."""
    n = len(x)
    i = np.arange(n)
    lo = np.clip(i - half_window, 0, n - 1)
    hi = np.clip(i + half_window, 0, n - 1)
    dx, dy, dt = x[hi] - x[lo], y[hi] - y[lo], t[hi] - t[lo]
    span = np.hypot(dx, dy)
    speed = np.where(dt > 0, span / np.where(dt > 0, dt, 1), 0.0)
    heading = compass_bearing(dx, dy)

    # turning: incoming direction (lo -> i) vs outgoing (i -> hi)
    h_in = compass_bearing(x - x[lo], y - y[lo])
    h_out = compass_bearing(x[hi] - x, y[hi] - y)
    turn = bearing_gap(h_in, h_out)
    turn[(i - lo == 0) | (hi - i == 0)] = 0.0  # ends of the track have no turn info

    dt_prev = np.diff(t, prepend=t[0])
    step = np.hypot(np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0]))
    jump = np.where(dt_prev > 0, step / np.where(dt_prev > 0, dt_prev, 1), 0.0)
    return {"speed": speed, "heading": heading, "turn": turn, "dt_prev": dt_prev, "jump": jump}


def _group_starts(obs_sorted):
    starts = np.flatnonzero(np.r_[True, obs_sorted[1:] != obs_sorted[:-1]])
    sizes = np.diff(np.r_[starts, obs_sorted.size])
    return starts, sizes


def candidate_features(px, py, t, net: RoadNetwork, cs: CandidateSet) -> pd.DataFrame:
    mot = motion_profile(px, py, t)
    o, a = cs.obs, cs.arc
    starts, sizes = _group_starts(o)

    d = cs.dist
    d_min = np.repeat(np.minimum.reduceat(d, starts), sizes)
    n_c = np.repeat(sizes, sizes)

    gap = bearing_gap(mot["heading"][o], net.bearing[a])
    head_cos = np.cos(np.radians(gap))
    # heading is noise when the car is (almost) standing still -> fade it out below ~4 m/s
    trust = np.clip(mot["speed"][o] / 4.0, 0.0, 1.0)
    align = head_cos * trust
    order = np.lexsort((-align, o))
    align_rank = np.empty(o.size, dtype=np.int64)
    align_rank[order] = np.arange(o.size) - np.repeat(starts, sizes)

    # neighbourhood support: mean distance of nearby fixes to this same arc
    n_fix = len(px)
    sup = np.zeros(o.size)
    for off in config.SUPPORT_OFFSETS:
        j = np.clip(o + off, 0, n_fix - 1)
        dj, _ = point_to_segment(px[j], py[j], net.ax[a], net.ay[a], net.bx[a], net.by[a])
        sup += np.minimum(dj, config.SUPPORT_CLIP_M)
    sup /= len(config.SUPPORT_OFFSETS)
    sup_min = np.repeat(np.minimum.reduceat(sup, starts), sizes)

    return pd.DataFrame({
        "obs_idx": o,
        "arc_id": a,
        "arc_src": net.src[a],
        "arc_dst": net.dst[a],
        "is_fallback": cs.fallback,
        "dist_m": d,
        "dist_rank": cs.rank,
        "dist_gap_m": d - d_min,
        "dist_ratio": (d + 1.0) / (d_min + 1.0),
        "n_cands": n_c,
        "head_cos": head_cos,
        "head_gap_deg": gap,
        "align_score": align,
        "align_rank": align_rank,
        "speed_mps": mot["speed"][o],
        "turn_deg": mot["turn"][o],
        "dt_prev_s": mot["dt_prev"][o],
        "jump_mps": mot["jump"][o],
        "arc_len_m": net.length[a],
        "proj_frac": cs.frac,
        "at_arc_end": ((cs.frac <= 0.0) | (cs.frac >= 1.0)).astype(np.int8),
        "tail_in_deg": net.in_deg[net.src[a]],
        "head_out_deg": net.out_deg[net.dst[a]],
        "two_way": net.two_way[a].astype(np.int8),
        "support_m": sup,
        "support_gap_m": sup - sup_min,
    })


def build_trip_tables(trip: Trip, radius=config.CANDIDATE_RADIUS_M, k_max=config.MAX_CANDIDATES,
                      label_max_dist=config.LABEL_MAX_DIST_M, use_route: Optional[bool] = None) -> TripTables:
    """Full per-trip processing: project -> index -> candidates -> features (-> labels)."""
    proj = LocalProjector.around(trip.lon, trip.lat)
    px, py = proj.forward(trip.lon, trip.lat)
    net = RoadNetwork(trip.nodes, trip.arcs, proj).build_index(px, py)
    cs = generate_candidates(px, py, net, radius=radius, k_max=k_max)
    cand = candidate_features(px, py, trip.t, net, cs)
    cand.insert(0, "trip_id", trip.trip_id)

    fixes = pd.DataFrame({
        "trip_id": trip.trip_id,
        "obs_idx": np.arange(len(px)),
        "lon": trip.lon, "lat": trip.lat, "t": trip.t,
        "x_m": px, "y_m": py,
    })
    counts = np.bincount(cs.obs, minlength=len(px))
    fixes["n_cands"] = counts

    if use_route is None:
        use_route = trip.route is not None
    if use_route and trip.route is not None:
        pos, lab_arc, lab_dist = align_fixes_to_route(px, py, net, trip.route)
        fixes["route_pos"] = pos
        fixes["label_arc"] = lab_arc
        fixes["label_dist_m"] = lab_dist
        fixes["label_ok"] = lab_dist <= label_max_dist
        cand["is_match"] = (cand["arc_id"].to_numpy() == lab_arc[cand["obs_idx"].to_numpy()]).astype(np.int8)
        hit = np.bincount(cand["obs_idx"].to_numpy(), weights=cand["is_match"].to_numpy(), minlength=len(px))
        fixes["label_in_cands"] = hit > 0
        cand["label_ok"] = fixes["label_ok"].to_numpy()[cand["obs_idx"].to_numpy()]

    return TripTables(cand, fixes, net, proj)
