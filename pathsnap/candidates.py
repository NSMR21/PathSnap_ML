"""Candidate road segments for each GPS fix.

The classifier never sees the whole network - only a short list of arcs near
each fix. A good candidate set is small (cheap, easy for the model) but still
contains the true arc (otherwise no model can recover it). The radius /
cap trade-off is measured in scripts/02_build_dataset.py.
"""
from dataclasses import dataclass

import numpy as np

from . import config
from .geo import point_to_segment
from .network import RoadNetwork


@dataclass
class CandidateSet:
    obs: np.ndarray       # fix index, sorted ascending
    arc: np.ndarray       # arc id
    dist: np.ndarray      # perpendicular distance fix -> arc (m)
    frac: np.ndarray      # where on the arc the fix projects (0..1)
    rank: np.ndarray      # 0 = closest arc for that fix
    fallback: np.ndarray  # True where the fix had nothing inside the radius


def _rank_within(obs_sorted):
    first = np.searchsorted(obs_sorted, obs_sorted, side="left")
    return np.arange(obs_sorted.size) - first


def generate_candidates(px, py, net: RoadNetwork,
                        radius=config.CANDIDATE_RADIUS_M,
                        k_max=config.MAX_CANDIDATES,
                        k_fallback=config.FALLBACK_CANDIDATES) -> CandidateSet:
    n_fix = len(px)
    near = net.arcs_within(px, py, radius)
    sizes = np.fromiter((a.size for a in near), dtype=np.int64, count=n_fix)
    obs = np.repeat(np.arange(n_fix), sizes)
    arc = np.concatenate(near) if sizes.sum() else np.empty(0, np.int64)

    dist, frac = point_to_segment(px[obs], py[obs], net.ax[arc], net.ay[arc], net.bx[arc], net.by[arc])
    inside = dist <= radius
    obs, arc, dist, frac = obs[inside], arc[inside], dist[inside], frac[inside]

    # sort by fix, then distance (arc id breaks exact ties so results are reproducible)
    order = np.lexsort((arc, dist, obs))
    obs, arc, dist, frac = obs[order], arc[order], dist[order], frac[order]
    rank = _rank_within(obs)
    keep = rank < k_max
    obs, arc, dist, frac, rank = obs[keep], arc[keep], dist[keep], frac[keep], rank[keep]
    fallback = np.zeros(obs.size, dtype=bool)

    # fixes with an empty set: take the closest few arcs whatever their distance
    empty = np.setdiff1d(np.arange(n_fix), obs)
    if empty.size:
        extra = []
        for i in empty:
            cand = net.nearest_arcs(px[i], py[i])
            d, f = point_to_segment(px[i], py[i], net.ax[cand], net.ay[cand], net.bx[cand], net.by[cand])
            o = np.lexsort((cand, d))[:k_fallback]
            extra.append((np.full(o.size, i), cand[o], d[o], f[o], np.arange(o.size)))
        e_obs, e_arc, e_d, e_f, e_r = (np.concatenate(z) for z in zip(*extra))
        obs = np.concatenate([obs, e_obs])
        arc = np.concatenate([arc, e_arc])
        dist = np.concatenate([dist, e_d])
        frac = np.concatenate([frac, e_f])
        rank = np.concatenate([rank, e_r])
        fallback = np.concatenate([fallback, np.ones(e_obs.size, dtype=bool)])
        order = np.lexsort((rank, obs))
        obs, arc, dist, frac, rank, fallback = (v[order] for v in (obs, arc, dist, frac, rank, fallback))

    return CandidateSet(obs, arc, dist, frac, rank, fallback)
