"""Per-fix labels from the arc-level reference route.

The .route file lists the arcs the vehicle drove, in order, but says nothing
about *which GPS fix* belongs to which arc. To train a per-fix classifier we
need that link, so we derive it ourselves.

Approach: monotone alignment (same idea as dynamic time warping).
Fix i is assigned route position j(i) such that j is non-decreasing in time
and the summed fix-to-arc distance is minimal:

    C[i, j] = d(i, j) + min_{k <= j} C[i-1, k]

The monotone constraint is what makes this better than "nearest route arc":
when a route passes the same street twice (out and back, loops) the nearest
arc is ambiguous, but the time order is not. Plain distance (not squared) is
used so that a few wild GPS outliers cannot drag the whole alignment around.
"""
import numpy as np

from .geo import point_to_segment
from .network import RoadNetwork


def align_fixes_to_route(px, py, net: RoadNetwork, route: np.ndarray):
    """Return (route_pos, arc_id, dist_m) per fix."""
    n_fix = len(px)
    n_route = len(route)
    ax, ay = net.ax[route], net.ay[route]
    bx, by = net.bx[route], net.by[route]

    back = np.empty((n_fix, n_route), dtype=np.int16 if n_route < 32000 else np.int32)
    pos = np.arange(n_route)

    cost, _ = point_to_segment(px[0], py[0], ax, ay, bx, by)
    back[0] = pos
    for i in range(1, n_fix):
        run_min = np.minimum.accumulate(cost)
        # index where the running minimum was attained (latest on ties)
        run_arg = np.maximum.accumulate(np.where(cost <= run_min, pos, 0))
        d_i, _ = point_to_segment(px[i], py[i], ax, ay, bx, by)
        cost = run_min + d_i
        back[i] = run_arg

    path = np.empty(n_fix, dtype=np.int64)
    path[-1] = int(np.argmin(cost))
    for i in range(n_fix - 1, 0, -1):
        path[i - 1] = back[i, path[i]]

    arc = route[path]
    dist, _ = point_to_segment(px, py, net.ax[arc], net.ay[arc], net.bx[arc], net.by[arc])
    return path, arc, dist


def route_connectivity(net: RoadNetwork, route: np.ndarray) -> float:
    """Fraction of consecutive reference arcs that actually join head-to-tail."""
    if len(route) < 2:
        return 1.0
    return float(np.mean(net.follows(route[:-1], route[1:])))
