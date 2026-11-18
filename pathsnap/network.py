"""Road network wrapper: projected arc geometry, degrees and a spatial index."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from . import config
from .geo import LocalProjector, compass_bearing


class RoadNetwork:
    """Directed road graph of one trip, with arcs as straight segments in metres.

    Arc ids are the row numbers of the .arcs file, node ids the row numbers of
    the .nodes file - both are kept unchanged so predictions can be written
    back in the same id space as the .route file.
    """

    def __init__(self, nodes_lonlat: np.ndarray, arcs: np.ndarray, projector: LocalProjector):
        self.projector = projector
        self.n_nodes = len(nodes_lonlat)
        self.n_arcs = len(arcs)
        self.src = arcs[:, 0].astype(np.int64)
        self.dst = arcs[:, 1].astype(np.int64)

        nx, ny = projector.forward(nodes_lonlat[:, 0], nodes_lonlat[:, 1])
        self.node_x, self.node_y = nx, ny
        self.ax, self.ay = nx[self.src], ny[self.src]
        self.bx, self.by = nx[self.dst], ny[self.dst]
        self.length = np.hypot(self.bx - self.ax, self.by - self.ay)
        self.bearing = compass_bearing(self.bx - self.ax, self.by - self.ay)

        self.out_deg = np.bincount(self.src, minlength=self.n_nodes)
        self.in_deg = np.bincount(self.dst, minlength=self.n_nodes)

        # an arc is "two way" if the reversed arc also exists
        key = self.src * self.n_nodes + self.dst
        rev = self.dst * self.n_nodes + self.src
        self.two_way = np.isin(rev, key)

        self._tree = None
        self._sample_arc = None
        self._step = None

    # ------------------------------------------------------------------ graph
    def follows(self, a, b):
        """True where arc b can be driven directly after arc a (a.dst == b.src)."""
        return self.dst[np.asarray(a)] == self.src[np.asarray(b)]

    def component_stats(self) -> dict:
        """Weak / strong connectivity summary (used in the EDA)."""
        n = self.n_nodes
        g = coo_matrix((np.ones(self.n_arcs, dtype=np.int8), (self.src, self.dst)), shape=(n, n)).tocsr()
        n_weak, weak_lab = connected_components(g, directed=True, connection="weak")
        n_strong, strong_lab = connected_components(g, directed=True, connection="strong")
        isolated = int(np.sum((self.in_deg + self.out_deg) == 0))
        weak_sizes = np.bincount(weak_lab)
        strong_sizes = np.bincount(strong_lab)
        return {
            "n_weak_components": int(n_weak),
            "largest_weak_frac": float(weak_sizes.max() / n),
            "n_strong_components": int(n_strong),
            "largest_strong_frac": float(strong_sizes.max() / n),
            "isolated_nodes": isolated,
        }

    # ------------------------------------------------------------------ spatial index
    def build_index(self, focus_x=None, focus_y=None,
                    margin=config.INDEX_MARGIN_M, step=config.INDEX_STEP_M):
        """Index arcs by points sampled every <= `step` metres along them.

        If focus coordinates are given (the GPS fixes) only arcs near their
        bounding box are indexed - the big trips carry 100k+ arcs that are
        nowhere near the vehicle.
        """
        keep = np.ones(self.n_arcs, dtype=bool)
        if focus_x is not None:
            x0, x1 = np.min(focus_x) - margin, np.max(focus_x) + margin
            y0, y1 = np.min(focus_y) - margin, np.max(focus_y) + margin
            lo_x = np.minimum(self.ax, self.bx)
            hi_x = np.maximum(self.ax, self.bx)
            lo_y = np.minimum(self.ay, self.by)
            hi_y = np.maximum(self.ay, self.by)
            keep = (hi_x >= x0) & (lo_x <= x1) & (hi_y >= y0) & (lo_y <= y1)
        arc_ids = np.flatnonzero(keep)

        pieces = np.maximum(1, np.ceil(self.length[arc_ids] / step)).astype(np.int64)
        counts = pieces + 1  # include both end points
        owner = np.repeat(arc_ids, counts)
        # fraction along the arc for every sample: 0, 1/p, ..., 1
        start = np.repeat(np.cumsum(counts) - counts, counts)
        k = np.arange(owner.size) - start
        frac = k / np.repeat(pieces, counts)
        sx = self.ax[owner] + frac * (self.bx[owner] - self.ax[owner])
        sy = self.ay[owner] + frac * (self.by[owner] - self.ay[owner])

        self._tree = cKDTree(np.column_stack([sx, sy]))
        self._sample_arc = owner
        self._step = step
        self.indexed_arcs = arc_ids
        return self

    def arcs_within(self, px, py, radius):
        """For each point, the unique arc ids that *may* lie within `radius` (superset)."""
        if self._tree is None:
            raise RuntimeError("call build_index() first")
        hits = self._tree.query_ball_point(np.column_stack([px, py]), r=radius + self._step / 2)
        return [np.unique(self._sample_arc[h]) if len(h) else np.empty(0, np.int64) for h in hits]

    def nearest_arcs(self, x, y, k_samples=40):
        """Arc ids around a single point regardless of distance (fallback search)."""
        k_samples = min(k_samples, len(self._sample_arc))
        _, idx = self._tree.query([x, y], k=k_samples)
        return np.unique(self._sample_arc[np.atleast_1d(idx)])
