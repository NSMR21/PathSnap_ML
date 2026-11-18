"""HMM route reconstruction with Viterbi decoding.

Hidden state
    At GPS fix t the vehicle is on one of that fix's candidate arcs
    (candidates.py). Every fix has >= 1 candidate (fallback), so the state
    space is never empty.

Emission
    log P(fix t | arc b) := w_e * log p_t(b), where p_t(b) is the tuned
    classifier's probability that b is the right arc for fix t.

Transition (in the spirit of Newson & Krumm 2009)
    For consecutive fixes, compare how far the vehicle would have to DRIVE
    between the two projected positions (along the network, shortest path)
    with how far the GPS actually MOVED (straight line):
        log T(a -> b) = -| d_route(a, b) - d_gps | / beta
    Same arc: d_route = distance along the arc. Small backwards moves on the
    same arc are GPS jitter and are scored the same way.
    No path within the search limit: log T = log_unreachable (large negative,
    not -inf), so a break is allowed when the evidence really demands it.

Gaps
    If the time between two fixes exceeds gap_split_s the trip is split there
    and each piece is decoded independently (after minutes without data the
    previous position says little about the next one).

Viterbi recurrence (log domain)
    delta_t(b) = w_e log p_t(b) + max_a [ delta_{t-1}(a) + log T(a -> b) ]
    psi_t(b)   = argmax_a [ ... ]          (back-pointer)
    The best final state is traced back through psi to give one arc per fix.

Expansion
    Consecutive decoded arcs that do not touch are joined with the shortest
    path between them, giving a complete drivable arc sequence for display.
"""
import heapq
from dataclasses import asdict, dataclass

import numpy as np

from .network import RoadNetwork

INF = float("inf")
LIMIT_BUCKETS = (300.0, 600.0, 1200.0, 2400.0, 4000.0)


@dataclass
class HMMParams:
    beta_m: float = 20.0
    emission_weight: float = 1.0
    log_unreachable: float = -40.0
    gap_split_s: float = 60.0

    def to_dict(self):
        return asdict(self)


def search_limit(d_gps: float) -> float:
    """Shortest-path search radius needed for one step, rounded up to a bucket so caches are reused."""
    need = 2.5 * d_gps + 150.0
    for b in LIMIT_BUCKETS:
        if need <= b:
            return b
    return LIMIT_BUCKETS[-1]


class RoadRouter:
    """Distance-limited Dijkstra on the trip's road graph, cached per start node."""

    def __init__(self, net: RoadNetwork, arc_ids=None):
        arcs = np.asarray(net.indexed_arcs if arc_ids is None else arc_ids, dtype=np.int64)
        arcs = arcs[np.argsort(net.src[arcs], kind="stable")]
        self.indptr = np.searchsorted(net.src[arcs], np.arange(net.n_nodes + 1)).tolist()
        self.nbr = net.dst[arcs].tolist()
        self.w = net.length[arcs].tolist()
        self.arc = arcs.tolist()
        self.src = net.src
        self._cache = {}

    def from_node(self, s: int, limit: float):
        hit = self._cache.get(s)
        if hit is not None and hit[0] >= limit:
            return hit[1], hit[2]
        indptr, nbr, w, arc = self.indptr, self.nbr, self.w, self.arc
        dist, pred = {s: 0.0}, {}
        heap = [(0.0, s)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for k in range(indptr[u], indptr[u + 1]):
                nd = d + w[k]
                if nd > limit:
                    continue
                v = nbr[k]
                if nd < dist.get(v, INF):
                    dist[v] = nd
                    pred[v] = arc[k]
                    heapq.heappush(heap, (nd, v))
        self._cache[s] = (limit, dist, pred)
        return dist, pred

    def path(self, s: int, t: int, limit: float = LIMIT_BUCKETS[-1]):
        """Arc ids of the shortest path s -> t, [] if s == t, None if not reachable within limit."""
        dist, pred = self.from_node(s, limit)
        if t not in dist:
            return None
        out, v = [], t
        while v != s:
            a = pred[v]
            out.append(a)
            v = int(self.src[a])
        return out[::-1]


class TripLattice:
    """Everything about one trip's HMM that does NOT depend on the parameters.

    Built once (shortest paths are the expensive part), then decode() can be
    called many times with different HMMParams - that is what makes tuning
    the decoder on 80 trips affordable.
    """

    def __init__(self, cand, fixes, net: RoadNetwork, router: RoadRouter = None):
        obs = cand["obs_idx"].to_numpy()
        n_fix = len(fixes)
        self.starts = np.searchsorted(obs, np.arange(n_fix))
        self.sizes = np.diff(np.r_[self.starts, len(obs)])
        if np.any(self.sizes == 0):
            raise ValueError("every fix needs at least one candidate")
        self.arc = cand["arc_id"].to_numpy()
        self.frac = cand["proj_frac"].to_numpy(np.float64)
        x, y, t = (fixes[c].to_numpy(np.float64) for c in ("x_m", "y_m", "t"))
        self.d_gps = np.r_[0.0, np.hypot(np.diff(x), np.diff(y))]
        self.dt = np.r_[0.0, np.diff(t)]
        self.net = net
        self.router = router or RoadRouter(net)
        self.route_dist = [None] * n_fix
        L, src, dst = net.length, net.src, net.dst
        for k in range(1, n_fix):
            a_sl = slice(self.starts[k - 1], self.starts[k - 1] + self.sizes[k - 1])
            b_sl = slice(self.starts[k], self.starts[k] + self.sizes[k])
            A, fa = self.arc[a_sl], self.frac[a_sl]
            B, fb = self.arc[b_sl], self.frac[b_sl]
            limit = search_limit(self.d_gps[k])
            D = np.full((len(A), len(B)), INF)
            b_src = src[B]
            tail_b = fb * L[B]
            for i, a in enumerate(A):
                dist, _ = self.router.from_node(int(dst[a]), limit)
                node_d = np.fromiter((dist.get(int(s), INF) for s in b_src), dtype=np.float64, count=len(B))
                D[i] = (1.0 - fa[i]) * L[a] + node_d + tail_b
                same = B == a
                if same.any():
                    D[i, same] = np.abs(fb[same] - fa[i]) * L[a]
            self.route_dist[k] = D

    # ------------------------------------------------------------------ decoding
    def decode(self, proba: np.ndarray, params: HMMParams):
        """Return (row index into the trip's candidate rows per fix, list of segment start fixes)."""
        logp = params.emission_weight * np.log(np.clip(proba, 1e-6, 1.0))
        n_fix = len(self.starts)
        choice = np.empty(n_fix, dtype=np.int64)
        back = [None] * n_fix
        seg_starts = [0]
        s0 = self.starts[0]
        score = logp[s0:s0 + self.sizes[0]].copy()
        for k in range(1, n_fix):
            st, n = self.starts[k], self.sizes[k]
            em = logp[st:st + n]
            if self.dt[k] > params.gap_split_s:
                self._trace(choice, back, score, seg_starts[-1], k - 1)
                seg_starts.append(k)
                score = em.copy()
                continue
            D = self.route_dist[k]
            log_t = np.where(np.isfinite(D), -np.abs(D - self.d_gps[k]) / params.beta_m, params.log_unreachable)
            total = score[:, None] + log_t
            arg = np.argmax(total, axis=0)
            score = total[arg, np.arange(n)] + em
            score -= score.max()  # keep numbers small; argmax is unchanged
            back[k] = arg
        self._trace(choice, back, score, seg_starts[-1], n_fix - 1)
        return self.starts + choice, seg_starts

    @staticmethod
    def _trace(choice, back, score, first, last):
        j = int(np.argmax(score))
        for k in range(last, first - 1, -1):
            choice[k] = j
            if k > first:
                j = int(back[k][j])

    def classifier_only(self, proba: np.ndarray):
        """Independent argmax per fix (the phase-1 decoding), as row indices."""
        return np.array([st + int(np.argmax(proba[st:st + n])) for st, n in zip(self.starts, self.sizes)])

    # ------------------------------------------------------------------ route expansion
    def expand(self, arcs_per_fix: np.ndarray, seg_starts):
        """Turn one-arc-per-fix into drivable chains: fill holes with shortest paths.

        Returns (list of arc lists - one per unbroken chain, number of unfillable breaks).
        """
        net, chains, breaks = self.net, [], 0
        bounds = list(seg_starts) + [len(arcs_per_fix)]
        for s, e in zip(bounds[:-1], bounds[1:]):
            cur = [int(arcs_per_fix[s])]
            for k in range(s + 1, e):
                a, b = int(arcs_per_fix[k - 1]), int(arcs_per_fix[k])
                if a == b:
                    continue
                if net.dst[a] == net.src[b]:
                    cur.append(b)
                    continue
                # same search radius as the decoder used for this step -> cached Dijkstra
                p = self.router.path(int(net.dst[a]), int(net.src[b]), search_limit(self.d_gps[k]))
                if p is None:
                    chains.append(cur)
                    cur = [b]
                    breaks += 1
                else:
                    cur.extend(p + [b])
            chains.append(cur)
        return chains, breaks
