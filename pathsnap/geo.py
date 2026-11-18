"""Geometry helpers.

Coordinates are lon/lat in degrees. Distances in degrees are meaningless
(1 degree of longitude shrinks with cos(latitude)), so every trip is
projected to a local east/north plane in metres centred on the median GPS fix.

Why not UTM? The trips are spread over several countries and UTM zones; a
per-trip tangent plane needs no extra dependency and the only distances we
care about are short ones (fix -> nearby road, a few tens of metres), where
the relative scale error of the flat-earth approximation stays far below 1%.
"""
import numpy as np

EARTH_RADIUS_M = 6_371_008.8  # mean earth radius


class LocalProjector:
    """Equirectangular projection around (lon0, lat0) -> metres east (x) / north (y)."""

    def __init__(self, lon0: float, lat0: float):
        self.lon0 = float(lon0)
        self.lat0 = float(lat0)
        self.m_per_deg_lat = np.radians(1.0) * EARTH_RADIUS_M
        self.m_per_deg_lon = self.m_per_deg_lat * np.cos(np.radians(self.lat0))

    @classmethod
    def around(cls, lon, lat):
        return cls(np.median(lon), np.median(lat))

    def forward(self, lon, lat):
        x = (np.asarray(lon, dtype=np.float64) - self.lon0) * self.m_per_deg_lon
        y = (np.asarray(lat, dtype=np.float64) - self.lat0) * self.m_per_deg_lat
        return x, y

    def inverse(self, x, y):
        lon = np.asarray(x) / self.m_per_deg_lon + self.lon0
        lat = np.asarray(y) / self.m_per_deg_lat + self.lat0
        return lon, lat


def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance, used only as a sanity check against the projection."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def point_to_segment(px, py, ax, ay, bx, by):
    """Distance from point(s) P to segment(s) AB plus the clamped projection fraction.

    All arguments broadcast. Returns (dist_m, frac) with frac in [0, 1]
    (0 = closest to A, 1 = closest to B). Degenerate segments (A == B) give frac 0.
    """
    vx = bx - ax
    vy = by - ay
    seg_len2 = vx * vx + vy * vy
    safe = np.where(seg_len2 > 0, seg_len2, 1.0)
    frac = ((px - ax) * vx + (py - ay) * vy) / safe
    frac = np.clip(np.where(seg_len2 > 0, frac, 0.0), 0.0, 1.0)
    qx = ax + frac * vx
    qy = ay + frac * vy
    return np.hypot(px - qx, py - qy), frac


def compass_bearing(dx, dy):
    """Bearing in degrees, 0 = north, 90 = east (clockwise)."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def bearing_gap(a_deg, b_deg):
    """Smallest absolute angle between two bearings, in [0, 180]."""
    d = np.abs((np.asarray(a_deg) - np.asarray(b_deg)) % 360.0)
    return np.minimum(d, 360.0 - d)
