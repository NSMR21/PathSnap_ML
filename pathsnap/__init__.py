"""PathSnap - map-matching noisy GPS tracks onto a road network.

Package layout
--------------
config      paths and tunable constants (one place to change things)
trip_io     reading the raw .track / .nodes / .arcs / .route files
geo         local metric projection + point-to-segment geometry
network     road graph wrapper with a spatial index over arcs
labels      turning the arc-level reference route into per-fix labels
candidates  candidate road segments for every GPS fix
features    feature table (one row per fix x candidate arc)
metrics     classification + route-level evaluation helpers
viz         folium map helpers
"""

__version__ = "0.1.0"
