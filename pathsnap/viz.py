"""Folium map helpers shared by the EDA, the experiments and (later) the API/UI."""
import folium
import numpy as np
from branca.element import MacroElement, Template

from .trip_io import Trip


def base_map(trip: Trip, zoom_start=15) -> folium.Map:
    lat_c, lon_c = float(np.median(trip.lat)), float(np.median(trip.lon))
    m = folium.Map(location=[lat_c, lon_c], zoom_start=zoom_start, tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[trip.lat.min(), trip.lon.min()], [trip.lat.max(), trip.lon.max()]])
    return m


def _arc_lines(trip: Trip, arc_ids):
    a = trip.arcs[np.asarray(arc_ids, dtype=np.int64)]
    p, q = trip.nodes[a[:, 0]], trip.nodes[a[:, 1]]
    return [[[p[i, 1], p[i, 0]], [q[i, 1], q[i, 0]]] for i in range(len(a))]


def _chains(trip: Trip, arc_seq):
    """Join consecutive connected arcs into polylines (fewer, nicer map objects)."""
    seq = np.asarray(arc_seq, dtype=np.int64)
    if seq.size == 0:
        return []
    seq = seq[np.r_[True, seq[1:] != seq[:-1]]]
    out, cur = [], None
    for arc in seq:
        s, d = trip.arcs[arc]
        if cur is not None and cur[-1] == s:
            cur.append(d)
        else:
            if cur is not None:
                out.append(cur)
            cur = [s, d]
    out.append(cur)
    return [[[trip.nodes[n, 1], trip.nodes[n, 0]] for n in chain] for chain in out]


def arc_polylines(trip: Trip, arc_seq):
    """Public helper: arc sequence -> list of [[lat, lon], ...] polylines (used by the API)."""
    return [[[round(float(a), 6), round(float(b), 6)] for a, b in line] for line in _chains(trip, arc_seq)]


def add_network_layer(m, trip: Trip, arc_ids, name="Road network (nearby)", color="#7a7a7a", show=True):
    fg = folium.FeatureGroup(name=name, show=show)
    for line in _arc_lines(trip, arc_ids):
        folium.PolyLine(line, color=color, weight=1.5, opacity=0.6).add_to(fg)
    fg.add_to(m)
    return fg


def add_route_layer(m, trip: Trip, arc_seq, name, color, weight=5, opacity=0.85, dash=None):
    fg = folium.FeatureGroup(name=name)
    for line in _chains(trip, arc_seq):
        folium.PolyLine(line, color=color, weight=weight, opacity=opacity, dash_array=dash).add_to(fg)
    fg.add_to(m)
    return fg


def add_gps_layer(m, trip: Trip, name="Raw GPS fixes", color="#1f4fd6", max_points=3000, idx=None):
    fg = folium.FeatureGroup(name=name)
    n = len(trip.track)
    sel = np.arange(n) if idx is None else np.asarray(idx)
    if sel.size > max_points:
        sel = sel[:: int(np.ceil(sel.size / max_points))]
    for i in sel:
        folium.CircleMarker(
            [trip.lat[i], trip.lon[i]], radius=2.5, color=color, weight=0,
            fill=True, fill_opacity=0.8,
            tooltip=f"fix {i}  t={trip.t[i]:.0f}s",
        ).add_to(fg)
    folium.Marker([trip.lat[0], trip.lon[0]], tooltip="start",
                  icon=folium.Icon(color="green", icon="play")).add_to(fg)
    folium.Marker([trip.lat[-1], trip.lon[-1]], tooltip="end",
                  icon=folium.Icon(color="red", icon="stop")).add_to(fg)
    fg.add_to(m)
    return fg


def add_legend(m, entries, title="Layers"):
    """entries: list of (label, css colour)."""
    rows = "".join(
        f'<div><span style="display:inline-block;width:14px;height:4px;background:{c};'
        f'margin-right:6px;vertical-align:middle"></span>{lbl}</div>' for lbl, c in entries
    )
    html = f"""
    {{% macro html(this, kwargs) %}}
    <div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;
                padding:8px 10px;border:1px solid #999;border-radius:4px;font:12px sans-serif">
      <b>{title}</b>{rows}
    </div>
    {{% endmacro %}}"""
    el = MacroElement()
    el._template = Template(html)
    m.get_root().add_child(el)
    return m
