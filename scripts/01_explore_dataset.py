"""Step 1 - dataset inventory, data-quality checks, EDA plots and folium maps.

Partitions used: train (80 trips) and student_test (10 trips). Every output
row / plot carries the partition name so the report can say which data each
analysis came from.

Run:  python scripts/01_explore_dataset.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap import config, viz
from pathsnap.geo import LocalProjector, haversine_m
from pathsnap.labels import route_connectivity
from pathsnap.network import RoadNetwork
from pathsnap.trip_io import list_trips, load_trip, trip_files

OUT = config.EDA_DIR
OUT.mkdir(parents=True, exist_ok=True)
config.MAPS_DIR.mkdir(parents=True, exist_ok=True)


def inspect_trip(part, tid):
    files = trip_files(part, tid)
    trip = load_trip(part, tid)
    lon, lat, t = trip.lon, trip.lat, trip.t
    dt = np.diff(t)
    proj = LocalProjector.around(lon, lat)
    x, y = proj.forward(lon, lat)
    step = np.hypot(np.diff(x), np.diff(y))
    net = RoadNetwork(trip.nodes, trip.arcs, proj)
    comp = net.component_stats()
    route = trip.route

    row = {
        "partition": part, "trip_id": tid,
        "n_fixes": len(t),
        "duration_s": float(t[-1] - t[0]),
        "path_len_km": float(step.sum() / 1000),
        "straight_km": float(haversine_m(lon[0], lat[0], lon[-1], lat[-1]) / 1000),
        "lon_min": lon.min(), "lon_max": lon.max(), "lat_min": lat.min(), "lat_max": lat.max(),
        "extent_x_km": float(np.ptp(x) / 1000), "extent_y_km": float(np.ptp(y) / 1000),
        # missing / invalid
        "nan_track": int(np.isnan(trip.track).sum()),
        "nan_nodes": int(np.isnan(trip.nodes).sum()),
        "bad_coord": int(np.sum((np.abs(lon) > 180) | (np.abs(lat) > 90))),
        "dup_position": int(np.sum(step == 0)),
        "non_increasing_t": int(np.sum(dt <= 0)),
        # sampling
        "dt_median": float(np.median(dt)),
        "dt_max": float(dt.max()),
        "share_dt_1s": float(np.mean(dt == 1)),
        "gaps_gt5s": int(np.sum(dt > 5)),
        "gaps_gt60s": int(np.sum(dt > 60)),
        "max_step_speed_mps": float(np.max(step / np.maximum(dt, 1e-9))),
        # network
        "n_nodes": net.n_nodes, "n_arcs": net.n_arcs,
        "two_way_share": float(net.two_way.mean()),
        "arc_len_median_m": float(np.median(net.length)),
        "arc_len_max_m": float(net.length.max()),
        **comp,
        # track inside network bbox? (also confirms the lon/lat column order)
        "track_inside_net_bbox": bool(
            lon.min() >= trip.nodes[:, 0].min() and lon.max() <= trip.nodes[:, 0].max()
            and lat.min() >= trip.nodes[:, 1].min() and lat.max() <= trip.nodes[:, 1].max()
        ),
        "file_kb_track": files["track"].stat().st_size / 1024,
        "file_kb_nodes": files["nodes"].stat().st_size / 1024,
        "file_kb_arcs": files["arcs"].stat().st_size / 1024,
        "has_route": route is not None,
    }
    if route is not None:
        row.update({
            "route_len_arcs": len(route),
            "route_len_km": float(net.length[route].sum() / 1000),
            "route_ids_out_of_range": int(np.sum((route < 0) | (route >= net.n_arcs))),
            "route_connectivity": route_connectivity(net, route),
            "route_repeated_arcs": int(len(route) - len(np.unique(route))),
        })
    return row, trip, net, x, y, step, dt


def main():
    rows, dts, speeds, arc_lens = [], {}, {}, {}
    keep_for_plot = {}
    for part in config.PARTITIONS:
        dts[part], speeds[part], arc_lens[part] = [], [], []
        for tid in list_trips(part):
            row, trip, net, x, y, step, dt = inspect_trip(part, tid)
            rows.append(row)
            dts[part].append(dt)
            ok = dt > 0
            speeds[part].append(step[ok] / dt[ok])
            arc_lens[part].append(net.length)
            print(f"{part:>12} {tid}  fixes={row['n_fixes']:6d}  arcs={row['n_arcs']:7d}  "
                  f"max_gap={row['dt_max']:.0f}s  route_conn={row.get('route_connectivity', float('nan')):.3f}")
    inv = pd.DataFrame(rows)
    inv.to_csv(OUT / "inventory.csv", index=False)

    summary = inv.groupby("partition").agg(
        trips=("trip_id", "count"),
        fixes_total=("n_fixes", "sum"),
        fixes_median=("n_fixes", "median"),
        fixes_min=("n_fixes", "min"),
        fixes_max=("n_fixes", "max"),
        duration_h_total=("duration_s", lambda s: s.sum() / 3600),
        path_km_total=("path_len_km", "sum"),
        nodes_median=("n_nodes", "median"),
        arcs_median=("n_arcs", "median"),
        nan_values=("nan_track", "sum"),
        trips_with_gap_gt60s=("gaps_gt60s", lambda s: int((s > 0).sum())),
        route_arcs_median=("route_len_arcs", "median"),
    )
    summary.to_csv(OUT / "partition_summary.csv")
    print(summary.T)

    all_dt = {p: np.concatenate(v) for p, v in dts.items()}
    all_sp = {p: np.concatenate(v) for p, v in speeds.items()}
    all_al = {p: np.concatenate(v) for p, v in arc_lens.items()}
    colors = {"train": "#2b6cb0", "student_test": "#dd6b20"}

    # 1. fixes per trip
    fig, ax = plt.subplots(figsize=(7, 4))
    for p, g in inv.groupby("partition"):
        ax.hist(g["n_fixes"], bins=np.logspace(2.5, 4.5, 25), alpha=0.7, label=f"{p} ({len(g)} trips)", color=colors[p])
    ax.set_xscale("log"); ax.set_xlabel("GPS fixes per trip"); ax.set_ylabel("trips")
    ax.set_title("Trip length in fixes"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "fixes_per_trip.png", dpi=130); plt.close(fig)

    # 2. sampling intervals
    bins = [0, 1, 2, 5, 10, 60, 600, np.inf]
    labels = ["1s", "2s", "3-5s", "6-10s", "11-60s", "1-10min", ">10min"]
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.4
    for k, (p, d) in enumerate(all_dt.items()):
        cnt = pd.cut(d, bins=bins, labels=labels, right=True).value_counts().reindex(labels)
        ax.bar(np.arange(len(labels)) + (k - 0.5) * w, cnt.values / d.size * 100, width=w, label=p, color=colors[p])
    ax.set_yscale("log"); ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_ylabel("% of consecutive steps (log)"); ax.set_title("Time between consecutive GPS fixes")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "sampling_intervals.png", dpi=130); plt.close(fig)

    # 3. implied speed between fixes
    fig, ax = plt.subplots(figsize=(7, 4))
    for p, s in all_sp.items():
        ax.hist(np.clip(s, 0, 60), bins=60, alpha=0.6, density=True, label=p, color=colors[p])
    ax.set_xlabel("speed between consecutive fixes (m/s, clipped at 60)"); ax.set_ylabel("density")
    ax.set_title("Implied vehicle speed"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "speed_distribution.png", dpi=130); plt.close(fig)

    # 4. where the trips are
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for p, g in inv.groupby("partition"):
        ax.scatter((g.lon_min + g.lon_max) / 2, (g.lat_min + g.lat_max) / 2, s=25, label=p, color=colors[p], alpha=0.8)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.set_title("Trip locations (bbox centres)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "trip_locations.png", dpi=130); plt.close(fig)

    # 5. network size
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for p, g in inv.groupby("partition"):
        axes[0].scatter(g["n_nodes"], g["n_arcs"], s=18, label=p, color=colors[p])
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("nodes"); axes[0].set_ylabel("arcs"); axes[0].set_title("Road network size per trip"); axes[0].legend()
    for p, a in all_al.items():
        axes[1].hist(np.clip(a, 0, 400), bins=80, alpha=0.6, density=True, label=p, color=colors[p])
    axes[1].set_xlabel("arc length (m, clipped at 400)"); axes[1].set_title("Road segment lengths"); axes[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "network_size.png", dpi=130); plt.close(fig)

    # 6. connectivity
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(inv["n_weak_components"], inv["largest_strong_frac"], c=inv["partition"].map(colors), s=20)
    ax.set_xscale("log"); ax.set_xlabel("weakly connected components"); ax.set_ylabel("share of nodes in largest SCC")
    ax.set_title("Road network connectivity (train=blue, student_test=orange)")
    fig.tight_layout(); fig.savefig(OUT / "connectivity.png", dpi=130); plt.close(fig)

    # 7. sample trajectories in metres with the route + nearby network
    sample = ["00000005", "00000013", "00000033", "00000054", "00000081", "00000092"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, tid in zip(axes.ravel(), sample):
        trip = load_trip("train", tid)
        proj = LocalProjector.around(trip.lon, trip.lat)
        x, y = proj.forward(trip.lon, trip.lat)
        net = RoadNetwork(trip.nodes, trip.arcs, proj).build_index(x, y, margin=150)
        near = net.indexed_arcs
        for a in near[:: max(1, len(near) // 4000)]:
            ax.plot([net.ax[a], net.bx[a]], [net.ay[a], net.by[a]], color="#bbbbbb", lw=0.5)
        for a in trip.route:
            ax.plot([net.ax[a], net.bx[a]], [net.ay[a], net.by[a]], color="#2f9e44", lw=2)
        ax.scatter(x, y, s=1.5, color="#1f4fd6")
        ax.set_aspect("equal"); ax.set_title(f"train/{tid}  ({len(x)} fixes)")
        ax.set_xlabel("east (m)"); ax.set_ylabel("north (m)")
    fig.suptitle("GPS fixes (blue), reference route (green), nearby road network (grey)")
    fig.tight_layout(); fig.savefig(OUT / "sample_trajectories.png", dpi=120); plt.close(fig)

    # 8. folium maps
    for part, tid in [("train", "00000005"), ("train", "00000054"), ("student_test", "00000007")]:
        trip = load_trip(part, tid)
        proj = LocalProjector.around(trip.lon, trip.lat)
        x, y = proj.forward(trip.lon, trip.lat)
        net = RoadNetwork(trip.nodes, trip.arcs, proj).build_index(x, y, margin=120)
        # only roads within 60 m of the track - drawing the whole bbox makes 50 MB maps
        near = np.unique(np.concatenate(net.arcs_within(x, y, 60.0)))
        m = viz.base_map(trip)
        viz.add_network_layer(m, trip, near)
        viz.add_route_layer(m, trip, trip.route, "Reference route", "#2f9e44")
        viz.add_gps_layer(m, trip)
        folium.LayerControl(collapsed=False).add_to(m)
        viz.add_legend(m, [("Raw GPS fixes", "#1f4fd6"), ("Reference route", "#2f9e44"),
                           ("Road network", "#7a7a7a")], title=f"{part}/{tid}")
        m.save(str(config.MAPS_DIR / f"eda_{part}_{tid}.html"))

    # 9. short machine-written fact sheet (numbers only - interpretation goes in the report)
    tr = inv[inv.partition == "train"]
    facts = {
        "trips": inv.groupby("partition")["trip_id"].count().to_dict(),
        "fixes_total": inv.groupby("partition")["n_fixes"].sum().to_dict(),
        "nan_values_track": int(inv["nan_track"].sum()),
        "nan_values_nodes": int(inv["nan_nodes"].sum()),
        "invalid_coordinates": int(inv["bad_coord"].sum()),
        "non_increasing_timestamps": int(inv["non_increasing_t"].sum()),
        "share_steps_exactly_1s": {p: float(np.mean(d == 1)) for p, d in all_dt.items()},
        "trips_with_gap_over_60s": inv.loc[inv.gaps_gt60s > 0, "trip_id"].tolist(),
        "largest_gap_s": inv.set_index("trip_id")["dt_max"].nlargest(5).to_dict(),
        "duplicate_positions_total": int(inv["dup_position"].sum()),
        "route_connectivity_min": float(inv["route_connectivity"].min()),
        "trips_route_not_fully_connected": inv.loc[inv.route_connectivity < 1, "trip_id"].tolist(),
        "route_ids_out_of_range": int(inv["route_ids_out_of_range"].sum()),
        "tracks_outside_network_bbox": inv.loc[~inv.track_inside_net_bbox, "trip_id"].tolist(),
        "train_two_way_share_median": float(tr["two_way_share"].median()),
        "isolated_nodes_total": int(inv["isolated_nodes"].sum()),
    }
    (OUT / "data_quality_facts.json").write_text(json.dumps(facts, indent=2))
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
