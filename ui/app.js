// PathSnap evaluator UI - runs on laptop 2, talks to the API on laptop 1.
// The UI only ever receives predictions and metrics; the reference route stays on the server.

const $ = (id) => document.getElementById(id);
const store = { get: (k, d) => { try { return localStorage.getItem(k) || d; } catch (e) { return d; } },
                set: (k, v) => { try { localStorage.setItem(k, v); } catch (e) {} } };

const apiInput = $("apiUrl");
apiInput.value = new URLSearchParams(location.search).get("api") || store.get("pathsnap_api", "http://localhost:8000");
let busy = false;

// ---------------------------------------------------------------- map (Leaflet, or plain SVG if offline)
let map = null, layers = [];
if (window.L) {
  map = L.map("map", { preferCanvas: true }).setView([50, 20], 4);
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19, attribution: "&copy; OpenStreetMap contributors" }).addTo(map);
}

function draw(res) {
  if (map) {
    layers.forEach((l) => map.removeLayer(l)); layers = [];
    const clf = L.layerGroup(res.classifier_only_route.polylines.map((p) => L.polyline(p, { color: "#7048e8", weight: 2, opacity: 0.7 })));
    const pred = L.layerGroup(res.predicted_route.polylines.map((p) => L.polyline(p, { color: "#e8590c", weight: 5, opacity: 0.9 })));
    const gps = L.layerGroup(res.gps.map((p) => L.circleMarker(p, { radius: 2, color: "#1f4fd6", weight: 0, fillOpacity: 0.8 })));
    [gps, clf, pred].forEach((l) => { l.addTo(map); layers.push(l); });  // predicted route drawn on top
    if (window._ctl) map.removeControl(window._ctl);
    window._ctl = L.control.layers(null, { "GPS fixes": gps, "Predicted route (HMM)": pred, "Classifier only": clf }, { collapsed: false }).addTo(map);
    map.fitBounds(L.latLngBounds(res.gps), { padding: [20, 20] });
    return;
  }
  // offline fallback: equirectangular SVG
  const all = res.gps.concat(...res.predicted_route.polylines);
  const lats = all.map((p) => p[0]), lons = all.map((p) => p[1]);
  const la0 = Math.min(...lats), la1 = Math.max(...lats), lo0 = Math.min(...lons), lo1 = Math.max(...lons);
  const k = Math.cos(((la0 + la1) / 2) * Math.PI / 180);
  const W = $("map").clientWidth, H = $("map").clientHeight, pad = 20;
  const s = Math.min((W - 2 * pad) / ((lo1 - lo0) * k || 1e-9), (H - 2 * pad) / ((la1 - la0) || 1e-9));
  const xy = (p) => `${(pad + (p[1] - lo0) * k * s).toFixed(1)},${(H - pad - (p[0] - la0) * s).toFixed(1)}`;
  const line = (p, c, w) => `<polyline points="${p.map(xy).join(" ")}" fill="none" stroke="${c}" stroke-width="${w}"/>`;
  $("map").innerHTML = `<svg width="${W}" height="${H}" style="background:#fff">${
    res.classifier_only_route.polylines.map((p) => line(p, "#7048e8", 1.5)).join("")}${
    res.predicted_route.polylines.map((p) => line(p, "#e8590c", 3.5)).join("")}${
    res.gps.map((p) => `<circle cx="${xy(p).split(",")[0]}" cy="${xy(p).split(",")[1]}" r="1.6" fill="#1f4fd6"/>`).join("")}</svg>`;
}

// ---------------------------------------------------------------- API calls
const api = () => apiInput.value.replace(/\/+$/, "");

let retries = 0;
async function connect(auto = false) {
  store.set("pathsnap_api", apiInput.value);
  $("apiStatus").textContent = "connecting...";
  try {
    const r = await fetch(`${api()}/trips`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    $("apiStatus").textContent = `connected - ${data.trips.length} trips`;
    $("tripList").innerHTML = "";
    data.trips.forEach((t) => {
      const b = document.createElement("button");
      b.className = "trip"; b.dataset.id = t.trip_id;
      b.innerHTML = `<b>Trip ${t.trip_id}</b><small>${t.n_fixes.toLocaleString()} GPS fixes</small>`;
      b.onclick = () => predict(t.trip_id, b);
      $("tripList").appendChild(b);
    });
  } catch (e) {
    $("apiStatus").textContent = "not connected";
    if (auto && retries++ < 5) { setTimeout(() => connect(true), 3000); }  // API may still be starting
    $("tripList").innerHTML = `<p class="error">Could not reach ${api()}/trips (${e.message}). Check the address, that the API is running on laptop 1, and the firewall.</p>`;
  }
}

const pct = (v) => (v === null || v === undefined ? "n/a" : (100 * v).toFixed(2) + "%");

async function predict(id, btn) {
  if (busy) return;
  busy = true;
  document.querySelectorAll(".trip").forEach((b) => { b.classList.toggle("active", b === btn); b.disabled = true; });
  $("results").innerHTML = `<p class="muted">Trip ${id}: running the pipeline on the server...</p>`;
  const t0 = performance.now();
  try {
    const r = await fetch(`${api()}/predict`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ trip_id: id }) });
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
    const res = await r.json();
    const m = res.metrics, t = res.timing_ms;
    $("results").innerHTML = `
      <p>Trip <b>${res.trip_id}</b> - ${res.n_fixes.toLocaleString()} fixes, ${res.n_candidates.toLocaleString()} candidates</p>
      ${m.available ? `<div class="muted">Top-1 accuracy (classifier + HMM)</div><div class="big">${pct(m.top1_hmm)}</div>` : `<p class="muted">No reference data on the server for this trip.</p>`}
      <table>
        ${m.available ? `
        <tr><td>Top-1, classifier only</td><td>${pct(m.top1_classifier_only)}</td></tr>
        <tr><td>Route F1 (HMM)</td><td>${pct(m.route_f1_hmm)}</td></tr>
        <tr><td>Route precision / recall</td><td>${pct(m.route_precision_hmm)} / ${pct(m.route_recall_hmm)}</td></tr>
        <tr><td>Route F1, classifier only</td><td>${pct(m.route_f1_classifier_only)}</td></tr>
        <tr><td>Scored fixes</td><td>${m.n_scored_fixes.toLocaleString()}</td></tr>` : ""}
        <tr><td>Continuity HMM / classifier</td><td>${m.continuity_hmm.toFixed(3)} / ${m.continuity_classifier_only.toFixed(3)}</td></tr>
        <tr><td>Unfillable breaks / segments</td><td>${m.unfillable_breaks_hmm} / ${m.segments}</td></tr>
        <tr><td>Server time</td><td>${(t.total_ms / 1000).toFixed(2)} s</td></tr>
        <tr><td>Round trip</td><td>${((performance.now() - t0) / 1000).toFixed(2)} s</td></tr>
      </table>
      <p class="muted" style="font-size:12px">Server stages (ms): ${Object.entries(t).map(([k, v]) => `${k.replace("_ms", "")} ${Math.round(v)}`).join(", ")}</p>`;
    draw(res);
  } catch (e) {
    $("results").innerHTML = `<p class="error">Request failed: ${e.message}</p>`;
  } finally {
    busy = false;
    document.querySelectorAll(".trip").forEach((b) => (b.disabled = false));
  }
}

$("connectBtn").onclick = () => connect();
apiInput.addEventListener("keydown", (e) => { if (e.key === "Enter") connect(); });
connect(true);
