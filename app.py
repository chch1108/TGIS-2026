"""
app.py — 雲林縣智慧救災決策支援系統
Yunlin Smart Disaster Relief Decision Support System
"""

import streamlit as st
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap, AntPath
from streamlit_folium import folium_static
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io, base64, json
import time
import requests
from PIL import Image

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="雲林縣智慧救災決策系統",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@500;600;700;800&family=Noto+Sans+TC:wght@400;500;600;700;800&family=Zen+Maru+Gothic:wght@500;700;900&display=swap');

:root {
    --blue-900: #123a72;
    --blue-700: #2468c9;
    --blue-500: #4a94ed;
    --blue-100: #eaf4ff;
    --ink: #20324a;
    --muted: #6e8199;
    --line: #dce9f7;
}

html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: 'Zen Maru Gothic', 'Noto Sans TC', 'Inter', sans-serif;
    color: var(--ink);
}
.stApp {
    background:
        radial-gradient(circle at 82% 4%, rgba(137,196,255,0.26), transparent 25rem),
        linear-gradient(180deg, #f8fbff 0%, #eef6ff 100%);
}
[data-testid="stAppViewContainer"] > .main { background: transparent; }
[data-testid="stHeader"] { background: rgba(248,251,255,0.82); }
.block-container { padding: 2rem 2.4rem 3rem; max-width: 1560px; }
h1, h2, h3, p, label, span { letter-spacing: 0.01em; }
hr {
    border: 0 !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, #cfe0f2 18%, #cfe0f2 82%, transparent);
    margin: 1.8rem 0 !important;
}

/* Hero banner */
.hero-banner {
    background:
        radial-gradient(circle at 88% 18%, rgba(255,255,255,0.48), transparent 12rem),
        linear-gradient(120deg, #e7f4ff 0%, #cce7ff 48%, #9fcfff 100%);
    border: 1px solid rgba(111,171,230,0.42);
    border-radius: 0;
    padding: 2.1rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 18px 45px rgba(50,117,184,0.13);
}
.hero-banner::before {
    content: '';
    position: absolute;
    width: 240px;
    height: 240px;
    right: -65px;
    bottom: -135px;
    border-radius: 0;
    background: rgba(255,255,255,0.32);
}
.hero-title {
    position: relative;
    font-size: 2.15rem;
    font-weight: 900;
    color: var(--blue-900);
    margin: 0.75rem 0 0;
}
.hero-sub {
    position: relative;
    font-size: 0.95rem;
    font-weight: 500;
    color: #496783;
    margin: 0.45rem 0 0;
    line-height: 1.8;
}
.hero-badge {
    position: relative;
    display: inline-block;
    background: rgba(255,255,255,0.72);
    border: 1px solid rgba(68,133,198,0.23);
    color: var(--blue-700);
    padding: 0.34rem 0.85rem;
    border-radius: 0;
    font-size: 0.78rem;
    font-weight: 700;
    margin: 0 0.35rem 0.25rem 0;
    box-shadow: 0 4px 12px rgba(69,125,180,0.08);
}

/* KPI Cards */
.kpi-card {
    min-height: 126px;
    background: linear-gradient(150deg, rgba(255,255,255,0.98), rgba(242,248,255,0.96));
    border: 1px solid rgba(185,213,240,0.72);
    border-radius: 0;
    padding: 1.25rem 1rem;
    text-align: center;
    box-shadow: 0 10px 28px rgba(50,104,158,0.09);
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}
.overview-kpi {
    min-height: 126px;
    padding: 1.25rem 0.45rem;
}
.overview-kpi .kpi-value {
    font-size: 1.45rem;
    white-space: nowrap;
}
.overview-kpi .kpi-unit {
    font-size: 0.72rem;
    margin-left: 0.12rem;
}
.overview-kpi .kpi-label {
    font-size: 0.72rem;
    white-space: nowrap;
}
.kpi-card:hover {
    transform: translateY(-4px);
    border-color: #83baf0;
    box-shadow: 0 16px 34px rgba(45,111,177,0.15);
}
.kpi-value {
    font-family: 'Inter', 'Noto Sans TC', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: var(--blue-700);
    line-height: 1.15;
}
.kpi-label { font-size: 0.82rem; font-weight: 700; color: #5f7590; margin-top: 0.55rem; }
.kpi-delta { font-size: 0.75rem; color: #8293a7; margin-top: 0.35rem; }

/* Risk colors */
.risk-none    { color: #20976c; }
.risk-low     { color: #d59820; }
.risk-med     { color: #e57b35; }
.risk-high    { color: #df5261; }
.risk-extreme { color: #8b61c9; }

/* Section headers */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    font-size: 1.12rem;
    font-weight: 900;
    color: var(--blue-900);
    margin: 1.7rem 0 1rem;
}
.section-header::before {
    content: '';
    width: 9px;
    height: 9px;
    border-radius: 0;
    background: linear-gradient(135deg, #73b7fa, #2d78d4);
    box-shadow: 0 0 0 6px #e3f1ff;
}

/* Warning badge */
.badge-green  { background:#e8f8f1; color:#16835c; border:1px solid #9edcc4; }
.badge-yellow { background:#fff7dd; color:#a66d00; border:1px solid #efd486; }
.badge-red    { background:#fff0f2; color:#c33d50; border:1px solid #efb2bb; }
.badge { display:inline-block; padding:0.2rem 0.6rem; border-radius:0;
         font-size:0.75rem; font-weight:600; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at 10% 0%, rgba(177,217,255,0.52), transparent 17rem),
        linear-gradient(180deg, #f8fbff 0%, #edf6ff 100%) !important;
    border-right: 1px solid #d7e7f7;
    box-shadow: 8px 0 28px rgba(53,102,151,0.06);
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--ink); }
.sidebar-brand { padding: 0.4rem 0 1rem; }
.sidebar-brand-title { font-size: 1.25rem; font-weight: 900; color: var(--blue-900); }
.sidebar-brand-sub { color: #6f849c; font-size: 0.76rem; margin-top: 0.2rem; }
.sidebar-section {
    color: #537493 !important;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    margin: 1.25rem 0 0.35rem !important;
    font-weight: 800;
}

/* Streamlit controls */
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-testid="stNumberInput"] input {
    background: rgba(255,255,255,0.9) !important;
    border-color: #cddff1 !important;
    border-radius: 0 !important;
    color: var(--ink) !important;
}
[data-testid="stRadio"] label,
[data-testid="stToggle"] label { color: #405a76 !important; font-weight: 600; }
[data-testid="stSidebar"] [role="radiogroup"] label {
    padding: 0.45rem 0.55rem;
    border-radius: 0;
    transition: background 0.18s ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: rgba(205,229,252,0.6);
}
.stButton > button {
    border: 0;
    border-radius: 0;
    min-height: 2.75rem;
    font-weight: 800;
    letter-spacing: 0.02em;
    box-shadow: 0 8px 18px rgba(41,105,173,0.16);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(110deg, #2c72cf, #5aa8f4);
    color: white;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 24px rgba(41,105,173,0.22);
}
[data-testid="stDataFrame"],
[data-testid="stPlotlyChart"],
[data-testid="stExpander"],
[data-testid="stAlert"] {
    border-radius: 0;
    overflow: hidden;
}
[data-testid="stAlert"] {
    background: rgba(235,246,255,0.92);
    border: 1px solid #c9e2f8;
    color: #365b7d;
}
iframe { border-radius: 0; }

@media (max-width: 900px) {
    .block-container { padding: 1.25rem 1rem 2rem; }
    .hero-banner { padding: 1.5rem 1.25rem; border-radius: 0; }
    .hero-title { font-size: 1.7rem; }
    .kpi-card { min-height: 112px; padding: 1rem 0.7rem; }
    .kpi-value { font-size: 1.65rem; }
    .overview-kpi .kpi-value { font-size: 1.2rem; }
    .overview-kpi .kpi-label { font-size: 0.66rem; }
}

/* Square visual language */
.stApp *,
.stApp *::before,
.stApp *::after {
    border-radius: 0 !important;
}

/* Keep sidebar on/off switches soft and easy to recognize. */
section[data-testid="stSidebar"] [data-testid="stToggle"] *,
section[data-testid="stSidebar"] [role="switch"],
section[data-testid="stSidebar"] [role="switch"] *,
section[data-testid="stSidebar"] [data-baseweb="checkbox"],
section[data-testid="stSidebar"] [data-baseweb="checkbox"] *,
section[data-testid="stSidebar"] [data-baseweb="checkbox"] *::before,
section[data-testid="stSidebar"] [data-baseweb="checkbox"] *::after {
    border-radius: 999px !important;
}
section[data-testid="stSidebar"] [data-testid="stToggle"] [role="switch"] {
    min-width: 2.75rem;
    box-shadow: inset 0 0 0 1px rgba(71, 112, 151, 0.18);
}

/* Restore circular radio selectors and the hero decoration. */
section[data-testid="stSidebar"] [data-testid="stRadio"] [data-baseweb="radio"] *,
section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radio"],
section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radio"] *,
section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radio"] *::before,
section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radio"] *::after {
    border-radius: 50% !important;
}
.hero-banner::before {
    border-radius: 50% !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Imports (lazy) ───────────────────────────────────────────────────────────
from data_loader import (
    TYPHOON_NAMES, INTENSITY_LABELS, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX,
    load_typhoon_info, load_typhoon_tracks, load_all_facilities,
    load_typhoon_max, load_typhoon_max_png, load_typhoon_animation_gif,
    compute_flood_statistics
)
from risk_assessment import (
    compute_population_exposure, full_facility_risk_report,
    compute_shelter_utilization, compute_county_risk_index
)

TYPHOON_NAMES_DISPLAY = {k: f"#{k} {v}" for k, v in TYPHOON_NAMES.items()}

# ─── Caching ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cached_flood_max(tid):       return load_typhoon_max(tid)
@st.cache_data(show_spinner=False)
def cached_flood_stats(tid):     return compute_flood_statistics(tid)
@st.cache_data(show_spinner=False)
def cached_pop_exposure(tid):    return compute_population_exposure(tid)
@st.cache_data(show_spinner=False)
def cached_risk_index(tid):      return compute_county_risk_index(tid)
@st.cache_data(show_spinner=False)
def cached_facility_risk(tid):   return full_facility_risk_report(tid)
@st.cache_data(show_spinner=False)
def cached_shelter_util(tid):    return compute_shelter_utilization(tid)
@st.cache_data(show_spinner=False)
def cached_typhoon_info():       return load_typhoon_info()
@st.cache_data(show_spinner=False)
def cached_facilities():         return load_all_facilities()
@st.cache_data(show_spinner=False)
def cached_tracks():             return load_typhoon_tracks()

def format_grouped(value) -> str:
    """Format a value without thousands separators."""
    return str(value)

@st.cache_data(ttl=86400, show_spinner=False)
def geocode_address(address: str) -> dict | None:
    """Resolve a Yunlin address or landmark with OpenStreetMap Nominatim."""
    query = address.strip()
    if not query:
        return None
    if "雲林" not in query:
        query = f"雲林縣 {query}"

    # Public Nominatim requests must be identifiable and kept below 1 request/s.
    time.sleep(1.05)
    response = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={
            "q": query,
            "format": "jsonv2",
            "limit": 5,
            "countrycodes": "tw",
            "accept-language": "zh-TW",
            "viewbox": f"{LON_MIN},{LAT_MAX},{LON_MAX},{LAT_MIN}",
            "bounded": 1,
        },
        headers={"User-Agent": "TGIS-Yunlin-Disaster-Support/1.0"},
        timeout=15,
    )
    response.raise_for_status()
    results = response.json()
    if not results:
        return None

    normalized_query = query.replace(" ", "").replace(",", "").replace("臺", "台")
    match = max(
        results,
        key=lambda item: (
            item.get("name", "").replace(" ", "").replace("臺", "台") == normalized_query,
            normalized_query in item["display_name"].replace(" ", "").replace("臺", "台"),
            float(item.get("importance", 0)),
        ),
    )
    return {
        "lat": float(match["lat"]),
        "lon": float(match["lon"]),
        "display_name": match["display_name"],
    }

def event_title(typhoon_id: int, custom_typhoon: dict | None = None) -> str:
    if custom_typhoon:
        return f"{custom_typhoon['name']}（自訂）"
    return f"颱風 #{typhoon_id}"

def risk_with_custom_wind(reference_id: int, custom_typhoon: dict | None) -> dict:
    risk = cached_risk_index(reference_id).copy()
    if not custom_typhoon:
        return risk

    reference_row = cached_typhoon_info()
    reference_row = reference_row[reference_row["id"] == reference_id].iloc[0]
    old_wind_score = min(10, float(reference_row["wind_speed_ms"]) / 6)
    new_wind_score = min(10, float(custom_typhoon["wind_speed_ms"]) / 6)
    total = risk["risk_index"] - old_wind_score + new_wind_score
    risk.update({
        "risk_index": round(total, 1),
        "risk_grade": "A" if total < 20 else ("B" if total < 40 else ("C" if total < 60 else "D")),
        "risk_label": "低風險" if total < 20 else (
            "中風險" if total < 40 else ("高風險" if total < 60 else "極高風險")
        ),
        "wind_score": round(new_wind_score, 1),
    })
    return risk

@st.dialog("自訂颱風資訊", width="large")
def custom_typhoon_dialog():
    info = cached_typhoon_info()
    current = st.session_state.get("custom_typhoon", {})
    reference_options = {
        int(row["id"]): f"#{int(row['id'])} {row['name']} ({int(row['year'])})"
        for _, row in info.iterrows()
    }
    reference_ids = list(reference_options)
    default_reference = int(current.get("reference_id", 5))
    reference_index = (
        reference_ids.index(default_reference)
        if default_reference in reference_ids else reference_ids.index(5)
    )

    st.caption("自訂氣象資訊會搭配一組既有淹水情境，供地圖、設施與路徑分析使用。")
    with st.form("custom_typhoon_form"):
        c1, c2 = st.columns(2)
        name = c1.text_input("颱風名稱", value=current.get("name", "自訂颱風"))
        year = c2.number_input(
            "年份", min_value=1900, max_value=2100,
            value=int(current.get("year", 2026)), step=1,
        )
        landfall_time = st.text_input(
            "登陸時間",
            value=current.get("landfall_time", "2026/06/11 12:00"),
            placeholder="YYYY/MM/DD HH:MM",
        )
        c3, c4 = st.columns(2)
        intensity = c3.selectbox(
            "近臺強度",
            options=[1, 2, 3],
            index=max(0, min(2, int(current.get("intensity", 2)) - 1)),
            format_func=lambda value: INTENSITY_LABELS[value],
        )
        pressure_hpa = c4.number_input(
            "近臺最低氣壓（hPa）",
            min_value=850, max_value=1050,
            value=int(current.get("pressure_hpa", 955)), step=1,
        )
        c5, c6 = st.columns(2)
        wind_speed_ms = c5.number_input(
            "近臺最大風速（m/s）",
            min_value=0.0, max_value=100.0,
            value=float(current.get("wind_speed_ms", 40.0)), step=1.0,
        )
        radius_7_km = c6.number_input(
            "7級暴風半徑（km）",
            min_value=0, max_value=500,
            value=int(current.get("radius_7_km", 250)), step=10,
        )
        c7, c8 = st.columns(2)
        radius_10_km = c7.number_input(
            "10級暴風半徑（km）",
            min_value=0, max_value=300,
            value=int(current.get("radius_10_km", 100)), step=10,
        )
        reference_id = c8.selectbox(
            "淹水情境參考",
            options=reference_ids,
            index=reference_index,
            format_func=lambda value: reference_options[value],
        )
        submitted = st.form_submit_button("套用自訂颱風", type="primary")

    if submitted:
        if not name.strip() or not landfall_time.strip():
            st.error("請填寫颱風名稱及登陸時間。")
            return
        st.session_state["custom_typhoon"] = {
            "name": name.strip(),
            "year": int(year),
            "landfall_time": landfall_time.strip(),
            "intensity": int(intensity),
            "intensity_label": INTENSITY_LABELS[intensity],
            "pressure_hpa": int(pressure_hpa),
            "wind_speed_ms": float(wind_speed_ms),
            "radius_7_km": int(radius_7_km),
            "radius_10_km": int(radius_10_km),
            "reference_id": int(reference_id),
        }
        st.rerun()

# ─── Helper: flood overlay PNG → base64 ──────────────────────────────────────
def flood_png_to_b64(tid: int) -> str:
    path = load_typhoon_max_png(tid)
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# ─── Helper: build Folium map ─────────────────────────────────────────────────
FACILITY_ICONS = {
    'hospitals':    ('red',   'plus',        '醫院'),
    'care_centers': ('orange','home',        '長照中心'),
    'substations':  ('yellow','bolt',        '變電所'),
    'shelters':     ('green', 'map-marker',  '收容所'),
}

def build_folium_map(typhoon_id: int,
                     show_facilities: dict,
                     show_heatmap: bool = True,
                     show_track: bool = True,
                     route_data: dict = None) -> folium.Map:
    cx = (LAT_MIN + LAT_MAX) / 2
    cy = (LON_MIN + LON_MAX) / 2
    m = folium.Map(location=[cx, cy], zoom_start=10,
                   tiles='CartoDB positron',
                   attr='CartoDB')
    m.get_root().header.add_child(folium.Element("""
        <style>
        .leaflet-bar,
        .leaflet-bar a,
        .leaflet-control,
        .leaflet-popup-content-wrapper,
        .leaflet-popup-tip {
            border-radius: 0 !important;
        }
        </style>
    """))

    # ── Flood heatmap ──
    if show_heatmap:
        flood_max = cached_flood_max(typhoon_id)
        step = 8   # sample every 8th cell for speed
        ys, xs = np.where(flood_max[::step, ::step] > 0)
        depths = flood_max[::step, ::step][ys, xs]
        lats_h = LAT_MAX - ys * step / (flood_max.shape[0]) * (LAT_MAX - LAT_MIN)
        lons_h = LON_MIN + xs * step / (flood_max.shape[1]) * (LON_MAX - LON_MIN)
        heat_data = [[float(lats_h[i]), float(lons_h[i]), float(depths[i])]
                     for i in range(len(lats_h))]
        HeatMap(heat_data, radius=10, blur=12,
                gradient={0.2:'#1a6fff',0.5:'#f39c12',0.8:'#e74c3c',1.0:'#8e44ad'},
                min_opacity=0.3).add_to(m)

    # ── Typhoon track ──
    if show_track:
        tracks = cached_tracks()
        t_track = tracks[tracks['typhoon_id'] == typhoon_id].sort_values('date')
        if len(t_track) > 1:
            pts = list(zip(t_track['lat'], t_track['lon']))
            AntPath(pts, color='#00d4ff', weight=3, opacity=0.8,
                    dash_array=[10, 20]).add_to(m)
            for _, row in t_track.iterrows():
                folium.CircleMarker(
                    [row['lat'], row['lon']], radius=5,
                    color='#00d4ff', fill=True, fill_color='white',
                    popup=str(row['date'])
                ).add_to(m)

    # ── Facilities ──
    facilities = cached_facilities()
    risk_df    = cached_facility_risk(typhoon_id)
    risk_lookup = {(r['name'], r['type']): r for _, r in risk_df.iterrows()}

    for ftype, enabled in show_facilities.items():
        if not enabled or ftype not in facilities:
            continue
        icon_color, icon_name, facility_label = FACILITY_ICONS.get(
            ftype, ('blue', 'info', '設施')
        )
        # glyphicon icon names (no fa prefix needed)
        glyph_map = {'plus':'plus-sign','home':'home','bolt':'flash','map-marker':'map-marker'}
        glyph_name = glyph_map.get(icon_name, 'info-sign')
        for _, row in facilities[ftype].iterrows():
            key = (row['name'], ftype)
            r_info = risk_lookup.get(key, {})
            depth  = r_info.get('flood_depth_cm', 0)
            level  = r_info.get('risk_level', '未知')
            cap    = row.get('capacity', '')
            cap_str = f' | 容量:{int(cap)}人' if str(cap) not in ('', 'nan', '0') else ''
            tip = f"{facility_label}｜{row['name']}｜淹水 {depth} cm｜{level}{cap_str}"
            folium.Marker(
                [row['lat'], row['lon']],
                popup=tip,
                tooltip=str(row['name'])[:30],
                icon=folium.Icon(color=icon_color, icon=glyph_name)
            ).add_to(m)

    # ── Routes ──
    if route_data:
        for route_type in ['dijkstra', 'rl']:
            rd = route_data.get(route_type, {})
            path = rd.get('path_wgs84', [])
            if len(path) > 1:
                folium.PolyLine(
                    [[p[1], p[0]] for p in path],
                    color=rd.get('color', '#fff'),
                    weight=4, opacity=0.85,
                    tooltip=rd.get('label', route_type)
                ).add_to(m)
        # Origin / destination markers
        if 'origin' in route_data:
            folium.Marker(route_data['origin'], tooltip='起點',
                          icon=folium.Icon(color='blue', icon='home')
                          ).add_to(m)
        if 'destination' in route_data:
            folium.Marker(route_data['destination'], tooltip='目的地',
                          icon=folium.Icon(color='green', icon='flag')
                          ).add_to(m)
    return m

# ─── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">雲林智慧防災</div>
            <div class="sidebar-brand-sub">Disaster Decision Support</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<p class="sidebar-section">颱風事件</p>', unsafe_allow_html=True)
        typhoon_info = cached_typhoon_info()
        options = {row['id']: f"#{row['id']} {row['name']} ({row['year']})"
                   for _, row in typhoon_info.iterrows()}
        options["custom"] = "自訂"
        selected_event = st.selectbox(
            "選擇颱風事件",
            list(options.keys()),
            format_func=lambda value: options.get(value, value),
            index=4,
            key="selected_typhoon_event",
        )
        custom_typhoon = None
        if selected_event in ("custom", "自訂"):
            custom_typhoon = st.session_state.get("custom_typhoon")
            if custom_typhoon is None:
                custom_typhoon_dialog()
            else:
                st.caption(
                    f"{custom_typhoon['name']} · "
                    f"{custom_typhoon['intensity_label']} · "
                    f"{custom_typhoon['wind_speed_ms']:g} m/s"
                )
                if st.button("編輯自訂資訊", use_container_width=True):
                    custom_typhoon_dialog()
            sel_id = int(custom_typhoon["reference_id"]) if custom_typhoon else 5
        else:
            sel_id = int(selected_event)

        st.markdown('<p class="sidebar-section">地圖圖層</p>', unsafe_allow_html=True)
        show_heatmap = st.toggle("淹水熱點圖", value=True)
        show_track   = st.toggle("颱風路徑", value=True)

        st.markdown('<p class="sidebar-section">設施顯示</p>', unsafe_allow_html=True)
        show_hospitals    = st.toggle("醫院", value=True)
        show_care_centers = st.toggle("長照中心", value=False)
        show_substations  = st.toggle("變電所", value=False)
        show_shelters     = st.toggle("避難收容所", value=True)

        st.markdown('<p class="sidebar-section">功能選單</p>', unsafe_allow_html=True)
        page = st.radio("功能模組", [
            "淹水總覽",
            "設施風險評估",
            "資源分配 (GA+ACO)",
            "避難路徑規劃",
            "收容所管理",
        ])

        show_fac = {
            'hospitals':    show_hospitals,
            'care_centers': show_care_centers,
            'substations':  show_substations,
            'shelters':     show_shelters,
        }
        return sel_id, page, show_fac, show_heatmap, show_track, custom_typhoon


# ─── Page: Flood Overview ─────────────────────────────────────────────────────
def page_flood_overview(typhoon_id, show_fac, show_heatmap, show_track,
                        custom_typhoon=None):
    info = cached_typhoon_info()
    row = custom_typhoon or info[info['id'] == typhoon_id].iloc[0]
    risk = risk_with_custom_wind(typhoon_id, custom_typhoon)
    pop  = cached_pop_exposure(typhoon_id)
    stats= cached_flood_stats(typhoon_id)

    # Hero
    risk_colors = {'低風險':'#27ae60','中風險':'#f1c40f','高風險':'#e74c3c','極高風險':'#8e44ad'}
    rc = risk_colors.get(risk['risk_label'], '#6e8199')
    event_badge = (
        "自訂颱風事件"
        if custom_typhoon else f"颱風事件 #{typhoon_id}"
    )
    reference_badge = (
        f"<span class='hero-badge'>淹水參考 #{typhoon_id}</span>"
        if custom_typhoon else ""
    )
    st.markdown(f"""
    <div class="hero-banner">
      <span class="hero-badge">{event_badge}</span>
      <span class="hero-badge">{row['intensity_label']}</span>
      {reference_badge}
      <span class="hero-badge" style="color:{rc};border-color:{rc}">
        風險指數 {risk['risk_index']}/100
      </span>
      <h1 class="hero-title">{row['name']}</h1>
      <p class="hero-sub">登陸時間：{row['landfall_time']}　
         最大風速：{row['wind_speed_ms']} m/s　
         近臺氣壓：{row['pressure_hpa']} hPa<br>
         7級暴風半徑：{row['radius_7_km']} km　
         10級暴風半徑：{row['radius_10_km']} km</p>
    </div>""", unsafe_allow_html=True)

    # Overview KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        (k1, f"{stats['total_flooded_area_km2']}", "km²", "淹水面積"),
        (k2, f"{stats['max_depth_cm']}", "cm", "最大淹水深度"),
        (k3, format_grouped(pop['total_exposed']), "人", "受影響人口"),
        (k4, f"{pop['exposure_rate_pct']}", "%", "人口暴露率"),
        (k5, f"{risk['risk_index']}", "/100", "縣市風險指數"),
    ]
    for col, val, unit, label in kpis:
        with col:
            st.markdown(
                f"""
                <div class="kpi-card overview-kpi">
                    <div class="kpi-value">
                        {val}<span class="kpi-unit" style="color:#71859b">{unit}</span>
                    </div>
                    <div class="kpi-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    col_map, col_info = st.columns([3, 2])

    with col_map:
        st.markdown('<div class="section-header">互動式淹水地圖</div>', unsafe_allow_html=True)
        with st.spinner("載入地圖中..."):
            m = build_folium_map(typhoon_id, show_fac, show_heatmap, show_track)
            folium_static(m, height=500)

    with col_info:
        st.markdown('<div class="section-header">淹水深度分佈</div>', unsafe_allow_html=True)
        flood_max = cached_flood_max(typhoon_id)
        flooded = flood_max[flood_max > 0].ravel()
        if len(flooded):
            bins   = [0, 30, 70, 100, 200]
            labels = ['輕微<br><30cm', '中度<br>30-70cm', '嚴重<br>70-100cm', '極嚴重<br>>100cm']
            colors = ['#1a6fff', '#f39c12', '#e74c3c', '#8e44ad']
            counts = [((flooded >= bins[i]) & (flooded < bins[i+1])).sum() for i in range(4)]
            fig = go.Figure(go.Bar(
                x=labels, y=counts, marker_color=colors,
                text=[format_grouped(c) for c in counts], textposition='outside',
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='#36516d', margin=dict(t=20, b=10, l=10, r=10),
                height=220, showlegend=False,
                xaxis=dict(gridcolor='#dfeaf5', zerolinecolor='#dfeaf5'),
                yaxis=dict(title="格網數量", gridcolor='#dfeaf5', zerolinecolor='#dfeaf5'),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">人口暴露分佈</div>', unsafe_allow_html=True)
        exp_labels = ['輕微', '中度', '嚴重', '極嚴重']
        exp_vals   = [pop['exposed_mild'], pop['exposed_moderate'],
                      pop['exposed_severe'], pop['exposed_extreme']]
        exp_colors = ['#1a6fff', '#f39c12', '#e74c3c', '#8e44ad']
        fig2 = go.Figure(go.Pie(
            labels=exp_labels, values=exp_vals, hole=0.5,
            marker_colors=exp_colors,
        ))
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', font_color='#36516d',
            margin=dict(t=10, b=10, l=10, r=10), height=200, showlegend=True,
            legend=dict(font_size=11),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Animation
    st.markdown('<div class="section-header">淹水動態演進</div>', unsafe_allow_html=True)
    gif_path = load_typhoon_animation_gif(typhoon_id)
    with open(gif_path, "rb") as gif_file:
        gif_data = base64.b64encode(gif_file.read()).decode("ascii")
    st.markdown(
        f"""
        <figure style="margin: 0; text-align: center;">
            <img
                src="data:image/gif;base64,{gif_data}"
                alt="{event_title(typhoon_id, custom_typhoon)}逐時淹水演進"
                style="display: block; width: 100%; height: auto; border-radius: 0;"
            >
            <figcaption style="margin-top: 0.65rem; color: #6e8199; font-weight: 600;">
                {event_title(typhoon_id, custom_typhoon)}逐時淹水演進
            </figcaption>
        </figure>
        """,
        unsafe_allow_html=True,
    )


# ─── Page: Facility Risk ──────────────────────────────────────────────────────
def page_facility_risk(typhoon_id, show_fac, show_heatmap, show_track,
                       custom_typhoon=None):
    st.markdown(f'<div class="section-header">重要設施風險評估 · {event_title(typhoon_id, custom_typhoon)}</div>',
                unsafe_allow_html=True)

    risk_df = cached_facility_risk(typhoon_id)

    # Summary counts
    c1, c2, c3, c4 = st.columns(4)
    for col, ftype, label in [
        (c1, 'hospitals', '醫院'),
        (c2, 'care_centers', '長照中心'),
        (c3, 'substations', '變電所'),
        (c4, 'shelters', '收容所'),
    ]:
        sub = risk_df[risk_df['type']==ftype]
        at_risk = (sub['flood_depth_cm'] > 0).sum()
        with col:
            st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value" style="font-size:1.7rem">{at_risk}</div>
            <div class="kpi-label">{label}受淹水影響</div>
            <div class="kpi-delta">共 {len(sub)} 筆</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_map, col_table = st.columns([3, 2])

    with col_map:
        st.markdown('<div class="section-header">地圖</div>', unsafe_allow_html=True)
        m = build_folium_map(typhoon_id, show_fac, show_heatmap, show_track)
        folium_static(m, height=480)

    with col_table:
        st.markdown('<div class="section-header">高風險設施列表</div>', unsafe_allow_html=True)
        ftype_filter = st.selectbox("篩選設施類型", ['全部','hospitals','care_centers','substations','shelters'],
                                     format_func=lambda x: {'全部':'全部','hospitals':'醫院',
                                     'care_centers':'長照中心','substations':'變電所','shelters':'收容所'}.get(x,x))
        show_df = risk_df if ftype_filter == '全部' else risk_df[risk_df['type']==ftype_filter]
        show_df = show_df[show_df['flood_depth_cm'] > 0].head(30)
        display_df = show_df[['type_label','name','flood_depth_cm','risk_level','risk_score']].copy()
        display_df.columns = ['類型','名稱','淹水深度(cm)','風險等級','風險分數']
        st.dataframe(display_df, use_container_width=True, height=400)

    # Risk score distribution
    st.markdown('<div class="section-header">各類型設施風險分數分佈</div>', unsafe_allow_html=True)
    type_labels = {'hospitals':'醫院','care_centers':'長照中心','substations':'變電所','shelters':'收容所'}
    fig = go.Figure()
    for ftype, label in type_labels.items():
        sub = risk_df[risk_df['type']==ftype]['risk_score']
        fig.add_trace(go.Box(y=sub, name=label, boxpoints='outliers'))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.62)',
        font_color='#36516d', height=300, margin=dict(t=20,b=10),
        xaxis=dict(gridcolor='#dfeaf5', zerolinecolor='#dfeaf5'),
        yaxis=dict(gridcolor='#dfeaf5', zerolinecolor='#dfeaf5'),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─── Page: Resource Allocation ────────────────────────────────────────────────
def page_resource_allocation(typhoon_id, show_fac, show_heatmap, show_track,
                             custom_typhoon=None):
    from resource_allocation import run_resource_allocation, get_high_risk_zones
    st.markdown(f'<div class="section-header">抽水機資源分配 · {event_title(typhoon_id, custom_typhoon)}</div>',
                unsafe_allow_html=True)

    with st.expander("分配參數設定", expanded=True):
        c1, c2, c3 = st.columns(3)
        n_pumps  = c1.slider("抽水機數量", 5, 20, 10)
        ga_gens  = c2.slider("GA 演化代數", 30, 200, 80)
        depot_lat= c3.number_input("車庫緯度", value=23.71, format="%.4f")
        depot_lon= c3.number_input("車庫經度", value=120.43, format="%.4f")

    run_key = f"alloc_{typhoon_id}_{n_pumps}_{ga_gens}"
    if run_key not in st.session_state:
        st.session_state[run_key] = None

    if st.button("開始執行 GA + ACO 資源分配", type="primary"):
        progress = st.progress(0, text="初始化...")
        def cb(pct, msg): progress.progress(pct/100, text=msg)
        with st.spinner("運算中（約30秒）..."):
            result = run_resource_allocation(
                typhoon_id, n_pumps=n_pumps, depot_lat=depot_lat, depot_lon=depot_lon,
                ga_generations=ga_gens, progress_cb=cb
            )
        st.session_state[run_key] = result

    result = st.session_state.get(run_key)
    if result is None:
        st.info("設定完成後，點擊上方按鈕執行資源分配演算法。")
        with st.spinner("載入高風險區域..."):
            zones = get_high_risk_zones(typhoon_id, top_k=25)
        st.markdown('<div class="section-header">高優先度淹水區域</div>', unsafe_allow_html=True)
        zone_display = zones[
            ['zone_id', 'lat', 'lon', 'flood_depth_cm', 'priority_score']
        ].head(15).copy()
        zone_display.columns = [
            '區域編號', '緯度', '經度', '淹水深度（cm）', '救災優先分數'
        ]
        st.dataframe(zone_display, use_container_width=True)
        return

    cmp = result['comparison']
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        (c1, f"{cmp['ga_distance_km']}", "km", "GA 總路線距離"),
        (c2, f"{cmp['distance_reduction_pct']}%", "", "相較貪婪縮短"),
        (c3, f"{cmp['ga_risk_covered']:.0f}", "", "GA 風險覆蓋分"),
        (c4, f"{cmp['risk_improvement_pct']}%", "", "風險覆蓋提升"),
    ]
    for col, val, unit, label in metrics:
        with col:
            st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{val}<span style="font-size:1rem;color:#71859b">{unit}</span></div>
            <div class="kpi-label">{label}</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_l, col_r = st.columns([3, 2])

    with col_l:
        # GA convergence
        st.markdown('<div class="section-header">GA 收斂曲線</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Scatter(y=result['ga_history'], mode='lines',
                                    line=dict(color='#1d8cf8', width=2)))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.62)',
            font_color='#36516d', height=250,
            xaxis_title='Generation', yaxis_title='Fitness',
            margin=dict(t=10,b=30,l=30,r=10),
            xaxis=dict(gridcolor='#dfeaf5', zerolinecolor='#dfeaf5'),
            yaxis=dict(gridcolor='#dfeaf5', zerolinecolor='#dfeaf5'),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Deployment map
        st.markdown('<div class="section-header">部署地圖</div>', unsafe_allow_html=True)
        m = build_folium_map(typhoon_id, show_fac, show_heatmap, show_track=False)
        depot = result['depot']
        folium.Marker([depot[0], depot[1]], tooltip='🏭 車庫起點',
                       icon=folium.Icon(color='darkblue', icon='arrow-right')).add_to(m)
        for _, z in result['ga_plan'].iterrows():
            folium.CircleMarker(
                [z['lat'], z['lon']], radius=8 + z['pumps_assigned']*3,
                color='#1d8cf8', fill=True, fill_color='#1d8cf8', fill_opacity=0.7,
                popup=f"Zone {z['zone_id']}: {z['pumps_assigned']} 抽水機<br>深度:{z['flood_depth_cm']}cm"
            ).add_to(m)
        if len(result['aco_route']) > 1:
            folium.PolyLine([[p[0],p[1]] for p in result['aco_route']],
                             color='#f39c12', weight=3, dash_array='5 10',
                             tooltip='ACO 最佳路線').add_to(m)
        folium_static(m, height=380)

    with col_r:
        st.markdown('<div class="section-header">部署計劃</div>', unsafe_allow_html=True)
        plan_df = result['ga_plan'][['zone_id','pumps_assigned','flood_depth_cm','dist_from_depot_km','priority_score']]
        plan_df.columns = ['區域','抽水機數','深度cm','距離km','優先分']
        st.dataframe(plan_df, use_container_width=True, height=300)

        st.markdown('<div class="section-header">方法比較</div>', unsafe_allow_html=True)
        comp_df = pd.DataFrame({
            '指標':    ['總路線距離(km)', '風險覆蓋分'],
            'GA+ACO': [cmp['ga_distance_km'],  cmp['ga_risk_covered']],
            '貪婪法':  [cmp['greedy_distance_km'], cmp['greedy_risk_covered']],
        })
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name='GA+ACO', x=comp_df['指標'], y=comp_df['GA+ACO'],
                               marker_color='#1d8cf8'))
        fig2.add_trace(go.Bar(name='貪婪法', x=comp_df['指標'], y=comp_df['貪婪法'],
                               marker_color='#e74c3c'))
        fig2.update_layout(
            barmode='group', paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(255,255,255,0.62)', font_color='#36516d',
            height=250, margin=dict(t=10,b=30),
            xaxis=dict(gridcolor='#dfeaf5', zerolinecolor='#dfeaf5'),
            yaxis=dict(gridcolor='#dfeaf5', zerolinecolor='#dfeaf5'),
        )
        st.plotly_chart(fig2, use_container_width=True)


# ─── Page: Evacuation Routing ─────────────────────────────────────────────────
def page_evacuation_routing(typhoon_id, show_fac, show_heatmap, show_track,
                            custom_typhoon=None):
    from evacuation_routing import compare_routes
    st.markdown(f'<div class="section-header">避難路徑規劃 · {event_title(typhoon_id, custom_typhoon)}</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**起點地址或地標**")
        origin_address = st.text_input(
            "起點",
            value="國立雲林科技大學",
            placeholder="例如：雲林縣斗六市大學路三段123號",
            label_visibility="collapsed",
        )
    with c2:
        st.markdown("**終點地址或地標**")
        destination_address = st.text_input(
            "終點",
            value="雲林縣政府",
            placeholder="例如：雲林縣斗六市雲林路二段515號",
            label_visibility="collapsed",
        )
    st.caption("可輸入雲林縣內的完整地址、車站、學校或政府機關名稱。")

    if st.button("計算建議路徑", type="primary"):
        st.session_state['route_result'] = None
        if not origin_address.strip() or not destination_address.strip():
            st.error("請完整輸入起點與終點。")
        else:
            try:
                with st.spinner("正在辨識地址..."):
                    origin_match = geocode_address(origin_address)
                    destination_match = geocode_address(destination_address)
            except requests.RequestException:
                st.error("地址定位服務暫時無法連線，請稍後再試。")
            else:
                missing = []
                if origin_match is None:
                    missing.append("起點")
                if destination_match is None:
                    missing.append("終點")
                if missing:
                    st.error(f"無法辨識{'、'.join(missing)}，請輸入更完整的雲林地址或知名地標。")
                else:
                    o_lat, o_lon = origin_match["lat"], origin_match["lon"]
                    d_lat, d_lon = destination_match["lat"], destination_match["lon"]
                    with st.spinner("計算 Dijkstra 及 RL 路徑..."):
                        result = compare_routes(typhoon_id, o_lat, o_lon, d_lat, d_lon)
                    st.session_state['route_result'] = result
                    st.session_state['route_origin'] = [o_lat, o_lon]
                    st.session_state['route_dest'] = [d_lat, d_lon]
                    st.session_state['route_origin_match'] = origin_match
                    st.session_state['route_dest_match'] = destination_match

    result = st.session_state.get('route_result')
    if result is None:
        st.info("輸入起點與終點地址後，點擊「計算建議路徑」。")
        return

    origin_match = st.session_state.get('route_origin_match')
    destination_match = st.session_state.get('route_dest_match')
    if origin_match and destination_match:
        st.info(
            f"起點辨識：{origin_match['display_name']}\n\n"
            f"終點辨識：{destination_match['display_name']}"
        )
        st.caption("地址定位資料 © OpenStreetMap contributors")

    cmp = result['comparison']
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="kpi-card">
        <div class="kpi-value" style="color:#e74c3c">{result['dijkstra']['distance_km']}<span style="font-size:1rem"> km</span></div>
        <div class="kpi-label">Dijkstra 路徑長度</div>
        <div class="kpi-delta" style="color:#e74c3c">暴露: {result['dijkstra']['avg_flood_exposure_cm']} cm</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi-card">
        <div class="kpi-value" style="color:#27ae60">{result['rl']['distance_km']}<span style="font-size:1rem"> km</span></div>
        <div class="kpi-label">RL 避災路徑長度</div>
        <div class="kpi-delta" style="color:#27ae60">暴露: {result['rl']['avg_flood_exposure_cm']} cm</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-card">
        <div class="kpi-value" style="color:#1d8cf8">{cmp['flood_reduction_pct']}<span style="font-size:1rem">%</span></div>
        <div class="kpi-label">淹水暴露降低</div>
        <div class="kpi-delta">多走 {cmp['extra_distance_km']} km</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    route_data = {
        'dijkstra': result['dijkstra'],
        'rl':       result['rl'],
        'origin':   st.session_state['route_origin'],
        'destination': st.session_state['route_dest'],
    }
    m = build_folium_map(typhoon_id, show_fac, show_heatmap, show_track=False,
                          route_data=route_data)
    folium_static(m, height=550)


# ─── Page: Shelter Management ─────────────────────────────────────────────────
def page_shelter_management(typhoon_id, show_fac, show_heatmap, show_track,
                            custom_typhoon=None):
    st.markdown(f'<div class="section-header">收容所容量管理 · {event_title(typhoon_id, custom_typhoon)}</div>',
                unsafe_allow_html=True)

    with st.spinner("計算收容所利用率..."):
        df = cached_shelter_util(typhoon_id)

    total_cap   = df['capacity'].sum()
    total_arr   = df['estimated_arrivals'].sum()
    overloaded  = (df['utilization_pct'] >= 90).sum()
    available   = (df['utilization_pct'] < 60).sum()

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [
        (c1, format_grouped(total_cap), "總收容容量（人）"),
        (c2, format_grouped(total_arr), "預估撤離人數"),
        (c3, f"{overloaded}", "超載收容所"),
        (c4, f"{available}", "容量充裕收容所"),
    ]:
        with col:
            st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{val}</div>
            <div class="kpi-label">{label}</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_l, col_r = st.columns([2, 3])

    with col_l:
        st.markdown('<div class="section-header">收容所狀態列表</div>', unsafe_allow_html=True)
        display = df[['name','capacity','estimated_arrivals','utilization_pct','status']].head(50)
        display.columns = ['名稱','容量','預估到達','使用率%','狀態']
        st.dataframe(display, use_container_width=True, height=480)

    with col_r:
        st.markdown('<div class="section-header">使用率分佈</div>', unsafe_allow_html=True)
        top20 = df.head(20)
        colors = ['#e74c3c' if u >= 90 else ('#f39c12' if u >= 60 else '#27ae60')
                  for u in top20['utilization_pct']]
        fig = go.Figure(go.Bar(
            x=top20['utilization_pct'],
            y=top20['name'].str[:15],
            orientation='h',
            marker_color=colors,
            text=[f"{u:.1f}%" for u in top20['utilization_pct']],
            textposition='outside',
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.62)',
            font_color='#36516d', height=500, margin=dict(t=10,b=10,l=0,r=60),
            xaxis=dict(range=[0,150], title="使用率 %", gridcolor='#dfeaf5'),
            yaxis=dict(autorange='reversed', gridcolor='#edf3f9'),
        )
        fig.add_vline(x=90, line_dash='dash', line_color='#e74c3c',
                       annotation_text='90%警戒線')
        st.plotly_chart(fig, use_container_width=True)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    sel_id, page, show_fac, show_heatmap, show_track, custom_typhoon = render_sidebar()

    if page == "淹水總覽":
        page_flood_overview(
            sel_id, show_fac, show_heatmap, show_track, custom_typhoon
        )
    elif page == "設施風險評估":
        page_facility_risk(
            sel_id, show_fac, show_heatmap, show_track, custom_typhoon
        )
    elif page == "資源分配 (GA+ACO)":
        page_resource_allocation(
            sel_id, show_fac, show_heatmap, show_track, custom_typhoon
        )
    elif page == "避難路徑規劃":
        page_evacuation_routing(
            sel_id, show_fac, show_heatmap, show_track, custom_typhoon
        )
    elif page == "收容所管理":
        page_shelter_management(
            sel_id, show_fac, show_heatmap, show_track, custom_typhoon
        )


if __name__ == "__main__":
    main()
