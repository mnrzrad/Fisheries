# app.py
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import numpy as np
import re
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from geopy.distance import geodesic
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# -------------------------
# Streamlit page config
# -------------------------
st.set_page_config(
    page_title="Fiscalização Marítima",
    page_icon="⚓",
    layout="wide"
)

# Sidebar font-size tweak
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] * { font-size: 11px !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Helper functions
# -------------------------
def dms_to_dd(dms_str):
    """Convert DMS (e.g., 39º22´8.90N) to decimal degrees."""
    if pd.isna(dms_str):
        return None
    dms_str = str(dms_str).strip()
    regex = r"(\d+)º(\d+)´([\d\.]+)([NSEW])"
    match = re.match(regex, dms_str)
    if not match:
        return None
    deg, minutes, seconds, direction = match.groups()
    dd = float(deg) + float(minutes) / 60 + float(seconds) / 3600
    if direction in ["S", "W"]:
        dd *= -1
    return dd

def escolher_clusters(coords, max_clusters=8):
    """Pick the optimal number of clusters via silhouette score."""
    if len(coords) < 2:
        return 1
    melhor_k, melhor_score = 2, -1
    for k in range(2, min(max_clusters, len(coords)) + 1):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(coords)
        labels = kmeans.labels_
        score = silhouette_score(coords, labels)
        if score > melhor_score:
            melhor_score = score
            melhor_k = k
    return melhor_k

def compute_distance_matrix(points):
    """Compute geodesic distance matrix in nautical miles."""
    n = len(points)
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_mat[i][j] = geodesic(
                    (points[i][0], points[i][1]), (points[j][0], points[j][1])
                ).nautical
    return dist_mat

def solve_tsp_or_tools(distance_matrix):
    """Solve TSP exactly using OR-Tools and return route order."""
    n = len(distance_matrix)
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)  # depot = 0 (port)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return int(distance_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)] * 1000)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 5

    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        index = routing.Start(0)
        route = []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))  # return to depot
        return route
    else:
        return list(range(n))  # fallback

# -------------------------
# Load data  (com cache)
# -------------------------
@st.cache_data(show_spinner=False)
def load_data():
    df0 = pd.read_csv(r"data/final_fiscrep_anonimized.csv")
    df0["lat_dd"] = df0["Latitude"].apply(dms_to_dd)
    df0["lon_dd"] = df0["Longitude"].apply(dms_to_dd)
    ports0 = pd.read_csv(r"data/stations.csv")
    return df0, ports0

df, ports = load_data()

# -------------------------
# UI Layout
# -------------------------
st.markdown(
    "<h1 style='text-align: center; color: navy;'>⚓ Fiscalização Marítima</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="text-align: center; font-size:16px;">
    <b>Autores:</b> Ricardo Moura e Mina Norouzirad <br>
    <b>Email:</b> rp.moura@fct.unl.pt • m.norouzirad@fct.unl.pt <br>
    <b>Referência:</b> Baseado em 
    <a href="https://www.nature.com/articles/s41597-024-03088-4" target="_blank">
    <i>Nature Scientific Data (2024)</i></a>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")
st.sidebar.subheader("ℹ️ Sobre as variáveis")
st.sidebar.markdown(
    """
    <div style="font-size:11px; color:gray; line-height:1.3; margin-top:0.5em;">
    <b>Latitude / Longitude</b>: localização (graus decimais).<br>
    <b>Vessel_Type</b>: tipo de embarcação.<br>
    <b>LOA</b>: comprimento total.<br>
    <b>Main fishing gear</b>: arte de pesca principal.<br>
    <b>Porto</b>: ponto de partida da fiscalização.<br>
    <b>Raio (MN)</b>: distância máxima a considerar.<br>
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown("---")
st.sidebar.subheader("Filtros")

# Filter by Result
result_filter = st.sidebar.radio("Resultado da fiscalização:", ["Todos", "Apenas PRESUM"])
if result_filter == "Apenas PRESUM":
    df = df[df["Result"].astype(str).str.strip().str.upper() == "PRESUM"]

# Filter by Vessel Type
vessel_options = df["Vessel_Type"].dropna().unique().tolist()
vessel_options.sort()
selected_vessels = st.sidebar.multiselect("Tipos de embarcação:", vessel_options)
df_filt = df[df["Vessel_Type"].isin(selected_vessels)] if selected_vessels else df.copy()

# Filter by Main Fishing Gear
gear_options = df_filt["Main fishing gear"].dropna().unique().tolist()
gear_options.sort()
selected_gears = st.sidebar.multiselect("Artes de pesca (Main fishing gear):", gear_options)
if selected_gears:
    df_filt = df_filt[df_filt["Main fishing gear"].isin(selected_gears)]

# Minimum LOA
min_loa = st.sidebar.number_input("LOA mínima (m)", min_value=0, value=0, step=1)
df_filt = df_filt[pd.to_numeric(df_filt["LOA"], errors="coerce") >= min_loa]

# Port selection
porto_options = ports["Porto"].dropna().unique()
porto_sel = st.sidebar.selectbox("Porto base:", porto_options)
porto_row = ports[ports["Porto"] == porto_sel].iloc[0]
porto_lat, porto_lon = float(porto_row["lat"]), float(porto_row["lon"])

# Radius slider
raio_nm = st.sidebar.slider("Raio (milhas náuticas)", 1, 54, 16, 1)

# Cluster mode
cluster_mode = st.sidebar.radio("Clusters:", ["Automático (Silhouette)", "Número fixo"])
n_clusters_fixed = st.sidebar.slider("Número de clusters", 1, 10, 5, 1) if cluster_mode == "Número fixo" else None

# -------------------------
# Main content
# -------------------------
st.subheader("📊 Resumo dos filtros")
col1, col2, col3 = st.columns(3)
col1.metric("Embarcações filtradas", len(df_filt))
col2.metric("Porto selecionado", porto_sel)
col3.metric("Raio (MN)", raio_nm)

# Filter vessels near port  (robusto a NaN)
valid_ll = df_filt["lat_dd"].notna() & df_filt["lon_dd"].notna()
distances = np.full(len(df_filt), np.inf, dtype=float)
if valid_ll.any():
    latlon = df_filt.loc[valid_ll, ["lat_dd", "lon_dd"]].to_numpy()
    distances[valid_ll.values] = [
        geodesic((lat, lon), (porto_lat, porto_lon)).nautical for lat, lon in latlon
    ]
mask = distances <= raio_nm
df_near = df_filt[mask]

st.subheader(f"🚢 Embarcações próximas ({raio_nm} MN)")
if not df_near.empty:
    m2 = folium.Map(location=[porto_lat, porto_lon], zoom_start=9)
    folium.Marker(
        [porto_lat, porto_lon],
        popup=f"Porto: {porto_sel}",
        icon=folium.Icon(color="green", icon="anchor", prefix="fa"),
    ).add_to(m2)
    for _, row in df_near.iterrows():
        color = "red" if str(row["Result"]).strip().upper() == "PRESUM" else "blue"
        folium.CircleMarker(
            [row["lat_dd"], row["lon_dd"]],
            radius=4,
            popup=f"{row['Local_Name']} ({row['Vessel_Type']}) | Gear: {row['Main fishing gear']} | LOA: {row['LOA']} | Result: {row['Result']}",
            color=color, fill=True, fill_opacity=0.7,
        ).add_to(m2)
    st_folium(m2, width=800, height=500)
else:
    st.info("Nenhuma embarcação encontrada neste raio.")

# -------------------------
# Route calculation
# -------------------------
if st.button("📍 Calcular rota de fiscalização"):
    st.session_state["show_route"] = True

if st.session_state.get("show_route", False) and not df_near.empty:
    coords = df_near[["lat_dd", "lon_dd"]].dropna().values
    if len(coords) < 2:
        st.warning("Poucas embarcações para formar rota.")
    else:
        # Determine number of clusters
        n_clusters = escolher_clusters(coords) if cluster_mode == "Automático (Silhouette)" else n_clusters_fixed
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(coords)
        centers = kmeans.cluster_centers_

        # Distance matrix (port + centers)
        points = np.vstack([[porto_lat, porto_lon], centers])
        dist_mat = compute_distance_matrix(points)

        # Solve optimal TSP
        route = solve_tsp_or_tools(dist_mat)
        route_coords = [points[i] for i in route]

        # Map
        m3 = folium.Map(location=[porto_lat, porto_lon], zoom_start=8)
        folium.Marker(
            [porto_lat, porto_lon],
            popup=f"Porto: {porto_sel} (Início/Fim)",
            icon=folium.Icon(color="green", icon="anchor", prefix="fa"),
        ).add_to(m3)

        rota_lista = [("Porto", porto_lat, porto_lon, 0)]
        for step, idx in enumerate(route[1:], start=1):
            if idx == 0:
                continue
            lat, lon = points[idx]
            prev = points[route[step-1]]
            dist_leg = geodesic((prev[0], prev[1]), (lat, lon)).nautical
            rota_lista.append((f"Zona {step}", lat, lon, round(dist_leg, 1)))
            folium.Marker(
                [lat, lon],
                popup=f"Passo {step}: Zona densa",
                icon=folium.DivIcon(html=f"<div style='font-size:14pt;font-weight:bold;color:darkred'>⚓ {step}</div>"),
            ).add_to(m3)

        folium.PolyLine(route_coords, color="blue", weight=3, dash_array="5, 10").add_to(m3)

        st.subheader("🗺️ Rota de fiscalização sugerida")
        st_folium(m3, width=800, height=500)

        st.subheader("📍 Lista de pontos a fiscalizar")
        rota_df = pd.DataFrame(rota_lista, columns=["Local", "Latitude", "Longitude", "Distância (NM)"])
        st.dataframe(rota_df, width="stretch")

# -------------------------
# Concluding note (sempre visível)
# -------------------------
st.markdown("---")
st.markdown(
    """
    <div style="font-size:16px; text-align:justify; line-height:1.5;">
    <b>Fiscalização Marítima</b><br><br>
    Este trabalho é inspirado no artigo publicado na <i>Nature Scientific Data (2024)</i>, que disponibiliza dados anonimizados de fiscalizações realizadas em embarcações, incluindo variáveis como localização, tipo de navio, dimensões (ex.: LOA), artes de pesca utilizadas e resultados da inspeção (com destaque para os casos <b>PRESUM</b>, presumíveis infrações).<br><br>
    O objetivo deste <b>dashboard interativo</b> é identificar as áreas com maior densidade de embarcações suspeitas, otimizando as rotas de fiscalização. 
    Com este painel, a <b>Marinha</b> pode priorizar áreas de intervenção, planear patrulhas mais eficientes e reduzir custos operacionais.
    </div>
    """,
    unsafe_allow_html=True
)
