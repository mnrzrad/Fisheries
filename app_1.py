# app.py
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import numpy as np
import re
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial import distance_matrix
import networkx as nx

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

def euclidean_km(lat1, lon1, lat2, lon2):
    """Approximate distance (km) using Euclidean formula with latitude correction."""
    km_per_deg_lat = 111
    km_per_deg_lon = 111 * np.cos(np.radians(lat2))
    dlat = lat1 - lat2
    dlon = lon1 - lon2
    return np.sqrt((dlat * km_per_deg_lat) ** 2 + (dlon * km_per_deg_lon) ** 2)

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

# -------------------------
# Load data
# -------------------------
df = pd.read_csv(r"data\final_fiscrep_anonimized.csv")
df["lat_dd"] = df["Latitude"].apply(dms_to_dd)
df["lon_dd"] = df["Longitude"].apply(dms_to_dd)

ports = pd.read_csv(r"data\stations.csv")

# -------------------------
# Streamlit UI
# -------------------------
st.title("Fiscalização Marítima")

# --- Filter by Result (PRESUM) ---
result_filter = st.selectbox(
    "Filtrar por resultado:",
    ["Todos", "Apenas PRESUM"]
)

if result_filter == "Apenas PRESUM":
    df = df[df["Result"].str.strip().str.upper() == "PRESUM"]

# --- Filter by Vessel Type ---
vessel_options = df["Vessel_Type"].dropna().unique().tolist()
vessel_options.sort()

selected_vessels = st.multiselect(
    "Seleciona os tipos de embarcação a incluir", vessel_options
)

if selected_vessels:
    df_filt = df[df["Vessel_Type"].isin(selected_vessels)]
else:
    df_filt = df.copy()  # if none selected, include all

# --- Minimum LOA filter ---
min_loa = st.number_input("Definir LOA mínima (m)", min_value=0, value=0, step=1)
df_filt = df_filt[pd.to_numeric(df_filt["LOA"], errors="coerce") >= min_loa]

st.write(f"Total de embarcações filtradas: {len(df_filt)}")

# --- Select Port ---
porto_options = ports["Porto"].dropna().unique()
porto_sel = st.selectbox("Seleciona o porto", porto_options)

porto_row = ports[ports["Porto"] == porto_sel].iloc[0]
porto_lat, porto_lon = float(porto_row["lat"]), float(porto_row["lon"])

# --- Radius slider in nautical miles ---
raio_nm = st.slider("Raio em milhas náuticas", min_value=1, max_value=54, value=16, step=1)
raio_km = raio_nm * 1.852

# --- Filter vessels near port ---
distances = euclidean_km(df_filt["lat_dd"].values, df_filt["lon_dd"].values, porto_lat, porto_lon)
mask = distances <= raio_km
df_near = df_filt[mask]

st.subheader(f"Embarcações perto do porto {porto_sel} (até {raio_nm} MN)")

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
            popup=f"{row['Local_Name']} ({row['Vessel_Type']}) - Result: {row['Result']} | LOA: {row['LOA']}",
            color=color,
            fill=True,
            fill_opacity=0.7,
        ).add_to(m2)

    st_folium(m2, width=700, height=500)
else:
    st.info("Nenhuma embarcação próxima encontrada.")

# --- Button to calculate route ---
if st.button("Calcular rota de fiscalização"):
    st.session_state["show_route"] = True

if st.session_state.get("show_route", False):
    if not df_near.empty:
        coords = df_near[["lat_dd", "lon_dd"]].dropna().values
        if len(coords) < 2:
            st.warning("Poucas embarcações para formar rota.")
        else:
            # --- Cluster choice ---
            cluster_mode = st.radio(
                "Modo de clusters:",
                ["Automático (Silhouette)", "Número fixo"]
            )

            if cluster_mode == "Automático (Silhouette)":
                n_clusters = escolher_clusters(coords)
            else:
                n_clusters = st.slider("Número de clusters fixo", min_value=1, max_value=10, value=5)

            # Clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(coords)
            centers = kmeans.cluster_centers_

            # Distance matrix (port + centers)
            points = np.vstack([[porto_lat, porto_lon], centers])
            dist_mat = distance_matrix(points, points)

            G = nx.complete_graph(len(points))
            for i in range(len(points)):
                for j in range(len(points)):
                    if i != j:
                        G[i][j]["weight"] = dist_mat[i, j]

            route = nx.approximation.traveling_salesman_problem(G, cycle=True)
            route_coords = [points[i] for i in route]

            m3 = folium.Map(location=[porto_lat, porto_lon], zoom_start=8)

            # Port marker
            folium.Marker(
                [porto_lat, porto_lon],
                popup=f"Porto: {porto_sel} (Início/Fim)",
                icon=folium.Icon(color="green", icon="anchor", prefix="fa"),
            ).add_to(m3)

            rota_lista = [("Porto", porto_lat, porto_lon)]
            for step, idx in enumerate(route[1:], start=1):
                if idx == 0:
                    continue
                lat, lon = points[idx]
                rota_lista.append((f"Zona {step}", lat, lon))
                folium.Marker(
                    [lat, lon],
                    popup=f"Passo {step}: Zona densa",
                    icon=folium.DivIcon(
                        html=f"""
                        <div style="
                            font-size: 14pt;
                            font-weight: bold;
                            color: darkred;
                            text-align: center;
                        ">
                            ⚓ {step}
                        </div>
                        """
                    ),
                ).add_to(m3)

            folium.PolyLine(
                route_coords, color="blue", weight=3, dash_array="5, 10"
            ).add_to(m3)

            st.subheader("Rota de fiscalização sugerida")
            st_folium(m3, width=700, height=500)

            st.subheader("Lista ordenada de pontos a fiscalizar")
            rota_df = pd.DataFrame(rota_lista, columns=["Local", "Latitude", "Longitude"])
            st.dataframe(rota_df, use_container_width=True)
    else:
        st.warning("Não há embarcações próximas para gerar uma rota.")
