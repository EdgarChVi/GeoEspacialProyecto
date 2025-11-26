# Tarea 3 - Análisis de cambio de uso del suelo basado en imágenes RGB
# Edgar Chaves Villalobos  - C5A022
# PF-3311
# ============================================================

# ----------------------------
# 1) IMPORTS
# ----------------------------
import streamlit as st
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.transform import xy as transform_xy
from shapely.geometry import box
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import base64
from io import BytesIO
from PIL import Image

# ----------------------------
# 2) CONFIGURACIÓN BÁSICA STREAMLIT
# ----------------------------
st.set_page_config(
    page_title="Cambio de uso del suelo - San Ramón",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tema oscuro 
st.markdown("""
<style>
:root { color-scheme: dark; }
</style>
""", unsafe_allow_html=True)

st.title("Análisis de cambio por color (RGB) — San Ramón (2006 vs 2015)")

st.write("El objetivo del proyecto en desarrollo, o con los datos obtenidos, es lograr determinar los " \
            "cambios en la geografía de una zona con el paso del tiempo. Por ejemplo, si en una zona forestal, el bosque " \
            "fue cortado, alguna zona fue urbanizada, o cualquier cambio significante que se presente.")


# ----------------------------
# 3) RUTAS, CLASES Y PALETAS
# ----------------------------
RUTA_2006 = "data/SanRamon_06-07.tif"
RUTA_2015 = "data/SanRamon_15-18.tif"

CLASSES = ["Suelo", "Agua/Sombra", "Vegetación", "Urbano/Construcción", "Indefinido"]

CMAPS = {
    "Vegetación": mcolors.LinearSegmentedColormap.from_list("veg", ["#F5FFF5", "#021402"]),     # verde claro → oscuro
    "Suelo": mcolors.LinearSegmentedColormap.from_list("soil", ["#FAECD3", "#4B2204"]),    # arena → café
    "Urbano/Construcción": mcolors.LinearSegmentedColormap.from_list("urb", ["#E6E6E6", "#411212"]), # gris claro → oscuro
    "Agua/Sombra": mcolors.LinearSegmentedColormap.from_list("water", ["#89AFD4", "#232379"]),  # azul claro → profundo
    "Indefinido": None
}

CLASS_COLORS = {
    "Suelo":                (210, 140,  80),   # marrón
    "Agua/Sombra":          ( 30,  30,  60),   # azul oscuro / sombra
    "Vegetación":           ( 34, 139,  34),   # verde bosque
    "Urbano/Construcción":  (200, 200, 200),   # gris claro
    "Indefinido":           (255, 255, 255),   # blanco
}

# ----------------------------
# 3.1) PARÁMETROS FIJOS DE CLASIFICACIÓN
# ----------------------------
DEFAULT_PARAMS = {
    "exg_thr": 0,     # ExG = 2G - R - B  (escala 0–255)
    "hue_tol": 15,     # tolerancia alrededor de 60° (verde)
    "sat_min": 0.07,   # S mínima para vegetación
    "v_dark": 0.07,    # V < v_dark → Agua/Sombra
    "sat_urban": 0.07, # S < sat_urban & V >= v_bright → Urbano
    "v_bright": 0.07,  # brillo mínimo urbano
    "sat_soil": 0.07,  # S mínima para Suelo
}

params = DEFAULT_PARAMS 

# ----------------------------
# 4) FUNCIONES — AJUSTE VISUAL (SATURACIÓN)
# ----------------------------
def ajustar_saturacion(arr, factor=1.2):
    """
    Aumenta o reduce la saturación (S) de una imagen RGB.
    arr: np.ndarray (bands, rows, cols), con valores 0–255.
    factor > 1 → más saturación | factor < 1 → menos saturación.
    """
    # Solo aplicar a las tres primeras bandas
    rgb = np.transpose(arr[:3], (1, 2, 0)).astype(np.float32) / 255.0  # (rows, cols, 3)
    hsv = rgb_to_hsv(rgb)

    # Aumentar saturación
    hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 1)

    # Reconstruir RGB
    rgb_sat = hsv_to_rgb(hsv)
    arr_out = (np.transpose(rgb_sat, (2, 0, 1)) * 255).astype(np.uint8)

    # Mantener banda alpha
    if arr.shape[0] == 4:
        arr_out = np.vstack([arr_out, arr[3:4]])

    return arr_out

# ----------------------------
# 5) FUNCIONES — CLASIFICACIÓN
# ----------------------------
def classify_rgb_image(arr, params):
    """
    Clasifica TODO el raster por píxel usando las mismas reglas de clasificar_rgb_df.
    Devuelve array de labels (rows, cols) y métricas intermedias (R,G,B,H,S,V,ExG).
    """
    bands, rows, cols = arr.shape
    R = arr[0].astype(np.float32)
    G = arr[1].astype(np.float32)
    B = arr[2].astype(np.float32)

    valid = arr[3] > 0 if bands >= 4 else np.ones((rows, cols), dtype=bool)

    rgb01 = np.stack([R, G, B], axis=-1) / 255.0
    hsv = rgb_to_hsv(rgb01)
    H = hsv[..., 0] * 360.0
    S = hsv[..., 1]
    V = hsv[..., 2]
    exg = 2*G - R - B

    veg   = (exg >= params["exg_thr"]) | ((np.abs(H-60) <= params["hue_tol"]) & (S >= params["sat_min"]))
    water = (V < params["v_dark"])
    urban = (S < params["sat_urban"]) & (V >= params["v_bright"])
    soil  = (R > G) & (R > B) & (S >= params["sat_soil"])

    labels = np.full((rows, cols), CLASSES.index("Indefinido"), dtype=np.int16)
    labels[soil]  = CLASSES.index("Suelo")
    labels[urban] = CLASSES.index("Urbano/Construcción")
    labels[veg]   = CLASSES.index("Vegetación")
    labels[water] = CLASSES.index("Agua/Sombra")
    labels[~valid] = -1

    
    return labels, (R, G, B, H, S, V, exg)

def clasificar_rgb_df(df, params):
    """
    Clasificación rápida basada en HSV/ExG:
      - Vegetación: ExG = 2G - R - B >= exg_thr  O  (H en 60±hue_tol y S>=sat_min)
      - Agua/Sombra: V < v_dark
      - Urbano/Construcción: S < sat_urban y V >= v_bright
      - Suelo: R dominante (R>G & R>B) y S>=sat_soil
      - Otro: 'Indefinido'
    """
    R = df["R"].values
    G = df["G"].values
    B = df["B"].values

    # Normalizar a 0-1 para HSV
    rgb01 = np.stack([R, G, B], axis=1) / 255.0
    hsv = rgb_to_hsv(rgb01.reshape(-1, 1, 3)).reshape(-1, 3)
    H = hsv[:, 0] * 360.0
    S = hsv[:, 1]
    V = hsv[:, 2]

    # Índice ExG
    exg = 2*G - R - B

    # Reglas
    veg = (exg >= params["exg_thr"]) | ((np.abs(H-60) <= params["hue_tol"]) & (S >= params["sat_min"]))
    water_shadow = (V < params["v_dark"])
    urban = (S < params["sat_urban"]) & (V >= params["v_bright"])
    soil = (R > G) & (R > B) & (S >= params["sat_soil"])

    clase = np.where(veg, "Vegetación",
             np.where(water_shadow, "Agua/Sombra",
             np.where(urban, "Urbano/Construcción",
             np.where(soil, "Suelo", "Indefinido"))))

    df_out = df.copy()
    df_out["Clase"] = clase
    return df_out

# ----------------------------
# 6) FUNCIONES — RENDERIZADO DE MAPAS
# ----------------------------
def labels_to_rgb(labels):
    """
    Convierte el array de labels a una imagen RGB uint8 (rows, cols, 3).
    Los píxeles -1 (NoData) se hacen blancos.
    """
    h, w = labels.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    for cls_name, color in CLASS_COLORS.items():
        code = CLASSES.index(cls_name)
        mask = (labels == code)
        out[mask] = np.array(color, dtype=np.uint8)

    # NoData -> blanco
    out[labels == -1] = (255, 255, 255)
    return out

def plot_class_map(labels, bounds, title):
    """
    Dibuja con Matplotlib usando extent para respetar coordenadas del raster.
    bounds: (left, bottom, right, top)
    """
    rgb = labels_to_rgb(labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    left, bottom, right, top = bounds
    ax.imshow(rgb, origin='upper', extent=(left, right, bottom, top))
    ax.set_title(title)
    ax.set_axis_off()

    legend_patches = [Patch(facecolor=np.array(CLASS_COLORS[c]) / 255.0, edgecolor='k', label=c) for c in CLASSES]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8, frameon=True)
    return fig

def render_class_colorramp_simple(arr, labels, bounds, metrics=None):
    """
    Colorea cada clase con un colormap fijo (claro→oscuro) usando una métrica simple:
      - Vegetación: usa G (0–255) para el tono del verde
      - Suelo: usa R (0–255) para el tono del café
      - Urbano: usa V (0–1) → multiplicado por 255 (más brillante, más oscuro)
      - Agua/Sombra: usa (1 - V) → 0–1 → *255 (más oscuro = agua más profunda)
      - Indefinido: blanco
    """
    R, G, B, H, S, V, exg = metrics
    h, w = labels.shape
    out = np.ones((h, w, 3), dtype=np.float32)  # arranca en blanco

    # Máscaras
    m_veg   = labels == CLASSES.index("Vegetación")
    m_soil  = labels == CLASSES.index("Suelo")
    m_urb   = labels == CLASSES.index("Urbano/Construcción")
    m_water = labels == CLASSES.index("Agua/Sombra")
    m_undef = labels == CLASSES.index("Indefinido")
    m_nodata= labels == -1

    # Vegetación → G en [0..255]
    if m_veg.any():
        s = np.clip(G/255.0, 0, 1)
        rgb = CMAPS["Vegetación"](s)[..., :3]
        out[m_veg] = rgb[m_veg]

    # Suelo → R en [0..255]
    if m_soil.any():
        s = np.clip(R/255.0, 0, 1)
        rgb = CMAPS["Suelo"](s)[..., :3]
        out[m_soil] = rgb[m_soil]

    # Urbano → V en [0..1]
    if m_urb.any():
        s = np.clip(V, 0, 1)
        rgb = CMAPS["Urbano/Construcción"](s)[..., :3]
        out[m_urb] = rgb[m_urb]

    # Agua/Sombra → (1 - V) en [0..1]
    if m_water.any():
        s = np.clip(1.0 - V, 0, 1)
        rgb = CMAPS["Agua/Sombra"](s)[..., :3]
        out[m_water] = rgb[m_water]

    # Indefinido y NoData → blanco
    out[m_undef] = (1, 1, 1)
    out[m_nodata]= (1, 1, 1)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 6))
    left, bottom, right, top = bounds
    ax.imshow(out, origin='upper', extent=(left, right, bottom, top))
    ax.set_axis_off()

    legend_handles = []
    for cls in CLASSES:
        if cls == "Indefinido":
            color = (1, 1, 1)
        else:
            color = CMAPS[cls](0.6)[0:3]  # color medio del gradiente
        legend_handles.append(Patch(facecolor=color, edgecolor='k', label=cls))
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.15), fontsize=8, frameon=True, ncol=len(legend_handles))
    plt.tight_layout()

    return fig

def difference_map(labels06, labels15):
    """
    Devuelve una imagen binaria (uint8) donde:
      0 = negro (misma clase)
      255 = blanco (clase distinta)
    Ignora NoData: si cualquiera de los dos es -1, saldrá 255 (tratado como "diferente")
    """
    h, w = labels06.shape
    same = (labels06 == labels15) & (labels06 != -1) & (labels15 != -1)
    diff = ~same
    out = np.zeros((h, w), dtype=np.uint8)
    out[diff] = 255
    return out

def plot_diff_map(diff_img, bounds, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    left, bottom, right, top = bounds
    ax.imshow(diff_img, cmap='gray', vmin=0, vmax=255, origin='upper', extent=(left, right, bottom, top))
    ax.set_title(title)
    ax.set_axis_off()
    # leyenda manual: negro=igual, blanco=diferente
    legend_patches = [Patch(facecolor='black', edgecolor='k', label='Misma clase'),
                      Patch(facecolor='white', edgecolor='k', label='Clase diferente')]
    ax.legend(handles=legend_patches, loc="lower center", bbox_to_anchor=(0.5, -0.15), fontsize=8, frameon=True, ncol=len(legend_patches))
    plt.tight_layout()
    return fig

def plot_rgb_map(arr, bounds, title):
    """
    Muestra el raster RGB original (bandas 1–3) respetando bounds.
    Si hay banda alpha (=0), pinta esos píxeles como blanco.
    """
    rgb = np.transpose(arr[:3], (1, 2, 0)).astype(np.uint8)  # (H,W,3)
    if arr.shape[0] >= 4:
        alpha = arr[3] == 0
        if alpha.any():
            rgb = rgb.copy()
            rgb[alpha] = 255

    fig, ax = plt.subplots(figsize=(7, 6))
    left, bottom, right, top = bounds
    ax.imshow(rgb, origin='upper', extent=(left, right, bottom, top))
    ax.set_title(title)
    ax.axis('off')
    return fig

# ----------------------------
# 7) FUNCIONES — UTILIDADES RASTER / MUESTRAS / ESTADÍSTICAS
# ----------------------------
@st.cache_data(show_spinner=False)
def cargar_raster(path):
    """Devuelve SOLO datos serializables: array y metadatos simples."""
    with rasterio.open(path) as ds:
        arr = ds.read()  # (bands, rows, cols)
        meta = {
            "transform": tuple(ds.transform),                  # Affine -> tuple
            "crs": ds.crs.to_string() if ds.crs else None,     # str
            "bounds": (ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top),
            "res": ds.res,                                     # (xres, yres)
            "width": ds.width,
            "height": ds.height,
            "count": ds.count,
        }
    return arr, meta

@st.cache_data(show_spinner=False)
def raster_bounds_gdf(path):
    with rasterio.open(path) as src:
        b = src.bounds
        crs = src.crs
    poly = box(b.left, b.bottom, b.right, b.top)
    return gpd.GeoDataFrame({"name": [path]}, geometry=[poly], crs=crs)

def sample_pixels(arr, transform_tuple, n=5000, seed=42):
    """Muestra aleatoria con coords usando transform (serializable)."""
    rng = np.random.default_rng(seed)
    bands, rows, cols = arr.shape

    # máscara alpha si existe
    if bands >= 4:
        alpha = arr[3]
        valid = alpha > 0
    else:
        valid = np.ones((rows, cols), dtype=bool)

    valid_idx = np.argwhere(valid)
    if valid_idx.shape[0] == 0:
        return pd.DataFrame(columns=["x", "y", "R", "G", "B"])

    n = min(n, valid_idx.shape[0])
    sel = rng.choice(valid_idx.shape[0], size=n, replace=False)
    picks = valid_idx[sel]  # (row, col)

    r = arr[0, picks[:, 0], picks[:, 1]].astype(np.float32)
    g = arr[1, picks[:, 0], picks[:, 1]].astype(np.float32)
    b = arr[2, picks[:, 0], picks[:, 1]].astype(np.float32)

    # reconstruir Affine a partir de la tupla
    transform = Affine(*transform_tuple)
    xs, ys = transform_xy(transform, picks[:, 0], picks[:, 1], offset='center')

    return pd.DataFrame({"x": xs, "y": ys, "R": r, "G": g, "B": b})

def histograma_rgb(arr, titulo):
    r = arr[0].ravel()
    g = arr[1].ravel()
    b = arr[2].ravel()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(r, bins=50, alpha=0.5, label='Rojo')
    ax.hist(g, bins=50, alpha=0.5, label='Verde')
    ax.hist(b, bins=50, alpha=0.5, label='Azul')
    ax.set_title(titulo)
    ax.set_xlabel("Intensidad (0–255)")
    ax.set_ylabel("Frecuencia")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig, clear_figure=True)

def delta_rgb_mean(arr2006, arr2015):
    """Promedio de diferencia absoluta en RGB (canales 1-3)."""
    d = np.abs(arr2015[:3].astype(np.float32) - arr2006[:3].astype(np.float32))
    return d.mean(axis=0)  # (rows, cols)

# ----------------------------
# 8) PANEL LATERAL (CONTROLES)
# ----------------------------
st.sidebar.header("Controles")

anio = st.sidebar.radio("Año para visualizar histograma y tabla:", ["2006–2007", "2015–2018"], index=0)
n_muestra = st.sidebar.slider("Tamaño de muestra de píxeles (tabla)", 500, 20000, 5000, 500)


# ----------------------------
# 9) CARGA DE DATOS
# ----------------------------
try:
    arr06, meta06 = cargar_raster(RUTA_2006)
    arr15, meta15 = cargar_raster(RUTA_2015)
    # Ajuste de saturación (ejemplo aplicado a 2006-2007)
    arr06 = ajustar_saturacion(arr06, factor=1.8)  # 80% más saturado
    arr15 = ajustar_saturacion(arr15, factor=1.01)
except Exception as e:
    st.error(f"No se pudieron cargar los GeoTIFF. Revisar existencia de archivos. Detalle: {e}")
    st.stop()

# ----------------------------
# 10) SECCIÓN 1 — TABLA INTERACTIVA (MUESTRA)
# ----------------------------
st.header("Tabla interactiva — Muestra de píxeles clasificados")
st.write("Se muestra una muestra aleatoria de píxeles válidos (no NoData) del raster seleccionado. " \
            "Donde se muestra la ubicación (x,y) y los valores RGB originales junto con la clase asignada según las reglas definidas.")

if anio.startswith("2006"):
    df_px = sample_pixels(arr06, meta06["transform"], n=n_muestra)
else:
    df_px = sample_pixels(arr15, meta15["transform"], n=n_muestra)

if df_px.empty:
    st.warning("No se pudieron muestrear píxeles.")
else:
    df_cls = clasificar_rgb_df(df_px, params)
    st.dataframe(df_cls, use_container_width=True, hide_index=True)


# ----------------------------
# 11) SECCIÓN 2 — HISTOGRAMA RGB
# ----------------------------
st.header("Gráfico estadístico — Histograma RGB")
if anio.startswith("2006"):
    histograma_rgb(arr06, "Distribución RGB — 2006–2007")
else:
    histograma_rgb(arr15, "Distribución RGB — 2015–2018")



# ----------------------------
# 12) SECCIÓN 3 — MAPA INTERACTIVO (FOOTPRINTS)
# ----------------------------
st.header("Mapa interactivo — Ubicación de imágenes.")

fp06 = raster_bounds_gdf(RUTA_2006)
fp15 = raster_bounds_gdf(RUTA_2015)

# reproyectar a WGS84 para web
fp06_wgs = fp06.to_crs(4326)
fp15_wgs = fp15.to_crs(4326)

# centro del mapa: centro del footprint 2015
centroid = fp15_wgs.geometry.iloc[0].centroid
m = folium.Map(location=[centroid.y, centroid.x], zoom_start=12, tiles="CartoDB Positron")

folium.GeoJson(fp06_wgs.__geo_interface__, name="2006–2007",
               style_function=lambda x: {"color": "#1f77b4", "weight": 3, "fillOpacity": 0.05}).add_to(m)
folium.GeoJson(fp15_wgs.__geo_interface__, name="2015–2018",
               style_function=lambda x: {"color": "#d62728", "weight": 3, "dashArray": "5,5", "fillOpacity": 0.05}).add_to(m)

def pil_to_base64(img):
    """Convierte una imagen PIL a base64 PNG, apta para Folium ImageOverlay."""
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return "data:image/png;base64," + encoded

def tif_to_png_scaled(arr, max_width=1024):
    """
    Convierte un raster RGB (bands, rows, cols) en una imagen PIL escalada.
    max_width define el ancho máximo para mejor performance en Streamlit Cloud.
    """
    rgb = np.transpose(arr[:3], (1, 2, 0)).astype(np.uint8)  # (H,W,3)
    img = Image.fromarray(rgb)
    w, h = img.size

    if w > max_width:
        scale = max_width / w
        new_size = (max_width, int(h * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    return img

img06 = tif_to_png_scaled(arr06)  
img15 = tif_to_png_scaled(arr15)

# ----------------------------
# Limites para ImageOverlay
# ----------------------------

b06 = fp06_wgs.total_bounds
b15 = fp15_wgs.total_bounds

bounds06 = [[b06[1], b06[0]], [b06[3], b06[2]]]  # [[sur, este], [norte, oeste]]
bounds15 = [[b15[1], b15[0]], [b15[3], b15[2]]]

# ----------------------------
# Agregar imágenes como capas raster
# ----------------------------
folium.raster_layers.ImageOverlay(
    image=pil_to_base64(img06),
    bounds=bounds06,
    name="Ortofoto 2006–2007 (preview)",
    opacity=0.75,
    interactive=True,
    cross_origin=False
).add_to(m)

folium.raster_layers.ImageOverlay(
    image=pil_to_base64(img15),
    bounds=bounds15,
    name="Ortofoto 2015–2018 (preview)",
    opacity=0.75,
    interactive=True,
    cross_origin=False
).add_to(m)


folium.LayerControl(collapsed=False).add_to(m)
st_folium(m, height=500, use_container_width=True)

# ----------------------------
# 13) SECCIÓN 4 — CAMBIO TEMPORAL (ΔRGB) + MAPAS
# ----------------------------

# Factor de reducción solo para mapas 
FACTOR_MAP = 8  

arr06_map = arr06[:, ::FACTOR_MAP, ::FACTOR_MAP]
arr15_map = arr15[:, ::FACTOR_MAP, ::FACTOR_MAP]

st.header("Cambio temporal — Métrica ΔRGB y transición de clases")

st.write("Se muestra primero las imagenes originales y sus clasificaciones respectivas, " \
            "y luego un mapa de diferencias donde se indica en blanco los píxeles que cambiaron de clase ")

# ΔRGB (promedio absoluto de diferencias por píxel)
delta = delta_rgb_mean(arr06, arr15)  # (rows, cols)

st.subheader("Mapas de clasificación completa y diferencias")

# Clasificación imágenes reducidas para mapas
labels06_map, metrics06_map = classify_rgb_image(arr06_map, params)
labels15_map, metrics15_map = classify_rgb_image(arr15_map, params)

# --- Fila superior: originales ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("2006 — Original")
    fig_o6 = plot_rgb_map(arr06, meta06["bounds"], " ")
    st.pyplot(fig_o6, clear_figure=True)
with col2:
    st.subheader("2015 — Original")
    fig_o15 = plot_rgb_map(arr15, meta15["bounds"], " ")
    st.pyplot(fig_o15, clear_figure=True)

# --- Fila inferior: clasificados ---
col3, col4 = st.columns(2)
with col3:
    st.subheader("2006 — Clasificación")
    fig_c6 = render_class_colorramp_simple(arr06_map, labels06_map, meta06["bounds"], metrics=metrics06_map)
    st.pyplot(fig_c6, clear_figure=True)
with col4:
    st.subheader("2015 — Clasificación")
    fig_c15 = render_class_colorramp_simple(arr15_map, labels15_map, meta15["bounds"], metrics=metrics15_map)
    st.pyplot(fig_c15, clear_figure=True)

# --- Debajo: diferencias en B/N ---
st.subheader("Diferencias de clase — Negro: igual, Blanco: distinto")
diff_img = difference_map(labels06_map, labels15_map)
fig_diff = plot_diff_map(diff_img, meta06["bounds"], " ")
st.pyplot(fig_diff, clear_figure=True)

