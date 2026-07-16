import os
import json
import random
import math
import io
import urllib.request
import numpy as np
from flask import Flask, jsonify, request, send_file, render_template_string
from PIL import Image, ImageDraw

try:
    import rasterio
    from rasterio.windows import from_bounds
    HAS_RASTERIO = True
except Exception as e:
    HAS_RASTERIO = False
    print(f"Advertencia: No se pudo importar rasterio ({e}). El recorte dinámico post-evento no estará disponible.")

try:
    from pyproj import Transformer
    HAS_PYPROJ = True
except Exception as e:
    HAS_PYPROJ = False
    print(f"Advertencia: No se pudo importar pyproj ({e}). El recorte dinámico pre-evento no estará disponible.")

# Rutas de los archivos relativas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuración de las tres zonas de trabajo
ZONES = {
    'TANAGUARENA': {
        'geojson': os.path.join(BASE_DIR, "TANAGUARENA_revisado.geojson"),
        'tif': os.path.join(BASE_DIR, "ORTOMOSAICO_TANAGUARENA.tif"),
        'display': 'Tanaguarena'
    },
    'MACUTO': {
        'geojson': os.path.join(BASE_DIR, "MACUTO_revisada.geojson"),
        'tif': os.path.join(BASE_DIR, "ORTOMOSAICO_MACUTO03JUL26.tif"),
        'display': 'Macuto'
    },
    'CARABALLEDA': {
        'geojson': os.path.join(BASE_DIR, "Caraballeda_revsiada.geojson"),
        'tif': os.path.join(BASE_DIR, "MOSAICO_CARABALLEDA.tif"),
        'display': 'Caraballeda'
    }
}
ACTIVE_ZONE = 'TANAGUARENA'

# Carpeta de caché para recortes
CROPS_DIR = os.path.join(BASE_DIR, "crops")
os.makedirs(CROPS_DIR, exist_ok=True)

app = Flask(__name__)

# Transformadores de coordenadas
if HAS_PYPROJ:
    to_mercator = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    to_geographic = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
else:
    to_mercator = None
    to_geographic = None

# Cargar GeoJSON en memoria según zona activa
def load_geojson():
    path = ZONES[ACTIVE_ZONE]['geojson']
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el GeoJSON en {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_geojson(data):
    path = ZONES[ACTIVE_ZONE]['geojson']
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    if os.path.exists(path):
        os.remove(path)
    os.rename(temp_path, path)

# Helpers para tiles de Google Satellite
def latlon_to_tile(lon: float, lat: float, zoom: int):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return xtile, ytile

def tile_bounds_3857(x: int, y: int, zoom: int):
    initial_resolution = 2 * math.pi * 6378137 / 256
    origin_shift = 2 * math.pi * 6378137 / 2.0
    res = initial_resolution / (2**zoom)
    tile_size_m = 256 * res
    minx = x * tile_size_m - origin_shift
    maxy = origin_shift - y * tile_size_m
    maxx = (x + 1) * tile_size_m - origin_shift
    miny = origin_shift - (y + 1) * tile_size_m
    return minx, miny, maxx, maxy

def agregar_grilla_y_norte(imagen: Image.Image, etiqueta: str = "") -> Image.Image:
    draw = ImageDraw.Draw(imagen)
    w, h = imagen.size
    
    draw.line([(w//2 - 15, h//2), (w//2 + 15, h//2)], fill=(255, 0, 0), width=2)
    draw.line([(w//2, h//2 - 15), (w//2, h//2 + 15)], fill=(255, 0, 0), width=2)
    draw.rectangle([w//2 - 2, h//2 - 2, w//2 + 2, h//2 + 2], fill=(255, 255, 0))

    cx, cy = w - 30, 30
    r = 15
    draw.ellipse([cx - r - 2, cy - r - 2, cx + r + 2, cy + r + 2], fill=(0, 0, 0, 120), outline=(255, 255, 255), width=1)
    draw.polygon([(cx, cy - r), (cx - 6, cy + 2), (cx + 6, cy + 2)], fill=(255, 0, 0))
    draw.polygon([(cx, cy + r), (cx - 6, cy - 2), (cx + 6, cy - 2)], fill=(200, 200, 200))
    draw.text((cx - 3, cy - r - 12), "N", fill=(255, 255, 255))
    
    if etiqueta:
        draw.rectangle([5, 5, 120, 25], fill=(0, 0, 0, 180))
        draw.text((10, 8), etiqueta, fill=(255, 255, 255))
        
    return imagen

def crop_pre_event(lon, lat, semilado_m=40.0):
    if not HAS_PYPROJ:
        return None
    x_centro, y_centro = to_mercator.transform(lon, lat)
    
    ulx_3857 = x_centro - semilado_m
    uly_3857 = y_centro + semilado_m
    lrx_3857 = x_centro + semilado_m
    lry_3857 = y_centro - semilado_m
    
    ul_lon, ul_lat = to_geographic.transform(ulx_3857, uly_3857)
    lr_lon, lr_lat = to_geographic.transform(lrx_3857, lry_3857)
    
    zoom = 19
    xtile, ytile = latlon_to_tile(lon, lat, zoom)
    
    imgs = []
    import time
    for dy in [-1, 0, 1]:
        row = []
        for dx in [-1, 0, 1]:
            url = f"https://mt1.google.com/vt/lyrs=s&x={xtile+dx}&y={ytile+dy}&z={zoom}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            req = urllib.request.Request(url, headers=headers)
            
            img = None
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(req, timeout=4) as response:
                        img = Image.open(io.BytesIO(response.read()))
                        break
                except Exception as e:
                    time.sleep(0.5)
            
            if img is None:
                img = Image.new('RGB', (256, 256), (30, 34, 42))
                draw = ImageDraw.Draw(img)
                draw.text((10, 120), "Error de Red Google", fill=(180, 180, 180))
            
            row.append(img)
        imgs.append(row)
        
    w_t, h_t = imgs[0][0].size
    merged = Image.new('RGB', (w_t*3, h_t*3))
    for i in range(3):
        for j in range(3):
            merged.paste(imgs[i][j], (j*w_t, i*h_t))
            
    minx_m, _, _, maxy_m = tile_bounds_3857(xtile - 1, ytile - 1, zoom)
    initial_resolution = 2 * math.pi * 6378137 / 256
    res_m = (initial_resolution / (2**zoom))
    
    px_ulx = int((ulx_3857 - minx_m) / res_m)
    px_uly = int((maxy_m - uly_3857) / res_m)
    px_lrx = int((lrx_3857 - minx_m) / res_m)
    px_lry = int((maxy_m - lry_3857) / res_m)
    
    crop = merged.crop((px_ulx, px_uly, px_lrx, px_lry))
    crop = crop.resize((512, 512), Image.Resampling.LANCZOS)
    crop = agregar_grilla_y_norte(crop, "PRE-EVENTO")
    return crop

def crop_post_event(lon, lat, semilado_m=40.0):
    if not HAS_RASTERIO:
        return None
        
    tif_path = ZONES[ACTIVE_ZONE]['tif']
    if not os.path.exists(tif_path):
        return None

    dlat = semilado_m / 111320.0
    dlon = semilado_m / (111320.0 * math.cos(math.radians(lat)))
    
    ulx = lon - dlon
    uly = lat + dlat
    lrx = lon + dlon
    lry = lat - dlat
        
    with rasterio.open(tif_path) as src:
        window = from_bounds(ulx, lry, lrx, uly, src.transform)
        if src.count >= 3:
            data = src.read([1, 2, 3], window=window)
            rgb = np.moveaxis(data, 0, -1)
        else:
            data = src.read(1, window=window)
            rgb = np.stack([data, data, data], axis=-1)
            
        if rgb.dtype != np.uint8:
            dmin, dmax = rgb.min(), rgb.max()
            if dmax > dmin:
                rgb = ((rgb - dmin) / (dmax - dmin) * 255).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)
            
        crop = Image.fromarray(rgb)
        crop = crop.resize((512, 512), Image.Resampling.LANCZOS)
        crop = agregar_grilla_y_norte(crop, "POST-EVENTO")
        return crop

@app.route("/")
def index():
    html = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <title>Orientación de Derrumbes</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg: #0d1117;
                --card-bg: #161b22;
                --text: #c9d1d9;
                --text-bright: #f0f6fc;
                --primary: #58a6ff;
                --primary-hover: #1f6feb;
                --accent: #ff7b72;
                --skip-btn: #30363d;
                --skip-btn-hover: #8b949e;
                --border: #30363d;
            }
            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
                font-family: 'Outfit', sans-serif;
                -webkit-tap-highlight-color: transparent;
            }
            body {
                background-color: var(--bg);
                color: var(--text);
                display: flex;
                flex-direction: column;
                min-height: 100vh;
                padding: 8px;
                align-items: center;
                justify-content: center;
            }
            header {
                width: 100%;
                max-width: 680px;
                text-align: center;
                margin-bottom: 8px;
            }
            h1 {
                font-size: 1.2rem;
                font-weight: 700;
                color: var(--text-bright);
            }
            .progress-container {
                width: 100%;
                background-color: var(--border);
                height: 6px;
                border-radius: 3px;
                overflow: hidden;
                margin-top: 4px;
            }
            .progress-bar {
                background: linear-gradient(90deg, var(--primary), #bc8cff);
                height: 100%;
                width: 0%;
                transition: width 0.3s ease;
            }
            .stats {
                font-size: 0.8rem;
                margin-top: 4px;
                color: #8b949e;
                display: flex;
                justify-content: space-between;
                padding: 0 4px;
            }
            .btn-download {
                background-color: #238636;
                color: #ffffff;
                font-size: 0.78rem;
                font-weight: 700;
                padding: 6px 12px;
                border-radius: 6px;
                border: none;
                cursor: pointer;
                text-decoration: none;
                transition: background-color 0.15s ease;
                box-shadow: 0 2px 4px rgba(0,0,0,0.15);
            }
            .btn-download:hover {
                background-color: #2ea043;
            }
            .zone-selector {
                display: flex;
                gap: 6px;
                margin: 8px 0;
                width: 100%;
                max-width: 680px;
            }
            .zone-btn {
                flex: 1;
                background-color: #21262d;
                border: 1px solid var(--border);
                color: var(--text);
                padding: 8px 0;
                font-size: 0.85rem;
                font-weight: 700;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.15s ease;
            }
            .zone-btn.active {
                background-color: #bc8cff;
                border-color: #bc8cff;
                color: #0d1117;
            }
            .zone-btn:hover:not(.active) {
                background-color: #30363d;
            }
            .main-card {
                background-color: var(--card-bg);
                border: 1px solid var(--border);
                border-radius: 14px;
                width: 100%;
                max-width: 680px;
                padding: 12px;
                box-shadow: 0 6px 20px rgba(0,0,0,0.5);
            }
            .app-layout {
                display: flex;
                flex-direction: row;
                gap: 12px;
                width: 100%;
                align-items: stretch;
            }
            .image-pane {
                flex: 1.2;
                display: flex;
                flex-direction: column;
                gap: 8px;
            }
            .controls-pane {
                flex: 1;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                gap: 8px;
            }
            .toggle-container {
                display: flex;
                background-color: #21262d;
                border-radius: 8px;
                padding: 3px;
                border: 1px solid var(--border);
                width: 100%;
            }
            .toggle-btn {
                flex: 1;
                border: none;
                background: none;
                color: var(--text);
                padding: 8px 0;
                font-size: 0.85rem;
                font-weight: 700;
                border-radius: 6px;
                cursor: pointer;
                transition: all 0.15s ease;
            }
            .toggle-btn.active {
                background-color: var(--primary);
                color: var(--text-bright);
                box-shadow: 0 2px 4px rgba(0,0,0,0.25);
            }
            .img-box {
                position: relative;
                width: 100%;
                aspect-ratio: 1;
                border-radius: 10px;
                overflow: hidden;
                border: 1px solid var(--border);
                background-color: #000;
                cursor: pointer;
            }
            .img-box img {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                object-fit: cover;
                transition: opacity 0.12s ease-in-out;
            }
            .img-hidden {
                opacity: 0;
                pointer-events: none;
            }
            .zoom-trigger {
                position: absolute;
                bottom: 8px;
                right: 8px;
                background-color: rgba(0, 0, 0, 0.75);
                border: 1px solid var(--border);
                color: var(--text-bright);
                padding: 5px 8px;
                border-radius: 6px;
                font-size: 0.72rem;
                font-weight: 600;
                z-index: 5;
                cursor: pointer;
                display: flex;
                align-items: center;
                gap: 3px;
            }
            .zoom-overlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background-color: rgba(13, 17, 23, 0.98);
                z-index: 1000;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                cursor: zoom-out;
                padding: 10px;
            }
            .zoom-overlay img {
                max-width: 96%;
                max-height: 80%;
                object-fit: contain;
                border: 1px solid var(--border);
                border-radius: 8px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.8);
            }
            .zoom-close {
                position: absolute;
                top: 15px;
                right: 15px;
                background-color: var(--skip-btn);
                border: 1px solid var(--border);
                color: var(--text-bright);
                padding: 8px 14px;
                border-radius: 8px;
                font-size: 0.9rem;
                font-weight: 600;
                cursor: pointer;
            }
            .zoom-btn-container {
                position: absolute;
                bottom: 25px;
                display: flex;
                gap: 12px;
                z-index: 1010;
                background-color: #161b22;
                padding: 6px;
                border-radius: 10px;
                border: 1px solid var(--border);
            }
            .zoom-btn {
                background-color: #21262d;
                border: 1px solid var(--border);
                color: var(--text-bright);
                padding: 10px 20px;
                border-radius: 8px;
                font-weight: 700;
                font-size: 0.95rem;
                cursor: pointer;
            }
            .zoom-btn.active {
                background-color: var(--primary);
            }
            .compass-grid {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 6px;
                width: 100%;
                flex-grow: 1;
            }
            .btn {
                background-color: #21262d;
                border: 1px solid var(--border);
                color: var(--text-bright);
                border-radius: 10px;
                font-size: 1rem;
                font-weight: 700;
                cursor: pointer;
                transition: all 0.12s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 3px 5px rgba(0,0,0,0.15);
            }
            .btn:active {
                transform: scale(0.94);
            }
            .btn.dir:hover {
                background-color: var(--primary);
                border-color: var(--primary);
            }
            .btn.center-btn {
                background-color: transparent;
                border: none;
                box-shadow: none;
                cursor: default;
                pointer-events: none;
            }
            .btn.skip {
                background-color: var(--skip-btn);
                padding: 10px 0;
                font-size: 0.85rem;
                color: #ff7b72;
                border-color: #492322;
                border-radius: 8px;
                width: 100%;
                font-weight: 600;
            }
            .btn.skip:hover {
                background-color: #492322;
                color: #ff7b72;
            }
            .loader {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                border: 3px solid var(--border);
                border-top: 3px solid var(--primary);
                border-radius: 50%;
                width: 26px;
                height: 26px;
                animation: spin 1s linear infinite;
                z-index: 10;
            }
            @keyframes spin {
                0% { transform: translate(-50%, -50%) rotate(0deg); }
                100% { transform: translate(-50%, -50%) rotate(360deg); }
            }
            .no-more {
                text-align: center;
                padding: 30px 10px;
                font-size: 1.1rem;
                width: 100%;
            }
            .tap-hint {
                font-size: 0.75rem;
                color: #8b949e;
                text-align: center;
                margin-top: 2px;
            }
        </style>
    </head>
    <body>
        <header>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <h1 id="appTitle">Detección de Derrumbes</h1>
                <a href="/api/download" download class="btn-download">📥 Descargar Shapefile (.zip)</a>
            </div>
            <div class="progress-container">
                <div class="progress-bar" id="progressBar"></div>
            </div>
            <div class="stats">
                <span id="progressText">Cargando...</span>
                <span id="currentId">ID: --</span>
            </div>
        </header>

        <!-- SELECCIÓN DE ZONAS -->
        <div class="zone-selector">
            <button id="zone_TANAGUARENA" class="zone-btn active" onclick="changeZone('TANAGUARENA')">Tanaguarena</button>
            <button id="zone_MACUTO" class="zone-btn" onclick="changeZone('MACUTO')">Macuto</button>
            <button id="zone_CARABALLEDA" class="zone-btn" onclick="changeZone('CARABALLEDA')">Caraballeda</button>
        </div>

        <main class="main-card" id="mainCard">
            <div class="app-layout">
                <!-- PANEL IZQUIERDO: IMAGEN -->
                <div class="image-pane">
                    <div class="toggle-container">
                        <button id="togglePre" class="toggle-btn" onclick="setMode('pre')">PRE</button>
                        <button id="togglePost" class="toggle-btn active" onclick="setMode('post')">POST</button>
                    </div>
                    
                    <div class="img-box" onclick="toggleImages()" title="Alternar Pre/Post">
                        <div class="loader" id="preLoader"></div>
                        <img id="imgPre" src="" alt="Pre-evento" class="img-hidden" onload="checkLoad('pre')" onerror="checkLoad('pre')">
                        <img id="imgPost" src="" alt="Post-evento" onload="checkLoad('post')" onerror="checkLoad('post')">
                        
                        <div class="zoom-trigger" onclick="openZoom(event)">🔍 Zoom</div>
                    </div>
                    <div class="tap-hint">💡 Toca la imagen para alternar PRE/POST</div>
                </div>

                <!-- PANEL DERECHO: COMPASS & SKIP -->
                <div class="controls-pane">
                    <div class="compass-grid">
                        <button class="btn dir" onclick="submitDirection('NO')">NO</button>
                        <button class="btn dir" onclick="submitDirection('N')">N</button>
                        <button class="btn dir" onclick="submitDirection('NE')">NE</button>
                        
                        <button class="btn dir" onclick="submitDirection('O')">O</button>
                        <div class="btn center-btn">🧭</div>
                        <button class="btn dir" onclick="submitDirection('E')">E</button>
                        
                        <button class="btn dir" onclick="submitDirection('SO')">SO</button>
                        <button class="btn dir" onclick="submitDirection('S')">S</button>
                        <button class="btn dir" onclick="submitDirection('SE')">SE</button>
                    </div>
                    
                    <button class="btn skip" onclick="submitDirection('Omitido')">Omitir Edificio</button>
                </div>
            </div>
        </main>

        <!-- MODAL DE ZOOM FULLSCREEN -->
        <div id="zoomModal" class="zoom-overlay" style="display: none;" onclick="closeZoom()">
            <button class="zoom-close" onclick="closeZoom(); event.stopPropagation();">Cerrar</button>
            <img id="zoomImg" src="" alt="Zoom Imagen" onclick="toggleZoomImage(event)">
            <div class="zoom-btn-container" onclick="event.stopPropagation();">
                <button id="zoomTogglePre" class="zoom-btn" onclick="setZoomMode('pre')">PRE</button>
                <button id="zoomTogglePost" class="zoom-btn active" onclick="setZoomMode('post')">POST</button>
            </div>
        </div>

        <script>
            let currentFeature = null;
            let currentMode = 'post';
            let loadedPre = false;
            let loadedPost = false;
            let activeZoneName = 'TANAGUARENA';

            function checkLoad(type) {
                if (type === 'pre') loadedPre = true;
                if (type === 'post') loadedPost = true;
                
                if (loadedPre && loadedPost) {
                    document.getElementById('preLoader').style.display = 'none';
                }
            }

            function setMode(mode) {
                currentMode = mode;
                const imgPre = document.getElementById('imgPre');
                const imgPost = document.getElementById('imgPost');
                const btnPre = document.getElementById('togglePre');
                const btnPost = document.getElementById('togglePost');

                if (mode === 'pre') {
                    imgPre.classList.remove('img-hidden');
                    imgPost.classList.add('img-hidden');
                    btnPre.classList.add('active');
                    btnPost.classList.remove('active');
                } else {
                    imgPre.classList.add('img-hidden');
                    imgPost.classList.remove('img-hidden');
                    btnPre.classList.remove('active');
                    btnPost.classList.add('active');
                }
            }

            function toggleImages() {
                setMode(currentMode === 'pre' ? 'post' : 'pre');
            }

            // Funciones de Zoom
            function openZoom(event) {
                event.stopPropagation();
                const modal = document.getElementById('zoomModal');
                const zoomImg = document.getElementById('zoomImg');
                
                zoomImg.src = currentMode === 'pre' ? document.getElementById('imgPre').src : document.getElementById('imgPost').src;
                modal.style.display = 'flex';
                updateZoomButtons();
            }

            function closeZoom() {
                document.getElementById('zoomModal').style.display = 'none';
            }

            function setZoomMode(mode) {
                setMode(mode);
                const zoomImg = document.getElementById('zoomImg');
                zoomImg.src = mode === 'pre' ? document.getElementById('imgPre').src : document.getElementById('imgPost').src;
                updateZoomButtons();
            }

            function toggleZoomImage(event) {
                event.stopPropagation();
                setZoomMode(currentMode === 'pre' ? 'post' : 'pre');
            }

            function updateZoomButtons() {
                const btnPre = document.getElementById('zoomTogglePre');
                const btnPost = document.getElementById('zoomTogglePost');
                if (currentMode === 'pre') {
                    btnPre.classList.add('active');
                    btnPost.classList.remove('active');
                } else {
                    btnPre.classList.remove('active');
                    btnPost.classList.add('active');
                }
            }

            async function changeZone(zone) {
                try {
                    const res = await fetch(`/api/set_zone/${zone}`);
                    const data = await res.json();
                    if (data.success) {
                        activeZoneName = zone;
                        document.querySelectorAll('.zone-btn').forEach(btn => btn.classList.remove('active'));
                        document.getElementById(`zone_${zone}`).classList.add('active');
                        
                        // Restablecer el card principal por si acaso estaba en no-more
                        document.getElementById('mainCard').innerHTML = `
                            <div class="app-layout">
                                <div class="image-pane">
                                    <div class="toggle-container">
                                        <button id="togglePre" class="toggle-btn" onclick="setMode('pre')">PRE</button>
                                        <button id="togglePost" class="toggle-btn active" onclick="setMode('post')">POST</button>
                                    </div>
                                    <div class="img-box" onclick="toggleImages()" title="Alternar Pre/Post">
                                        <div class="loader" id="preLoader"></div>
                                        <img id="imgPre" src="" alt="Pre-evento" class="img-hidden" onload="checkLoad('pre')" onerror="checkLoad('pre')">
                                        <img id="imgPost" src="" alt="Post-evento" onload="checkLoad('post')" onerror="checkLoad('post')">
                                        <div class="zoom-trigger" onclick="openZoom(event)">🔍 Zoom</div>
                                    </div>
                                    <div class="tap-hint">💡 Toca la imagen para alternar PRE/POST</div>
                                </div>
                                <div class="controls-pane">
                                    <div class="compass-grid">
                                        <button class="btn dir" onclick="submitDirection('NO')">NO</button>
                                        <button class="btn dir" onclick="submitDirection('N')">N</button>
                                        <button class="btn dir" onclick="submitDirection('NE')">NE</button>
                                        <button class="btn dir" onclick="submitDirection('O')">O</button>
                                        <div class="btn center-btn">🧭</div>
                                        <button class="btn dir" onclick="submitDirection('E')">E</button>
                                        <button class="btn dir" onclick="submitDirection('SO')">SO</button>
                                        <button class="btn dir" onclick="submitDirection('S')">S</button>
                                        <button class="btn dir" onclick="submitDirection('SE')">SE</button>
                                    </div>
                                    <button class="btn skip" onclick="submitDirection('Omitido')">Omitir Edificio</button>
                                </div>
                            </div>
                        `;
                        
                        fetchNext();
                    }
                } catch (e) {
                    console.error("Error al cambiar zona", e);
                }
            }

            async function fetchNext() {
                const loader = document.getElementById('preLoader');
                if (loader) loader.style.display = 'block';
                loadedPre = false;
                loadedPost = false;
                
                const imgPre = document.getElementById('imgPre');
                const imgPost = document.getElementById('imgPost');
                if (imgPre) imgPre.src = '';
                if (imgPost) imgPost.src = '';

                try {
                    const res = await fetch('/api/next');
                    const data = await res.json();
                    
                    if (data.finished) {
                        document.getElementById('mainCard').innerHTML = `
                            <div class="no-more">
                                <h2>🎉 ¡Felicidades!</h2>
                                <p style="margin-top: 10px; color: #8b949e;">Todos los edificios de esta zona han sido clasificados.</p>
                            </div>
                        `;
                        document.getElementById('currentId').innerText = 'Completado';
                        updateProgressBar(data.total, data.total);
                        return;
                    }

                    currentFeature = data.feature;
                    document.getElementById('currentId').innerText = `ID: ${currentFeature.properties.id_infraestructura}`;
                    updateProgressBar(data.classified, data.total);

                    setMode('post');

                    const id = currentFeature.properties.id_infraestructura;
                    if (imgPre) imgPre.src = `/api/crop/${id}/pre?t=${Date.now()}`;
                    if (imgPost) imgPost.src = `/api/crop/${id}/post?t=${Date.now()}`;

                } catch (e) {
                    console.error("Error cargando el siguiente elemento", e);
                    alert("Error de conexión al cargar datos.");
                }
            }

            function updateProgressBar(classified, total) {
                const pct = total > 0 ? (classified / total) * 100 : 0;
                document.getElementById('progressBar').style.width = `${pct}%`;
                document.getElementById('progressText').innerText = `${classified} / ${total} (${Math.round(pct)}%)`;
            }

            async function submitDirection(direction) {
                if (!currentFeature) return;

                const id = currentFeature.properties.id_infraestructura;
                try {
                    const res = await fetch('/api/classify', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ id_infraestructura: id, orientacion: direction })
                    });
                    const data = await res.json();
                    if (data.success) {
                        fetchNext();
                    } else {
                        alert("Error al guardar: " + data.error);
                    }
                } catch (e) {
                    console.error("Error al enviar clasificación", e);
                    alert("Error al enviar la respuesta.");
                }
            }

            fetchNext();
        </script>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route("/api/set_zone/<zone>")
def api_set_zone(zone):
    global ACTIVE_ZONE
    if zone in ZONES:
        ACTIVE_ZONE = zone
        return jsonify({"success": True, "active_zone": ACTIVE_ZONE})
    return jsonify({"success": False, "error": "Zona no válida"}), 400

@app.route("/api/next")
def api_next():
    try:
        data = load_geojson()
        features = data.get("features", [])
        
        affected = [
            f for f in features 
            if 'derrumbado' in f.get("properties", {}).get("label", "").lower() or
               'derrumado' in f.get("properties", {}).get("label", "").lower()
        ]
        
        total = len(affected)
        
        unclassified = []
        classified_count = 0
        for f in affected:
            props = f.get("properties", {})
            if "Orientacion del Derrumbe" in props and props["Orientacion del Derrumbe"]:
                classified_count += 1
            else:
                unclassified.append(f)
                
        if not unclassified:
            return jsonify({"finished": True, "total": total, "classified": total})
            
        selected = random.choice(unclassified)
        return jsonify({
            "finished": False,
            "total": total,
            "classified": classified_count,
            "feature": selected
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/crop/<id_infraestructura>/<tipo>")
def api_crop(id_infraestructura, tipo):
    try:
        data = load_geojson()
        selected_feature = None
        for f in data.get("features", []):
            if f.get("properties", {}).get("id_infraestructura") == id_infraestructura:
                selected_feature = f
                break
                
        if not selected_feature:
            return "Infraestructura no encontrada", 404
            
        props = selected_feature.get("properties", {})
        lon = props.get("lon_wgs84")
        lat = props.get("lat_wgs84")
        
        cache_path = os.path.join(CROPS_DIR, f"{id_infraestructura}_{tipo}.png")
        
        if os.path.exists(cache_path):
            return send_file(cache_path, mimetype="image/png")
            
        if tipo == "pre":
            if not HAS_PYPROJ:
                fallback = Image.new('RGB', (512, 512), (30, 34, 42))
                draw = ImageDraw.Draw(fallback)
                draw.text((60, 240), f"PRE {id_infraestructura} no pre-generada\n(Servidor sin pyproj)", fill=(180, 180, 180))
                img_io = io.BytesIO()
                fallback.save(img_io, 'PNG')
                img_io.seek(0)
                return send_file(img_io, mimetype='image/png')
            crop_img = crop_pre_event(lon, lat)
        elif tipo == "post":
            if not HAS_RASTERIO:
                fallback = Image.new('RGB', (512, 512), (30, 34, 42))
                draw = ImageDraw.Draw(fallback)
                draw.text((60, 240), f"POST {id_infraestructura} no pre-generada\n(Servidor sin rasterio)", fill=(180, 180, 180))
                img_io = io.BytesIO()
                fallback.save(img_io, 'PNG')
                img_io.seek(0)
                return send_file(img_io, mimetype='image/png')
            crop_img = crop_post_event(lon, lat)
        else:
            return "Tipo incorrecto", 400
            
        if crop_img is None:
            return "No se pudo realizar el recorte", 500
            
        crop_img.save(cache_path, format="PNG")
        
        img_io = io.BytesIO()
        crop_img.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error al generar recorte: {str(e)}", 500

@app.route("/api/classify", methods=["POST"])
def api_classify():
    try:
        req_data = request.get_json()
        id_infra = req_data.get("id_infraestructura")
        orientacion = req_data.get("orientacion")
        
        if not id_infra or not orientacion:
            return jsonify({"success": False, "error": "Parámetros faltantes"}), 400
            
        data = load_geojson()
        features = data.get("features", [])
        modified = False
        
        for f in features:
            if f.get("properties", {}).get("id_infraestructura") == id_infra:
                f["properties"]["Orientacion del Derrumbe"] = orientacion
                modified = True
                break
                
        if modified:
            save_geojson(data)
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "error": "ID de edificación no encontrado"}), 404
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/download")
def api_download():
    try:
        import geopandas as gpd
        import zipfile
        import tempfile
        
        geojson_path = ZONES[ACTIVE_ZONE]['geojson']
        gdf = gpd.read_file(geojson_path)
        
        # Filtrar solo derrumbados que tengan información de orientación de derrumbe válida
        gdf['label_lower'] = gdf['label'].str.lower().fillna('')
        
        # Filtro de orientación válida (no nula, no vacía y que no sea Omitido)
        if 'Orientacion del Derrumbe' in gdf.columns:
            gdf_affected = gdf[
                (gdf['label_lower'].str.contains('derrumbado') | gdf['label_lower'].str.contains('derrumado')) &
                gdf['Orientacion del Derrumbe'].notna() &
                (gdf['Orientacion del Derrumbe'] != '') &
                (gdf['Orientacion del Derrumbe'] != 'Omitido')
            ].copy()
        else:
            gdf_affected = gdf.iloc[0:0].copy() # DataFrame vacío si la columna no existe aún
        
        if 'label_lower' in gdf_affected.columns:
            gdf_affected = gdf_affected.drop(columns=['label_lower'])
            
        if gdf_affected.empty:
            return "No hay edificaciones afectadas para exportar", 400
            
        capa_nombre = f"orientacion_derrumbes_{ACTIVE_ZONE.lower()}"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            shp_path = os.path.join(tmpdir, f"{capa_nombre}.shp")
            gdf_affected.to_file(shp_path, driver='ESRI Shapefile')
            
            memory_file = io.BytesIO()
            with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
                for filename in os.listdir(tmpdir):
                    filepath = os.path.join(tmpdir, filename)
                    zf.write(filepath, filename)
            memory_file.seek(0)
            
            return send_file(
                memory_file,
                mimetype="application/zip",
                as_attachment=True,
                download_name=f"{capa_nombre}.zip"
            )
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
