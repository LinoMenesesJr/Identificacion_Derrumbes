import os
import json
import urllib.request
import io
import time
import math
from PIL import Image, ImageDraw
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer

GEOJSON_PATH = r"c:\Users\MR\Documents\Deteccion de orientacion de derrumbe\TANAGUARENA_revisado.geojson"
TIFF_TANAGUARENA = r"c:\Users\MR\Documents\Deteccion de orientacion de derrumbe\ORTOMOSAICO_TANAGUARENA.tif"
TIFF_MACUTO = r"c:\Users\MR\Documents\Deteccion de orientacion de derrumbe\ORTOMOSAICO_MACUTO03JUL26.tif"
TIFF_CARABALLEDA = r"c:\Users\MR\Documents\Deteccion de orientacion de derrumbe\MOSAICO_CARABALLEDA.tif"
CROPS_DIR = r"c:\Users\MR\Documents\Deteccion de orientacion de derrumbe\crops"

os.makedirs(CROPS_DIR, exist_ok=True)

to_mercator = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
to_geographic = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

def latlon_to_tile(lon, lat, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return xtile, ytile

def tile_bounds_3857(x, y, zoom):
    initial_resolution = 2 * math.pi * 6378137 / 256
    origin_shift = 2 * math.pi * 6378137 / 2.0
    res = initial_resolution / (2**zoom)
    tile_size_m = 256 * res
    minx = x * tile_size_m - origin_shift
    maxy = origin_shift - y * tile_size_m
    maxx = (x + 1) * tile_size_m - origin_shift
    miny = origin_shift - (y + 1) * tile_size_m
    return minx, miny, maxx, maxy

def agregar_grilla_y_norte(imagen, etiqueta=""):
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
    x_centro, y_centro = to_mercator.transform(lon, lat)
    ulx_3857 = x_centro - semilado_m
    uly_3857 = y_centro + semilado_m
    lrx_3857 = x_centro + semilado_m
    lry_3857 = y_centro - semilado_m
    
    zoom = 19
    xtile, ytile = latlon_to_tile(lon, lat, zoom)
    
    imgs = []
    for dy in [-1, 0, 1]:
        row = []
        for dx in [-1, 0, 1]:
            url = f"https://mt1.google.com/vt/lyrs=s&x={xtile+dx}&y={ytile+dy}&z={zoom}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            req = urllib.request.Request(url, headers=headers)
            img = None
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(req, timeout=4) as r:
                        img = Image.open(io.BytesIO(r.read()))
                        break
                except Exception as e:
                    time.sleep(0.5)
            if img is None:
                img = Image.new('RGB', (256, 256), (30, 34, 42))
                draw = ImageDraw.Draw(img)
                draw.text((10, 120), "Error de Red/Límite Google", fill=(180, 180, 180))
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
    if (-66.8420 <= lon <= -66.8092) and (10.6071 <= lat <= 10.6198):
        tif_path = TIFF_TANAGUARENA
    elif (-66.8727 <= lon <= -66.8327) and (10.6059 <= lat <= 10.6233):
        tif_path = TIFF_CARABALLEDA
    elif (-66.9104 <= lon <= -66.8663) and (10.5957 <= lat <= 10.6138):
        tif_path = TIFF_MACUTO
    else:
        tif_path = TIFF_TANAGUARENA

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

def main():
    print("Cargando GeoJSON...")
    with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    features = data.get("features", [])
    affected = [
        f for f in features 
        if 'derrumbado' in f.get("properties", {}).get("label", "").lower() or
           'derrumado' in f.get("properties", {}).get("label", "").lower()
    ]
    
    total = len(affected)
    print(f"Total de edificios afectados a procesar: {total}")
    
    for i, f in enumerate(affected):
        props = f.get("properties", {})
        id_infra = props.get("id_infraestructura")
        lon = props.get("lon_wgs84")
        lat = props.get("lat_wgs84")
        
        print(f"[{i+1}/{total}] Generando recortes para {id_infra}...")
        
        # Generar PRE
        pre_path = os.path.join(CROPS_DIR, f"{id_infra}_pre.png")
        if not os.path.exists(pre_path):
            try:
                pre_img = crop_pre_event(lon, lat)
                pre_img.save(pre_path, format="PNG")
            except Exception as e:
                print(f"  Error en PRE {id_infra}: {e}")
        
        # Generar POST
        post_path = os.path.join(CROPS_DIR, f"{id_infra}_post.png")
        if not os.path.exists(post_path):
            try:
                post_img = crop_post_event(lon, lat)
                post_img.save(post_path, format="PNG")
            except Exception as e:
                print(f"  Error en POST {id_infra}: {e}")
                
    print("¡Pre-generación completada con éxito!")

if __name__ == "__main__":
    main()
