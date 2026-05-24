import os
import cv2
import json
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import urllib.request

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import mediapipe as mp
import sys

def get_model_path(filename):
    path = os.path.join(_DIR, filename)
    if sys.platform == "win32":
        import ctypes
        buf = ctypes.create_unicode_buffer(32767)
        ctypes.windll.kernel32.GetShortPathNameW(os.path.dirname(path), buf, 32767)
        return os.path.join(buf.value, filename)
    return path

# ─── Configuración ───────────────────────────────────────────
LAR_THRESHOLD     = 0.03
MAX_FACES         = 7
PROCESS_W         = 960
PROCESS_H         = 540
REID_UMBRAL       = 0.65
MIN_BBOX_W        = 25
MIN_BBOX_H        = 25
MAX_ASPECT_RATIO  = 2.5
MIN_FRAMES_EXPORT = 5
NMS_IOU_UMBRAL    = 0.5   # subido de 0.2 — caras cercanas no se eliminan
MAX_DIST_TRACK    = 80    # px máximo de desplazamiento del centro entre frames

OUTPUT_VIDEO = "results/videos/output_annotated.mp4"
OUTPUT_JSON  = "data/json/lip_tracking_data.json"

_DIR       = os.path.dirname(os.path.abspath(__file__))
YUNET_PATH = get_model_path("yunet.onnx")
MODEL_PATH = get_model_path("face_landmarker.task")

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

COLORS = [
    (0,255,100),(0,180,255),(255,80,80),
    (200,0,255),(255,200,0),(200,0,200),(255,128,0)
]
LIP_TOP, LIP_BOTTOM, LIP_LEFT, LIP_RIGHT = 13, 14, 78, 308
LIPS_OUTER = [61,146,91,181,84,17,314,405,321,375,291,308,
              324,318,402,317,14,87,178,88,95,78,191,80,81,82]


def crear_dirs():
    os.makedirs("results/videos", exist_ok=True)
    os.makedirs("data/json",      exist_ok=True)


def descargar_modelos():
    for path, url, nombre in [
        (YUNET_PATH, YUNET_URL, "YuNet"),
        (MODEL_PATH, MODEL_URL, "FaceLandmarker"),
    ]:
        if not os.path.exists(path):
            print(f"Descargando {nombre}...")
            urllib.request.urlretrieve(url, path)
            print(f"  OK: {path}")
        else:
            print(f"  {nombre} OK: {os.path.abspath(path)}")


def calcular_lar(lm):
    top    = np.array([lm[LIP_TOP].x,    lm[LIP_TOP].y])
    bottom = np.array([lm[LIP_BOTTOM].x, lm[LIP_BOTTOM].y])
    left   = np.array([lm[LIP_LEFT].x,   lm[LIP_LEFT].y])
    right  = np.array([lm[LIP_RIGHT].x,  lm[LIP_RIGHT].y])
    return np.linalg.norm(top - bottom) / max(np.linalg.norm(left - right), 1e-6)


def similitud_coseno(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def iou(boxA, boxB):
    ax1, ay1 = boxA[0], boxA[1]
    ax2, ay2 = ax1 + boxA[2], ay1 + boxA[3]
    bx1, by1 = boxB[0], boxB[1]
    bx2, by2 = bx1 + boxB[2], by1 + boxB[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


def centro(bbox):
    x, y, w, h = bbox
    return x + w / 2, y + h / 2


def nms(caras, umbral_iou=NMS_IOU_UMBRAL):
    if len(caras) == 0:
        return caras
    caras = sorted(caras, key=lambda c: c[2] * c[3], reverse=True)
    resultado = []
    for cara in caras:
        duplicado = False
        for r in resultado:
            if iou(cara, r) > umbral_iou:
                duplicado = True
                break
        if not duplicado:
            resultado.append(cara)
    return resultado


def get_embedding(crop_bgr):
    try:
        from deepface import DeepFace
        resultado = DeepFace.represent(
            img_path=crop_bgr,
            model_name="ArcFace",
            enforce_detection=False,
            detector_backend="skip",
        )
        return np.array(resultado[0]["embedding"], dtype=np.float32)
    except Exception:
        return None


# ─── Clustering ──────────────────────────────────────────────

def clustering_kmeans(banco_embeddings, n_personas):
    from sklearn.cluster import KMeans
    pids = list(banco_embeddings.keys())
    if len(pids) == 0:
        return {}
    if len(pids) <= n_personas:
        return {pid: i for i, pid in enumerate(pids)}
    embs = np.stack([banco_embeddings[p] for p in pids])
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    embs_norm = embs / norms
    km = KMeans(n_clusters=n_personas, random_state=42, n_init=10)
    labels = km.fit_predict(embs_norm)
    remap = {pid: int(label) for pid, label in zip(pids, labels)}
    print(f"\nClustering K-Means: {len(pids)} IDs → {n_personas} personas")
    return remap


def clustering_dbscan(banco_embeddings, eps=0.7, min_samples=1):
    from sklearn.cluster import DBSCAN
    pids = list(banco_embeddings.keys())
    if len(pids) == 0:
        return {}
    embs = np.stack([banco_embeddings[p] for p in pids])
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    embs_norm = embs / norms
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = db.fit_predict(embs_norm)
    remap = {}
    next_outlier = max(labels) + 1
    for pid, label in zip(pids, labels):
        remap[pid] = next_outlier if label == -1 else label
        if label == -1:
            next_outlier += 1
    valores_unicos = sorted(set(remap.values()))
    renumber = {v: i for i, v in enumerate(valores_unicos)}
    remap = {pid: renumber[v] for pid, v in remap.items()}
    n_personas = len(set(remap.values()))
    print(f"\nClustering DBSCAN: {len(pids)} IDs → {n_personas} personas")
    return remap


def remap_historiales(hist_lar, hist_t, hist_bbox, remap):
    new_lar  = defaultdict(list)
    new_t    = defaultdict(list)
    new_bbox = defaultdict(list)
    for pid_tmp, pid_final in remap.items():
        if pid_tmp in hist_lar:
            combined = sorted(
                zip(hist_t[pid_tmp], hist_lar[pid_tmp], hist_bbox[pid_tmp]),
                key=lambda x: x[0]
            )
            for t, lar, bbox in combined:
                new_t[pid_final].append(t)
                new_lar[pid_final].append(lar)
                new_bbox[pid_final].append(bbox)
    for pid in new_t:
        orden = np.argsort(new_t[pid])
        new_t[pid]    = [new_t[pid][i]    for i in orden]
        new_lar[pid]  = [new_lar[pid][i]  for i in orden]
        new_bbox[pid] = [new_bbox[pid][i] for i in orden]
    return new_lar, new_t, new_bbox


# ─── Pipeline principal ──────────────────────────────────────

def procesar_video(video_path):
    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sx, sy = orig_w / PROCESS_W, orig_h / PROCESS_H

    print(f"{orig_w}x{orig_h} | {fps:.1f} fps | {total} frames")
    print(f"yunet.onnx     : {os.path.abspath(YUNET_PATH)}  existe={os.path.exists(YUNET_PATH)}")
    print(f"face_landmarker: {os.path.abspath(MODEL_PATH)}  existe={os.path.exists(MODEL_PATH)}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (orig_w, orig_h))

    historial_lar  = defaultdict(list)
    historial_t    = defaultdict(list)
    historial_bbox = defaultdict(list)
    banco_emb      = {}
    conteos_emb    = {}

    yunet = cv2.FaceDetectorYN.create(
        YUNET_PATH, "", (PROCESS_W, PROCESS_H),
        score_threshold=0.4, top_k=MAX_FACES
    )

    lm_options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        num_faces=2,
        min_face_detection_confidence=0.1,
        min_face_presence_confidence=0.1,
        min_tracking_confidence=0.1,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(lm_options)

    caras_prev = {}
    next_pid   = 0
    frames_con_caras     = 0
    frames_con_landmarks = 0
    frames_sin_landmarks = 0

    for i in tqdm(range(total), desc="Procesando"):
        ret, frame = cap.read()
        if not ret:
            break

        t     = i / fps
        small = cv2.resize(frame, (PROCESS_W, PROCESS_H))

        # ── 1. Detectar + NMS ────────────────────────────────
        _, faces_raw = yunet.detect(small)
        caras_curr = []
        if faces_raw is not None:
            for f in faces_raw:
                x, y, w, h = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                aspect = w / max(h, 1)
                if w >= MIN_BBOX_W and h >= MIN_BBOX_H and aspect <= MAX_ASPECT_RATIO:
                    caras_curr.append([x, y, w, h])
        caras_curr = nms(caras_curr)

        if caras_curr:
            frames_con_caras += 1

        # ── 2. Tracking por IoU + distancia de centro ────────
        asignados  = {}
        usados_pid = set()
        usados_det = set()

        pares = []
        for pid, bbox_prev in caras_prev.items():
            cx_prev, cy_prev = centro(bbox_prev)
            for j, bbox_curr in enumerate(caras_curr):
                cx_curr, cy_curr = centro(bbox_curr)
                dist = np.sqrt((cx_prev - cx_curr)**2 + (cy_prev - cy_curr)**2)
                score_iou = iou(bbox_prev, bbox_curr)
                # Match si IoU > 0.1 O si el centro se movió menos de MAX_DIST_TRACK px
                if score_iou > 0.1 or dist < MAX_DIST_TRACK:
                    score_combinado = score_iou + max(0, 1 - dist / MAX_DIST_TRACK)
                    pares.append((score_combinado, pid, j))
        pares.sort(reverse=True)

        for score, pid, j in pares:
            if pid not in usados_pid and j not in usados_det:
                asignados[pid] = j
                usados_pid.add(pid)
                usados_det.add(j)

        # ── 3. Re-identificar sin match ──────────────────────
        for j in range(len(caras_curr)):
            if j in usados_det:
                continue
            x, y, w, h = caras_curr[j]
            margen = int(min(w, h) * 0.1)
            x1c = max(0, x-margen);           y1c = max(0, y-margen)
            x2c = min(PROCESS_W, x+w+margen); y2c = min(PROCESS_H, y+h+margen)
            crop = small[y1c:y2c, x1c:x2c]
            if crop.size == 0:
                continue

            emb = get_embedding(crop)

            if emb is not None and len(banco_emb) > 0:
                mejor_pid, mejor_sim = None, -1.0
                for pid_b, emb_b in banco_emb.items():
                    sim = similitud_coseno(emb, emb_b)
                    if sim > mejor_sim:
                        mejor_sim = sim
                        mejor_pid = pid_b
                if mejor_sim >= REID_UMBRAL:
                    pid = mejor_pid
                    n = conteos_emb[pid]
                    banco_emb[pid] = (banco_emb[pid] * n + emb) / (n + 1)
                    conteos_emb[pid] = n + 1
                else:
                    pid = next_pid; next_pid += 1
                    banco_emb[pid] = emb; conteos_emb[pid] = 1
            else:
                pid = next_pid; next_pid += 1
                if emb is not None:
                    banco_emb[pid] = emb; conteos_emb[pid] = 1

            asignados[pid] = j
            usados_pid.add(pid)

        caras_prev = {pid: caras_curr[idx]
                      for pid, idx in asignados.items()
                      if idx < len(caras_curr)}

        # ── 4. LAR + dibujo ──────────────────────────────────
        for pid, idx in asignados.items():
            if idx >= len(caras_curr):
                continue
            x, y, w, h = caras_curr[idx]

            margen = int(min(w, h) * 0.3)
            x1c = max(0, x-margen);           y1c = max(0, y-margen)
            x2c = min(PROCESS_W, x+w+margen); y2c = min(PROCESS_H, y+h+margen)
            crop_small = small[y1c:y2c, x1c:x2c]
            if crop_small.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop_small, cv2.COLOR_BGR2RGB)
            mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
            res      = landmarker.detect(mp_img)

            lar = 0.0
            if res.face_landmarks:
                frames_con_landmarks += 1
                lm  = res.face_landmarks[0]
                lar = calcular_lar(lm)
                ch, cw = y2c - y1c, x2c - x1c
                for li in LIPS_OUTER:
                    lx = int(lm[li].x * cw * sx) + int(x1c * sx)
                    ly = int(lm[li].y * ch * sy) + int(y1c * sy)
                    cv2.circle(frame, (lx, ly), 2, COLORS[pid % len(COLORS)], -1)
            else:
                frames_sin_landmarks += 1

            hablando = lar > LAR_THRESHOLD
            color    = COLORS[pid % len(COLORS)]
            ox1 = int(x * sx);   oy1 = int(y * sy)
            ox2 = int((x+w)*sx); oy2 = int((y+h)*sy)

            historial_lar[pid].append(lar)
            historial_t[pid].append(round(t, 4))
            historial_bbox[pid].append([ox1, oy1, ox2, oy2])

            grosor = 3 if hablando else 1
            cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), color, grosor)
            label = f"P{pid} {'HABLA' if hablando else '...'} {lar:.3f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            ly_label = max(oy1 - 8, 18)
            cv2.rectangle(frame, (ox1, ly_label-th-4), (ox1+tw+4, ly_label+4), (0,0,0), -1)
            cv2.putText(frame, label, (ox1+2, ly_label),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            barra_max = 80
            barra_val = min(int(lar / 0.12 * barra_max), barra_max)
            cv2.rectangle(frame, (ox1, oy2+4), (ox1+barra_max, oy2+14), (40,40,40), -1)
            cv2.rectangle(frame, (ox1, oy2+4), (ox1+barra_val, oy2+14), color, -1)

        n_habla = sum(1 for pid in asignados
                      if historial_lar[pid] and historial_lar[pid][-1] > LAR_THRESHOLD)
        cv2.putText(frame, f"t={t:.1f}s | caras={len(caras_curr)} | hablan={n_habla}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220,220,220), 2)
        writer.write(frame)

    cap.release()
    writer.release()
    landmarker.close()

    print(f"\n[DIAGNÓSTICO]")
    print(f"  Frames totales        : {total}")
    print(f"  Frames con caras      : {frames_con_caras}")
    print(f"  Frames con landmarks  : {frames_con_landmarks}")
    print(f"  Frames sin landmarks  : {frames_sin_landmarks}")
    print(f"  IDs acumulados        : {len(historial_lar)}")
    top10 = sorted(historial_lar.keys(), key=lambda p: len(historial_lar[p]), reverse=True)[:10]
    for pid in top10:
        n = len(historial_lar[pid])
        lares = historial_lar[pid]
        print(f"  pid={pid}: {n} frames, lar_max={max(lares):.4f}, lar_mean={sum(lares)/n:.4f}")

    return historial_lar, historial_t, historial_bbox, fps, total, banco_emb


# ─── Exportar ────────────────────────────────────────────────

def exportar(hist_lar, hist_t, hist_bbox, fps, total, video, banco_emb, n_personas=None):
    pids_validos   = {pid for pid in hist_lar if len(hist_lar[pid]) >= MIN_FRAMES_EXPORT}
    banco_filtrado = {pid: emb for pid, emb in banco_emb.items() if pid in pids_validos}

    print(f"\n[EXPORTAR] pids válidos: {len(pids_validos)}, con embedding: {len(banco_filtrado)}")

    if len(banco_filtrado) == 0:
        n = n_personas if n_personas is not None else 2
        print(f"[EXPORTAR] Sin embeddings — agrupando por posición X en {n} personas")

        centros_x = {}
        for pid in pids_validos:
            bboxes = hist_bbox[pid]
            if bboxes:
                cx = np.mean([(b[0] + b[2]) / 2 for b in bboxes])
                centros_x[pid] = cx

        from sklearn.cluster import KMeans
        pids_list = list(centros_x.keys())
        X = np.array([[centros_x[p]] for p in pids_list])
        km = KMeans(n_clusters=n, random_state=42, n_init=10)
        labels = km.fit_predict(X)

        centros_cluster = {i: np.mean(X[labels == i]) for i in range(n)}
        orden = sorted(centros_cluster.keys(), key=lambda i: centros_cluster[i])
        cluster_a_pid_final = {cluster: pid_final for pid_final, cluster in enumerate(orden)}

        remap_valido = {pid: cluster_a_pid_final[labels[i]]
                        for i, pid in enumerate(pids_list)}

        print(f"[EXPORTAR] Resultado:")
        for pid_final in range(n):
            pids_en_cluster = [p for p, pf in remap_valido.items() if pf == pid_final]
            frames_total = sum(len(hist_lar[p]) for p in pids_en_cluster)
            cx_avg = np.mean([centros_x[p] for p in pids_en_cluster])
            print(f"  Persona {pid_final}: {len(pids_en_cluster)} PIDs, "
                  f"{frames_total} frames, centro_x={cx_avg:.0f}px")
    else:
        if n_personas is not None:
            remap = clustering_kmeans(banco_filtrado, n_personas)
        else:
            remap = clustering_dbscan(banco_filtrado, eps=0.7, min_samples=1)
        remap_valido = {pid: remap[pid] for pid in pids_validos if pid in remap}

    hist_lar_f, hist_t_f, hist_bbox_f = remap_historiales(
        hist_lar, hist_t, hist_bbox, remap_valido
    )

    data = {
        "video":        video,
        "fps":          fps,
        "duracion_seg": total / fps,
        "n_personas":   len(hist_lar_f),
        "lar_series": {
            str(pid): {
                "tiempos": hist_t_f[pid],
                "valores": hist_lar_f[pid],
                "bboxes":  hist_bbox_f[pid]
            }
            for pid in sorted(hist_lar_f.keys())
        }
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f, indent=2)
    print(f"JSON guardado: {OUTPUT_JSON}  ({data['n_personas']} personas finales)")
    return hist_lar_f


def main():
    global LAR_THRESHOLD
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",      required=True)
    parser.add_argument("--umbral",     type=float, default=LAR_THRESHOLD)
    parser.add_argument("--n_personas", type=int,   default=None)
    args = parser.parse_args()
    LAR_THRESHOLD = args.umbral

    crear_dirs()
    descargar_modelos()

    lar, t, bboxes, fps, total, banco_emb = procesar_video(args.video)
    lar_final = exportar(lar, t, bboxes, fps, total, args.video, banco_emb, args.n_personas)

    print("\n=== RESUMEN FINAL ===")
    if not lar_final:
        print("No se detectaron caras.")
        return
    for pid in sorted(lar_final.keys()):
        vals     = lar_final[pid]
        hablando = sum(1 for v in vals if v > LAR_THRESHOLD)
        pct      = 100 * hablando / len(vals) if vals else 0
        print(f"Persona {pid}: {len(vals)} frames | hablando {hablando} ({pct:.1f}%)")


if __name__ == "__main__":
    main()