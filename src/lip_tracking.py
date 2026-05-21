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

LAR_THRESHOLD = 0.03
MAX_FACES     = 7
PROCESS_W     = 960
PROCESS_H     = 540

OUTPUT_VIDEO = "results/videos/output_annotated.mp4"
OUTPUT_JSON  = "data/json/lip_tracking_data.json"

YUNET_PATH   = "yunet.onnx"
YUNET_URL    = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
MODEL_PATH   = "face_landmarker.task"
MODEL_URL    = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

COLORS = [(0,255,100),(0,180,255),(255,80,80),(200,0,255),(255,200,0),(200,0,200),(255,128,0)]

LIP_TOP, LIP_BOTTOM, LIP_LEFT, LIP_RIGHT = 13, 14, 78, 308
LIPS_OUTER = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95,78,191,80,81,82]


def crear_dirs():
    os.makedirs("results/videos", exist_ok=True)
    os.makedirs("data/json", exist_ok=True)


def descargar_modelos():
    for path, url, nombre in [(YUNET_PATH, YUNET_URL, "YuNet"), (MODEL_PATH, MODEL_URL, "FaceLandmarker")]:
        if not os.path.exists(path):
            print(f"Descargando {nombre}...")
            urllib.request.urlretrieve(url, path)
            print(f"  OK: {path}")


def calcular_lar(lm):
    top    = np.array([lm[LIP_TOP].x,    lm[LIP_TOP].y])
    bottom = np.array([lm[LIP_BOTTOM].x, lm[LIP_BOTTOM].y])
    left   = np.array([lm[LIP_LEFT].x,   lm[LIP_LEFT].y])
    right  = np.array([lm[LIP_RIGHT].x,  lm[LIP_RIGHT].y])
    return np.linalg.norm(top - bottom) / max(np.linalg.norm(left - right), 1e-6)


def iou(boxA, boxB):
    """Intersection over Union entre dos boxes [x,y,w,h]."""
    ax1, ay1 = boxA[0], boxA[1]
    ax2, ay2 = ax1 + boxA[2], ay1 + boxA[3]
    bx1, by1 = boxB[0], boxB[1]
    bx2, by2 = bx1 + boxB[2], by1 + boxB[3]
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0


def asignar_ids(caras_prev, caras_curr, umbral_iou=0.3):
    """
    Asigna IDs estables entre frames usando IoU de bounding boxes.
    caras_prev: dict {pid: [x,y,w,h]}
    caras_curr: lista de [x,y,w,h]
    Retorna dict {pid: idx_en_curr}
    """
    if not caras_prev:
        return {i: i for i in range(len(caras_curr))}

    asignados  = {}
    usados_pid = set()
    usados_det = set()

    # Ordenar por IoU descendente para asignar primero los mejores matches
    pares = []
    for pid, bbox_prev in caras_prev.items():
        for j, bbox_curr in enumerate(caras_curr):
            score = iou(bbox_prev, bbox_curr)
            if score > umbral_iou:
                pares.append((score, pid, j))
    pares.sort(reverse=True)

    for score, pid, j in pares:
        if pid not in usados_pid and j not in usados_det:
            asignados[pid] = j
            usados_pid.add(pid)
            usados_det.add(j)

    # IDs nuevos para detecciones sin match
    next_pid = max(caras_prev.keys()) + 1 if caras_prev else 0
    for j in range(len(caras_curr)):
        if j not in usados_det:
            asignados[next_pid] = j
            next_pid += 1

    return asignados


def procesar_video(video_path):
    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    sx = orig_w / PROCESS_W
    sy = orig_h / PROCESS_H

    print(f"{orig_w}x{orig_h} | {fps:.1f} fps | {total} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (orig_w, orig_h))

    historial_lar  = defaultdict(list)
    historial_t    = defaultdict(list)
    historial_bbox = defaultdict(list)

    # ── Detectores ──
    yunet = cv2.FaceDetectorYN.create(YUNET_PATH, "", (PROCESS_W, PROCESS_H),
                                       score_threshold=0.5, top_k=MAX_FACES)

    lm_options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        num_faces=1,
        min_face_detection_confidence=0.1,
        min_face_presence_confidence=0.1,
        min_tracking_confidence=0.1,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(lm_options)

    caras_prev = {}   # {pid: [x,y,w,h]} en coords PROCESS

    for i in tqdm(range(total), desc="Procesando"):
        ret, frame = cap.read()
        if not ret:
            break

        t     = i / fps
        small = cv2.resize(frame, (PROCESS_W, PROCESS_H))

        # ── 1. Detectar caras con YuNet ──
        _, faces_raw = yunet.detect(small)
        caras_curr = []   # [x,y,w,h] en PROCESS coords
        if faces_raw is not None:
            for f in faces_raw:
                x,y,w,h = int(f[0]),int(f[1]),int(f[2]),int(f[3])
                # Filtrar detecciones muy pequeñas o fuera de frame
                if w > 20 and h > 20:
                    caras_curr.append([x, y, w, h])

        # ── 2. Asignar IDs estables ──
        id_map = asignar_ids(caras_prev, caras_curr)  # {pid: idx_en_curr}
        caras_prev = {pid: caras_curr[idx] for pid, idx in id_map.items() if idx < len(caras_curr)}

        # ── 3. Por cada cara: recortar y calcular LAR con MediaPipe ──
        for pid, idx in id_map.items():
            if idx >= len(caras_curr):
                continue
            x, y, w, h = caras_curr[idx]

            # Recorte con margen para MediaPipe
            margen = int(min(w, h) * 0.15)
            x1c = max(0, x - margen)
            y1c = max(0, y - margen)
            x2c = min(PROCESS_W, x + w + margen)
            y2c = min(PROCESS_H, y + h + margen)

            crop_small = small[y1c:y2c, x1c:x2c]
            if crop_small.size == 0:
                continue

            crop_rgb  = cv2.cvtColor(crop_small, cv2.COLOR_BGR2RGB)
            mp_img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
            res       = landmarker.detect(mp_img)

            lar = 0.0
            if res.face_landmarks:
                lm  = res.face_landmarks[0]
                lar = calcular_lar(lm)

                # Dibujar puntos de labios en frame original
                ch = y2c - y1c
                cw = x2c - x1c
                for li in LIPS_OUTER:
                    lx = int(lm[li].x * cw * sx) + int(x1c * sx)
                    ly = int(lm[li].y * ch * sy) + int(y1c * sy)
                    cv2.circle(frame, (lx, ly), 2, COLORS[pid % len(COLORS)], -1)

            hablando = lar > LAR_THRESHOLD
            color    = COLORS[pid % len(COLORS)]

            # Coords en frame original
            ox1 = int(x * sx);      oy1 = int(y * sy)
            ox2 = int((x+w) * sx);  oy2 = int((y+h) * sy)

            historial_lar[pid].append(lar)
            historial_t[pid].append(round(t, 4))
            historial_bbox[pid].append([ox1, oy1, ox2, oy2])

            # Dibujar bbox
            grosor = 3 if hablando else 1
            cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), color, grosor)

            # Etiqueta con fondo
            label = f"P{pid} {'HABLA' if hablando else '...'} {lar:.3f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            ly_label = max(oy1 - 8, 18)
            cv2.rectangle(frame, (ox1, ly_label - th - 4), (ox1 + tw + 4, ly_label + 4),
                          (0,0,0), -1)
            cv2.putText(frame, label, (ox1 + 2, ly_label),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # Barra LAR
            barra_max = 80
            barra_val = min(int(lar / 0.12 * barra_max), barra_max)
            cv2.rectangle(frame, (ox1, oy2+4), (ox1+barra_max, oy2+14), (40,40,40), -1)
            cv2.rectangle(frame, (ox1, oy2+4), (ox1+barra_val, oy2+14), color, -1)

        # HUD
        n_habla = sum(1 for pid in id_map
                      if historial_lar[pid] and historial_lar[pid][-1] > LAR_THRESHOLD)
        cv2.putText(frame, f"t={t:.1f}s | caras={len(caras_curr)} | hablan={n_habla}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220,220,220), 2)

        writer.write(frame)

    cap.release()
    writer.release()
    landmarker.close()

    return historial_lar, historial_t, historial_bbox, fps, total


def exportar(hist_lar, hist_t, hist_bbox, fps, total, video):
    data = {
        "video": video,
        "fps": fps,
        "duracion_seg": total / fps,
        "n_personas": len(hist_lar),
        "lar_series": {
            str(pid): {
                "tiempos": hist_t[pid],
                "valores": hist_lar[pid],
                "bboxes":  hist_bbox[pid]
            } for pid in hist_lar
        }
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f, indent=2)
    print(f"JSON guardado: {OUTPUT_JSON}")


def main():
    global LAR_THRESHOLD
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--umbral", type=float, default=LAR_THRESHOLD)
    args = parser.parse_args()

    LAR_THRESHOLD = args.umbral

    crear_dirs()
    descargar_modelos()

    lar, t, bboxes, fps, total = procesar_video(args.video)
    exportar(lar, t, bboxes, fps, total, args.video)

    print("\n=== RESUMEN ===")
    if not lar:
        print("No se detectaron caras.")
        return
    for pid in sorted(lar.keys()):
        vals     = lar[pid]
        hablando = sum(1 for v in vals if v > LAR_THRESHOLD)
        pct      = 100 * hablando / len(vals) if vals else 0
        print(f"Persona {pid}: {len(vals)} frames | hablando {hablando} ({pct:.1f}%)")


if __name__ == "__main__":
    main()