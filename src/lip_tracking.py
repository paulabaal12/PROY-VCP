import os
import cv2
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import urllib.request

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import mediapipe as mp

LAR_THRESHOLD = 0.03
MAX_FACES = 4

OUTPUT_VIDEO = "results/videos/output_annotated2.mp4"
OUTPUT_JSON  = "data/json/lip_tracking_data2.json"

MODEL_PATH = "face_landmarker.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

COLORS = [(0,255,100),(0,180,255),(255,80,80),(200,0,255)]

LIP_TOP, LIP_BOTTOM, LIP_LEFT, LIP_RIGHT = 13,14,78,308

SMOOTHING_WINDOW   = 5


class FaceTracker:
    def __init__(self, max_distancia=0.2, max_frames_ausente=9999, n_personas=None):
        self.personas           = {}   # pid -> {"centro": (cx,cy), "frames_ausente": int}
        self.siguiente_id       = 0
        self.max_distancia      = max_distancia
        self.max_frames_ausente = max_frames_ausente
        self.n_personas         = n_personas  # None = sin límite

    def actualizar(self, centros):
        asignados = {}
        usados    = set()

        tope = self.n_personas is not None and self.siguiente_id >= self.n_personas

        for i, centro in enumerate(centros):
            mejor_pid  = None
            mejor_dist = float("inf")
            for pid, info in self.personas.items():
                if pid in usados:
                    continue
                dist = np.linalg.norm(np.array(centro) - np.array(info["centro"]))
                if dist < mejor_dist and (tope or dist < self.max_distancia):
                    mejor_dist = dist
                    mejor_pid  = pid
            if mejor_pid is not None:
                asignados[i] = mejor_pid
                usados.add(mejor_pid)
            else:
                asignados[i] = self.siguiente_id
                self.siguiente_id += 1

        pids_vistos = set(asignados.values())

        for i, centro in enumerate(centros):
            pid = asignados[i]
            self.personas[pid] = {"centro": centro, "frames_ausente": 0}

        for pid in list(self.personas.keys()):
            if pid not in pids_vistos:
                self.personas[pid]["frames_ausente"] += 1
                if self.personas[pid]["frames_ausente"] > self.max_frames_ausente:
                    del self.personas[pid]

        return [asignados[i] for i in range(len(centros))]


def crear_dirs():
    os.makedirs("results/videos", exist_ok=True)
    os.makedirs("data/json", exist_ok=True)


def descargar_modelo():
    if os.path.exists(MODEL_PATH):
        return
    print("Descargando modelo...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def calcular_lar(lm):
    top    = np.array([lm[LIP_TOP].x, lm[LIP_TOP].y])
    bottom = np.array([lm[LIP_BOTTOM].x, lm[LIP_BOTTOM].y])
    left   = np.array([lm[LIP_LEFT].x, lm[LIP_LEFT].y])
    right  = np.array([lm[LIP_RIGHT].x, lm[LIP_RIGHT].y])

    return np.linalg.norm(top-bottom) / max(np.linalg.norm(left-right), 1e-6)


def bbox(lm, w, h):
    xs = [p.x for p in lm]
    ys = [p.y for p in lm]
    return int(min(xs)*w), int(min(ys)*h), int(max(xs)*w), int(max(ys)*h)

def procesar_video(video_path, n_personas=None):

    cap = cv2.VideoCapture(video_path)

    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"{w}x{h} | {fps:.1f} fps | {total} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w,h))

    historial_lar = defaultdict(list)
    historial_t   = defaultdict(list)

    tracker = FaceTracker(max_distancia=0.2, max_frames_ausente=total, n_personas=n_personas)

    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        num_faces=MAX_FACES,
        running_mode=mp_vision.RunningMode.VIDEO,
    )

    with mp_vision.FaceLandmarker.create_from_options(options) as detector:

        for i in tqdm(range(total)):

            ret, frame = cap.read()
            if not ret:
                break

            t = i / fps
            mp_img = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

            res = detector.detect_for_video(mp_img, int(t*1000))

            if res.face_landmarks:

                centros = []
                for lm in res.face_landmarks:
                    centros.append((lm[1].x, lm[1].y))  # nariz: coordenadas normalizadas [0,1]

                pids = tracker.actualizar(centros)

                for pid, lm in zip(pids, res.face_landmarks):

                    lar = calcular_lar(lm)

                    historial_lar[pid].append(lar)
                    historial_t[pid].append(t)

                    hablando = lar > LAR_THRESHOLD
                    color = COLORS[pid % len(COLORS)]

                    x1,y1,x2,y2 = bbox(lm, w, h)

                    cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
                    cv2.putText(frame,
                        f"P{pid} {'HABLA' if hablando else '...'} {lar:.3f}",
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

            writer.write(frame)

    cap.release()
    writer.release()

    return historial_lar, historial_t, fps, total

def exportar(hist_lar, hist_t, fps, total, video):

    data = {
        "video": video,
        "fps": fps,
        "duracion_seg": total/fps,
        "n_personas": len(hist_lar),
        "lar_series": {
            str(pid): {
                "tiempos": hist_t[pid],
                "valores": hist_lar[pid]
            } for pid in hist_lar
        }
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f, indent=2)

    print(f"JSON guardado: {OUTPUT_JSON}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--n_personas", type=int, default=None,
                        help="Número esperado de personas (ej. 2 para un podcast de 2)")
    args = parser.parse_args()

    crear_dirs()
    descargar_modelo()

    lar, t, fps, total = procesar_video(args.video, n_personas=args.n_personas)

    exportar(lar, t, fps, total, args.video)

    print("\nListo")

if __name__ == "__main__":
    main()