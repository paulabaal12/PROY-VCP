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
MAX_FACES = 6

OUTPUT_VIDEO = "results/videos/output_annotated.mp4"
OUTPUT_JSON  = "data/json/lip_tracking_data.json"

MODEL_PATH = "face_landmarker.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

COLORS = [(0,255,100),(0,180,255),(255,80,80),(200,0,255),(255,200,0),(200,0,200)]

LIP_TOP, LIP_BOTTOM, LIP_LEFT, LIP_RIGHT = 13, 14, 78, 308


def crear_dirs():
    os.makedirs("results/videos", exist_ok=True)
    os.makedirs("data/json", exist_ok=True)


def descargar_modelo():
    if os.path.exists(MODEL_PATH):
        return
    print("Descargando modelo...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def calcular_lar(lm):
    top    = np.array([lm[LIP_TOP].x,    lm[LIP_TOP].y])
    bottom = np.array([lm[LIP_BOTTOM].x, lm[LIP_BOTTOM].y])
    left   = np.array([lm[LIP_LEFT].x,   lm[LIP_LEFT].y])
    right  = np.array([lm[LIP_RIGHT].x,  lm[LIP_RIGHT].y])
    return np.linalg.norm(top - bottom) / max(np.linalg.norm(left - right), 1e-6)


def bbox(lm, w, h):
    xs = [p.x for p in lm]
    ys = [p.y for p in lm]
    return int(min(xs)*w), int(min(ys)*h), int(max(xs)*w), int(max(ys)*h)


def procesar_video(video_path):
    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"{w}x{h} | {fps:.1f} fps | {total} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))

    historial_lar  = defaultdict(list)
    historial_t    = defaultdict(list)
    historial_bbox = defaultdict(list)  # NUEVO: guardamos coordenadas

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

            t      = i / fps
            mp_img = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )
            res = detector.detect_for_video(mp_img, int(t * 1000))

            if res.face_landmarks:
                for pid, lm in enumerate(res.face_landmarks):
                    lar      = calcular_lar(lm)
                    hablando = lar > LAR_THRESHOLD
                    color    = COLORS[pid % len(COLORS)]
                    x1, y1, x2, y2 = bbox(lm, w, h)

                    historial_lar[pid].append(lar)
                    historial_t[pid].append(t)
                    historial_bbox[pid].append([x1, y1, x2, y2])  # NUEVO

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame,
                        f"P{pid} {'HABLA' if hablando else '...'} {lar:.3f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            writer.write(frame)

    cap.release()
    writer.release()

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
                "bboxes": hist_bbox[pid]  # NUEVO: [x1,y1,x2,y2] por frame
            } for pid in hist_lar
        }
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f, indent=2)

    print(f"JSON guardado: {OUTPUT_JSON}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    args = parser.parse_args()

    crear_dirs()
    descargar_modelo()

    lar, t, bboxes, fps, total = procesar_video(args.video)
    exportar(lar, t, bboxes, fps, total, args.video)

    print("\nListo")


if __name__ == "__main__":
    main()