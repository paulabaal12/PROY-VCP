"""
    python modulo1_lip_tracking.py --video video.mp4.mp4
    python modulo1_lip_tracking.py --video video.mp4.mp4 --umbral 0.025
    python modulo1_lip_tracking.py --video video.mp4.mp4 --diagnostico
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import os
import urllib.request
from collections import defaultdict
from tqdm import tqdm

LAR_THRESHOLD    = 0.03
SMOOTHING_WINDOW = 5
MAX_FACES        = 4
OUTPUT_VIDEO     = "output_annotated.mp4"
OUTPUT_PLOT      = "senal_lar.png"
OUTPUT_JSON      = "lip_tracking_data.json"
MODEL_PATH       = "face_landmarker.task"
MODEL_URL        = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

COLORS = [
    (0, 255, 100),
    (0, 180, 255),
    (255, 80,  80),
    (200, 0,  255),
]

LIPS_OUTER = [
    61, 146, 91, 181, 84, 17, 314, 405,
    321, 375, 291, 308, 324, 318, 402,
    317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82
]

LIP_TOP    = 13
LIP_BOTTOM = 14
LIP_LEFT   = 78
LIP_RIGHT  = 308

#  DESCARGAR MODELO (solo la primera vez, ~6 MB)

def descargar_modelo():
    if os.path.exists(MODEL_PATH):
        print(f"Modelo encontrado: {MODEL_PATH}")
        return
    print(f"Descargando modelo FaceLandmarker (~6 MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"Modelo descargado: {MODEL_PATH}")
    except Exception as e:
        print(f"Error al descargar el modelo: {e}")
        print(f"Descargalo manualmente desde:\n  {MODEL_URL}")
        print(f"y colocalo como '{MODEL_PATH}' en la misma carpeta que el script.")
        raise

def calcular_lar(landmarks, idx_top, idx_bottom, idx_left, idx_right):
    top    = np.array([landmarks[idx_top].x,    landmarks[idx_top].y])
    bottom = np.array([landmarks[idx_bottom].x, landmarks[idx_bottom].y])
    left   = np.array([landmarks[idx_left].x,   landmarks[idx_left].y])
    right  = np.array([landmarks[idx_right].x,  landmarks[idx_right].y])
    apertura = np.linalg.norm(top - bottom)
    ancho    = np.linalg.norm(left - right)
    if ancho < 1e-6:
        return 0.0
    return apertura / ancho


def suavizar_senal(senal, ventana):
    senal = np.array(senal)
    ventana = min(ventana, len(senal))
    if ventana == 0:
        return senal
    kernel = np.ones(ventana) / ventana
    return np.convolve(senal, kernel, mode='same')


def obtener_bbox(landmarks, ancho_frame, alto_frame, margen=0.05):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x1 = int(max(0.0, min(xs) - margen) * ancho_frame)
    y1 = int(max(0.0, min(ys) - margen) * alto_frame)
    x2 = int(min(1.0, max(xs) + margen) * ancho_frame)
    y2 = int(min(1.0, max(ys) + margen) * alto_frame)
    return x1, y1, x2, y2


def dibujar_labios(frame, landmarks, ancho, alto, color):
    for idx in LIPS_OUTER:
        lm = landmarks[idx]
        cx = int(lm.x * ancho)
        cy = int(lm.y * alto)
        cv2.circle(frame, (cx, cy), 2, color, -1)


def extraer_segmentos(tiempos, lar_vals, umbral, ventana):
    lar_s    = suavizar_senal(lar_vals, ventana)
    hablando = lar_s > umbral
    segmentos = []
    en_seg    = False
    t_inicio  = 0.0
    for i, (t, h) in enumerate(zip(tiempos, hablando)):
        if h and not en_seg:
            t_inicio = t
            en_seg   = True
        elif not h and en_seg:
            dur = tiempos[i - 1] - t_inicio
            if dur > 0.1:
                segmentos.append((t_inicio, tiempos[i - 1], dur))
            en_seg = False
    if en_seg:
        segmentos.append((t_inicio, tiempos[-1], tiempos[-1] - t_inicio))
    return segmentos



#  Procesar video

def procesar_video(video_path):
    assert os.path.exists(video_path), f"No se encontro el video: {video_path}"

    cap          = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    ancho        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto         = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\nVideo: {video_path}")
    print(f"   {ancho}x{alto}  |  {fps:.1f} fps  |  {total_frames} frames  |  {total_frames/fps:.1f}s\n")

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (ancho, alto))
    if not writer.isOpened():
        print("avc1 no disponible, usando mp4v (abri con VLC si falla)")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (ancho, alto))

    historial_lar    = defaultdict(list)
    historial_tiempo = defaultdict(list)

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=MAX_FACES,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.VIDEO,
    )

    with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:

        for frame_num in tqdm(range(total_frames), desc="Procesando frames"):
            ret, frame = cap.read()
            if not ret:
                break

            tiempo    = frame_num / fps
            tiempo_ms = int(tiempo * 1000)

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            resultado = landmarker.detect_for_video(mp_image, tiempo_ms)

            if resultado.face_landmarks:
                for cara_idx, face_lms in enumerate(resultado.face_landmarks):

                    lar = calcular_lar(face_lms, LIP_TOP, LIP_BOTTOM, LIP_LEFT, LIP_RIGHT)
                    historial_lar[cara_idx].append(lar)
                    historial_tiempo[cara_idx].append(tiempo)

                    hablando = lar > LAR_THRESHOLD
                    color    = COLORS[cara_idx % len(COLORS)]
                    x1, y1, x2, y2 = obtener_bbox(face_lms, ancho, alto)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if hablando else 1)
                    dibujar_labios(frame, face_lms, ancho, alto, color)

                    label = f"P{cara_idx} | LAR:{lar:.3f} | {'HABLANDO' if hablando else 'silencio'}"
                    cv2.putText(frame, label, (x1, max(y1 - 10, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                    barra_max = 80
                    barra_val = min(int(lar / 0.1 * barra_max), barra_max)
                    cv2.rectangle(frame, (x1, y2 + 5), (x1 + barra_max, y2 + 15), (60, 60, 60), -1)
                    cv2.rectangle(frame, (x1, y2 + 5), (x1 + barra_val, y2 + 15), color, -1)

            cv2.putText(frame, f"t={tiempo:.2f}s  frame={frame_num}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
            writer.write(frame)

    cap.release()
    writer.release()

    print(f"\nVideo anotado guardado: {OUTPUT_VIDEO}")
    print(f"   Personas detectadas: {len(historial_lar)}")
    return historial_lar, historial_tiempo, fps, total_frames


def graficar_senal(historial_lar, historial_tiempo):
    n = len(historial_lar)
    if n == 0:
        print("No se detectaron caras.")
        return

    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    fig.suptitle("Senal LAR por Persona - Deteccion de Habla",
                 fontsize=14, fontweight="bold", y=1.01)

    for idx, ax in enumerate(axes):
        if idx not in historial_lar:
            ax.set_visible(False)
            continue

        tiempos       = np.array(historial_tiempo[idx])
        lar_raw       = np.array(historial_lar[idx])
        lar_suavizado = suavizar_senal(lar_raw, SMOOTHING_WINDOW)

        bgr       = COLORS[idx % len(COLORS)]
        color_mpl = (bgr[2] / 255, bgr[1] / 255, bgr[0] / 255)

        ax.plot(tiempos, lar_raw, color=color_mpl, alpha=0.3, linewidth=0.8, label="LAR crudo")
        ax.plot(tiempos, lar_suavizado, color=color_mpl, linewidth=2,
                label=f"LAR suavizado (ventana={SMOOTHING_WINDOW})")
        ax.axhline(LAR_THRESHOLD, color="red", linewidth=1.5,
                   linestyle="--", label=f"Umbral = {LAR_THRESHOLD}")

        hablando_mask = lar_suavizado > LAR_THRESHOLD
        ax.fill_between(tiempos, 0, lar_suavizado, where=hablando_mask,
                        alpha=0.25, color=color_mpl, label="Hablando")

        pct = 100 * np.sum(hablando_mask) / len(hablando_mask)
        ax.set_title(f"Persona {idx}  -  {pct:.1f}% del tiempo hablando", fontsize=11)
        ax.set_ylabel("LAR", fontsize=10)
        ax.set_ylim(0, max(0.12, lar_raw.max() * 1.15))
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Tiempo (segundos)", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Grafica guardada: {OUTPUT_PLOT}")


def exportar_resultados(historial_lar, historial_tiempo, fps, total_frames, video_path):
    print("\n" + "=" * 55)
    print("SEGMENTOS DE HABLA DETECTADOS")
    print("=" * 55)

    todos_segmentos = {}

    for pid in sorted(historial_lar.keys()):
        tiempos   = np.array(historial_tiempo[pid])
        vals      = historial_lar[pid]
        segmentos = extraer_segmentos(tiempos, vals, LAR_THRESHOLD, SMOOTHING_WINDOW)
        todos_segmentos[pid] = segmentos

        total_habla = sum(s[2] for s in segmentos)
        print(f"\nPersona {pid}  ({len(segmentos)} segmentos, {total_habla:.1f}s total)")
        print(f"   {'Inicio':>8}  {'Fin':>8}  {'Duracion':>10}")
        print(f"   {'-'*8}  {'-'*8}  {'-'*10}")
        for seg in segmentos:
            print(f"   {seg[0]:>7.2f}s  {seg[1]:>7.2f}s  {seg[2]:>9.2f}s")

    export = {
        "video": video_path,
        "fps": fps,
        "total_frames": total_frames,
        "duracion_seg": round(total_frames / fps, 3),
        "umbral_lar": LAR_THRESHOLD,
        "n_personas": len(todos_segmentos),
        "segmentos_por_persona": {
            str(pid): [
                {"inicio": round(s[0], 3), "fin": round(s[1], 3), "duracion": round(s[2], 3)}
                for s in segs
            ]
            for pid, segs in todos_segmentos.items()
        },
        "lar_series": {
            str(pid): {
                "tiempos": [round(t, 3) for t in historial_tiempo[pid]],
                "valores": [round(v, 5) for v in historial_lar[pid]]
            }
            for pid in historial_lar
        }
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(export, f, indent=2)

    print(f"\nDatos exportados: {OUTPUT_JSON}")
    print("   -> Entrada del Modulo 2 (pyannote + Whisper)")

def diagnostico_umbral(historial_lar):
    n = len(historial_lar)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        if idx not in historial_lar:
            continue
        vals      = np.array(historial_lar[idx])
        bgr       = COLORS[idx % len(COLORS)]
        color_mpl = (bgr[2] / 255, bgr[1] / 255, bgr[0] / 255)

        ax.hist(vals, bins=50, color=color_mpl, alpha=0.7, edgecolor="white")
        ax.axvline(LAR_THRESHOLD, color="red", linewidth=2,
                   linestyle="--", label=f"Umbral actual = {LAR_THRESHOLD}")
        umbral_auto = np.percentile(vals, 60)
        ax.axvline(umbral_auto, color="orange", linewidth=1.5,
                   linestyle=":", label=f"Umbral sugerido = {umbral_auto:.3f}")

        ax.set_title(f"Persona {idx} - Distribucion LAR", fontsize=11)
        ax.set_xlabel("LAR")
        ax.set_ylabel("Frecuencia")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        print(f"Persona {idx}: min={vals.min():.4f}  max={vals.max():.4f}  "
              f"media={vals.mean():.4f}  umbral_sugerido={umbral_auto:.4f}")

    plt.suptitle("Diagnostico de Umbral LAR", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("diagnostico_umbral.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nMuchos falsos positivos -> subi LAR_THRESHOLD")
    print("Muchos falsos negativos  -> baja LAR_THRESHOLD")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deteccion de labios y senal LAR")
    parser.add_argument("--video",       required=True,                      help="Ruta al video (ej: video.mp4)")
    parser.add_argument("--umbral",      type=float, default=LAR_THRESHOLD,  help=f"Umbral LAR (default: {LAR_THRESHOLD})")
    parser.add_argument("--diagnostico", action="store_true",                help="Grafica de distribucion para ajustar umbral")
    args = parser.parse_args()

    LAR_THRESHOLD = args.umbral

    descargar_modelo()

    historial_lar, historial_tiempo, fps, total_frames = procesar_video(args.video)
    graficar_senal(historial_lar, historial_tiempo)
    exportar_resultados(historial_lar, historial_tiempo, fps, total_frames, args.video)

    if args.diagnostico:
        diagnostico_umbral(historial_lar)