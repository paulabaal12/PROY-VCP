import cv2
import json
import argparse
import subprocess
import numpy as np
import os

OUTPUT_VIDEO_NOAUDIO = "results/videos/video_final_tmp.mp4"
OUTPUT_VIDEO_FINAL   = "results/videos/video_final.mp4"
OUTPUT_METRICAS      = "results/logs/metricas.txt"

COLORS = [(0,255,100),(0,180,255),(255,80,80),(200,0,255),(255,200,0),(200,0,200)]
LAR_THRESHOLD = 0.03
EVAL_STEP     = 0.1  


# ─────────────────────────────────────────────
#  LECTURA DE ARCHIVOS
# ─────────────────────────────────────────────

def leer_srt(srt_path):
    subtitulos = []
    with open(srt_path, encoding="utf-8") as f:
        contenido = f.read().strip()
    for bloque in contenido.split("\n\n"):
        lineas = bloque.strip().split("\n")
        if len(lineas) < 3:
            continue
        tiempos = lineas[1].split(" --> ")
        inicio  = srt_a_segundos(tiempos[0].strip())
        fin     = srt_a_segundos(tiempos[1].strip())
        texto   = " ".join(lineas[2:]).strip()
        subtitulos.append({"inicio": inicio, "fin": fin, "texto": texto})
    return subtitulos


def srt_a_segundos(s):
    h, m, rest = s.split(":")
    seg, ms = rest.split(",")
    return int(h)*3600 + int(m)*60 + int(seg) + int(ms)/1000


# ─────────────────────────────────────────────
#  COMPOSICIÓN DEL VIDEO FINAL
# ─────────────────────────────────────────────

def obtener_subtitulo(t, subtitulos):
    for sub in subtitulos:
        if sub["inicio"] <= t <= sub["fin"]:
            return sub["texto"]
    return None


def dibujar_subtitulo(frame, texto, w, h):
    if not texto:
        return
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness  = 2
    max_chars  = 80
    palabras   = texto.split()
    lineas     = []
    linea      = ""
    for p in palabras:
        if len(linea) + len(p) + 1 <= max_chars:
            linea += (" " if linea else "") + p
        else:
            if linea:
                lineas.append(linea)
            linea = p
    if linea:
        lineas.append(linea)

    line_h  = 30
    total_h = len(lineas) * line_h + 20
    y_start = h - total_h - 20

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y_start - 10), (w, h - 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    for i, linea in enumerate(lineas):
        (tw, th), _ = cv2.getTextSize(linea, font, font_scale, thickness)
        x = (w - tw) // 2
        y = y_start + i * line_h + th
        cv2.putText(frame, linea, (x, y), font, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(frame, linea, (x, y), font, font_scale, (255, 255, 255), thickness)


def obtener_info_frame(lip_data, t):
    resultados = []
    for pid_str, serie in lip_data["lar_series"].items():
        pid     = int(pid_str)
        tiempos = np.array(serie["tiempos"])
        valores = np.array(serie["valores"])
        bboxes  = serie.get("bboxes", [])

        idx = np.searchsorted(tiempos, t, side="right") - 1
        if 0 <= idx < len(valores):
            lar      = valores[idx]
            hablando = lar > LAR_THRESHOLD
            bbox     = bboxes[idx] if idx < len(bboxes) else None
            resultados.append({"pid": pid, "lar": lar, "hablando": hablando, "bbox": bbox})
    return resultados


def procesar(video_path, json_path, srt_path):
    with open(json_path) as f:
        lip_data = json.load(f)

    subtitulos = leer_srt(srt_path)

    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {w}x{h} | {fps:.1f} fps | {total} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO_NOAUDIO, fourcc, fps, (w, h))

    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        t     = i / fps
        infos = obtener_info_frame(lip_data, t)

        for info in infos:
            pid      = info["pid"]
            hablando = info["hablando"]
            color    = COLORS[pid % len(COLORS)]
            grosor   = 3 if hablando else 1
            bbox     = info["bbox"]

            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, grosor)
                label = f"P{pid} {'HABLA' if hablando else '...'}"
                cv2.putText(frame, label, (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
                cv2.putText(frame, label, (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        texto = obtener_subtitulo(t, subtitulos)
        dibujar_subtitulo(frame, texto, w, h)

        writer.write(frame)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{total} frames")

    cap.release()
    writer.release()
    print("Frames listos, agregando audio...")


def agregar_audio(video_path):
    os.makedirs("results/videos", exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", OUTPUT_VIDEO_NOAUDIO,
        "-i", video_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        OUTPUT_VIDEO_FINAL
    ]
    subprocess.run(cmd)
    os.remove(OUTPUT_VIDEO_NOAUDIO)
    print(f"\nVideo final: {OUTPUT_VIDEO_FINAL}")


# ─────────────────────────────────────────────
#  EVALUACIÓN DE MÉTRICAS
# ─────────────────────────────────────────────

def extraer_personas_srt(subtitulos):
    """
    Del SRT generado por diarizacion.py extrae qué persona habla en cada intervalo.
    Formato esperado en texto: 'Persona 0: blah blah' o 'Persona 1: ...'
    Devuelve lista de {pid, inicio, fin}.
    """
    segmentos = []
    for sub in subtitulos:
        texto = sub["texto"]
        if texto.startswith("Persona "):
            try:
                pid = int(texto.split(":")[0].replace("Persona", "").strip())
                segmentos.append({"pid": pid, "inicio": sub["inicio"], "fin": sub["fin"]})
            except ValueError:
                pass
    return segmentos


def muestrear_señal(lip_data, pid, duracion, paso=EVAL_STEP):
    """Devuelve array binario (habla=1, silencio=0) muestreado cada `paso` segundos."""
    serie   = lip_data["lar_series"].get(str(pid))
    if serie is None:
        return np.array([])
    tiempos = np.array(serie["tiempos"])
    valores = np.array(serie["valores"])
    ticks   = np.arange(0, duracion, paso)
    señal   = np.zeros(len(ticks), dtype=int)
    for k, t in enumerate(ticks):
        idx = np.searchsorted(tiempos, t, side="right") - 1
        if 0 <= idx < len(valores):
            señal[k] = int(valores[idx] > LAR_THRESHOLD)
    return señal, ticks


def muestrear_ground_truth(segmentos_srt, pid, duracion, paso=EVAL_STEP):
    """Devuelve array binario basado en los segmentos del SRT para la persona pid."""
    ticks = np.arange(0, duracion, paso)
    gt    = np.zeros(len(ticks), dtype=int)
    for seg in segmentos_srt:
        if seg["pid"] == pid:
            mask = (ticks >= seg["inicio"]) & (ticks <= seg["fin"])
            gt[mask] = 1
    return gt


def calcular_metricas(gt, pred):
    """Calcula TP, FP, FN, TN y deriva Precisión, Recall, F1 y Accuracy."""
    tp = int(np.sum((pred == 1) & (gt == 1)))
    fp = int(np.sum((pred == 1) & (gt == 0)))
    fn = int(np.sum((pred == 0) & (gt == 1)))
    tn = int(np.sum((pred == 0) & (gt == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (tp + tn) / len(gt) if len(gt) > 0 else 0.0

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall,
            "f1": f1, "accuracy": accuracy}


def evaluar_metricas(json_path, srt_path):
    """
    Función principal de evaluación.
    Compara la detección LAR (lip_tracking_data.json) contra el ground truth
    derivado automáticamente del SRT generado por diarizacion.py.
    """
    print("\n" + "=" * 60)
    print("  EVALUACIÓN DE MÉTRICAS")
    print("=" * 60)

    with open(json_path) as f:
        lip_data = json.load(f)

    subtitulos      = leer_srt(srt_path)
    segmentos_gt    = extraer_personas_srt(subtitulos)
    duracion        = lip_data.get("duracion_seg", 0)
    personas        = sorted(lip_data["lar_series"].keys(), key=int)

    if not segmentos_gt:
        print("\n[!] No se encontraron etiquetas 'Persona X:' en el SRT.")
        print("    Asegurate de que diarizacion.py generó el archivo con ese formato.")
        return

    personas_gt = sorted(set(s["pid"] for s in segmentos_gt))
    print(f"\nDuración del video : {duracion:.1f}s")
    print(f"Paso de muestreo   : {EVAL_STEP}s  ({int(duracion/EVAL_STEP)} muestras)")
    print(f"Personas en SRT    : {personas_gt}")
    print(f"Personas en JSON   : {[int(p) for p in personas]}")

    resultados = {}

    for pid_str in personas:
        pid = int(pid_str)

        señal_resultado = muestrear_señal(lip_data, pid, duracion)
        if len(señal_resultado) == 0:
            continue
        pred, ticks = señal_resultado

        gt   = muestrear_ground_truth(segmentos_gt, pid, duracion)
        met  = calcular_metricas(gt, pred)
        resultados[pid] = met

        print(f"\n  Persona {pid}")
        print(f"    Ground truth (SRT) habla : {np.sum(gt) * EVAL_STEP:.1f}s")
        print(f"    Detección LAR habla      : {np.sum(pred) * EVAL_STEP:.1f}s")
        print(f"    TP={met['tp']}  FP={met['fp']}  FN={met['fn']}  TN={met['tn']}")
        print(f"    Precisión  : {met['precision']:.3f}")
        print(f"    Recall     : {met['recall']:.3f}")
        print(f"    F1-score   : {met['f1']:.3f}")
        print(f"    Accuracy   : {met['accuracy']:.3f}")

    # ── Promedio macro ──
    if resultados:
        prec_avg = np.mean([m["precision"] for m in resultados.values()])
        rec_avg  = np.mean([m["recall"]    for m in resultados.values()])
        f1_avg   = np.mean([m["f1"]        for m in resultados.values()])
        acc_avg  = np.mean([m["accuracy"]  for m in resultados.values()])

        print("\n" + "-" * 40)
        print("  PROMEDIO MACRO (todas las personas)")
        print(f"    Precisión  : {prec_avg:.3f}")
        print(f"    Recall     : {rec_avg:.3f}")
        print(f"    F1-score   : {f1_avg:.3f}")
        print(f"    Accuracy   : {acc_avg:.3f}")

    # ── Guardar reporte ──
    os.makedirs(os.path.dirname(OUTPUT_METRICAS), exist_ok=True)
    with open(OUTPUT_METRICAS, "w", encoding="utf-8") as f:
        f.write("REPORTE DE MÉTRICAS — Lip Tracking vs Ground Truth (SRT)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Duración        : {duracion:.1f}s\n")
        f.write(f"Paso muestreo   : {EVAL_STEP}s\n")
        f.write(f"Umbral LAR      : {LAR_THRESHOLD}\n\n")

        for pid, met in resultados.items():
            f.write(f"Persona {pid}\n")
            f.write(f"  TP={met['tp']}  FP={met['fp']}  FN={met['fn']}  TN={met['tn']}\n")
            f.write(f"  Precisión : {met['precision']:.4f}\n")
            f.write(f"  Recall    : {met['recall']:.4f}\n")
            f.write(f"  F1-score  : {met['f1']:.4f}\n")
            f.write(f"  Accuracy  : {met['accuracy']:.4f}\n\n")

        if resultados:
            f.write("PROMEDIO MACRO\n")
            f.write(f"  Precisión : {prec_avg:.4f}\n")
            f.write(f"  Recall    : {rec_avg:.4f}\n")
            f.write(f"  F1-score  : {f1_avg:.4f}\n")
            f.write(f"  Accuracy  : {acc_avg:.4f}\n")

    print(f"\nReporte guardado: {OUTPUT_METRICAS}")
    print("=" * 60)


# ─────────────────────────────────────────────
#  PUNTO DE ENTRADA
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",    required=True,  help="Video original")
    parser.add_argument("--json",     required=True,  help="lip_tracking_data.json")
    parser.add_argument("--srt",      required=True,  help="subtitulos.srt")
    parser.add_argument("--metricas", action="store_true",
                        help="Solo calcular métricas, sin recomponer el video")
    args = parser.parse_args()

    if args.metricas:
        evaluar_metricas(args.json, args.srt)
    else:
        procesar(args.video, args.json, args.srt)
        agregar_audio(args.video)
        evaluar_metricas(args.json, args.srt)  


if __name__ == "__main__":
    main()