import cv2
import json
import argparse
import subprocess
import numpy as np
import os

OUTPUT_VIDEO_NOAUDIO = "results/videos/video_final_tmp.mp4"
OUTPUT_VIDEO_FINAL   = "results/videos/video_final.mp4"

COLORS = [(0,255,100),(0,180,255),(255,80,80),(200,0,255),(255,200,0),(200,0,200)]
LAR_THRESHOLD = 0.03


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Video original")
    parser.add_argument("--json",  required=True, help="lip_tracking_data.json")
    parser.add_argument("--srt",   required=True, help="subtitulos.srt")
    args = parser.parse_args()

    procesar(args.video, args.json, args.srt)
    agregar_audio(args.video)


if __name__ == "__main__":
    main()