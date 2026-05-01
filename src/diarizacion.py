import os
import json
import argparse
import subprocess
import warnings
from multiprocessing import Pool, cpu_count

import numpy as np
import matplotlib.pyplot as plt

from pyannote.audio import Pipeline
import torch
import whisper
import torchaudio

warnings.filterwarnings("ignore")


OUTPUT_SRT   = "results/srt/subtitulos.srt"
OUTPUT_JSON  = "results/logs/diarizacion_data.json"
CACHE_JSON   = "results/cache/transcripciones_cache.json"
OUTPUT_PLOT  = "results/plots/diarizacion.png"
AUDIO_TMP    = "data/audio/audio_tmp.wav"


def extraer_audio(video_path, salida=AUDIO_TMP):
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1", "-ar", "16000", "-vn", salida
    ]
    subprocess.run(cmd, capture_output=True)
    return salida


def cargar_audio_array(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    return waveform.squeeze().numpy(), sr

def cargar_pipeline(token):
    pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=token
)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(torch.device(device))
    return pipeline


def diarizar_audio(audio_path, pipeline):
    diarizacion = pipeline(audio_path)

    segmentos = []
    for turno, _, hablante in diarizacion.itertracks(yield_label=True):
        segmentos.append({
            "speaker": hablante,
            "inicio": round(turno.start, 3),
            "fin": round(turno.end, 3),
            "duracion": round(turno.end - turno.start, 3)
        })
    return segmentos

def fusionar_con_visual(segmentos_audio, lip_data):
    visual_por_persona = {}

    for pid_str, serie in lip_data["lar_series"].items():
        pid = int(pid_str)
        tiempos = np.array(serie["tiempos"])
        valores = np.array(serie["valores"])

        umbral = lip_data.get("umbral_lar", 0.03)
        hablando = valores > umbral

        visual_por_persona[pid] = (tiempos, hablando)

    votos = {}

    for seg in segmentos_audio:
        sp = seg["speaker"]
        t0, t1 = seg["inicio"], seg["fin"]
        votos.setdefault(sp, {})

        for pid, (tiempos, hablando) in visual_por_persona.items():
            mask = (tiempos >= t0) & (tiempos <= t1)

            total = np.sum(mask)
            activos = np.sum(hablando[mask])

            if total > 0:
                ratio = activos / total
                votos[sp][pid] = votos[sp].get(pid, 0) + ratio

    mapeo = {
        sp: max(c, key=c.get) if c else -1
        for sp, c in votos.items()
    }

    for seg in segmentos_audio:
        seg["persona_visual"] = mapeo.get(seg["speaker"], -1)

    return segmentos_audio, mapeo

def cargar_cache():
    if os.path.exists(CACHE_JSON):
        with open(CACHE_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def guardar_cache(cache):
    os.makedirs(os.path.dirname(CACHE_JSON), exist_ok=True)
    with open(CACHE_JSON, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def transcribir_segmento(args):
    seg, audio_array, sr, modelo = args

    key = f"{seg['inicio']:.2f}-{seg['fin']:.2f}"

    # Cada proceso carga su propio modelo (necesario en multiprocessing)
    model = whisper.load_model(modelo)

    t0 = int(seg["inicio"] * sr)
    t1 = int(seg["fin"] * sr)

    if (t1 - t0) < sr * 0.5:
        seg["texto"] = ""
        return seg

    fragmento = audio_array[t0:t1]

    result = model.transcribe(fragmento, language="es", fp16=False)
    seg["texto"] = result["text"].strip()

    return seg


def transcribir_segmentos_paralelo(segmentos, audio_array, sr, modelo):
    print("\nTranscribiendo en paralelo...")

    cache = cargar_cache()
    resultados = []

    tareas = []

    for seg in segmentos:
        key = f"{seg['inicio']:.2f}-{seg['fin']:.2f}"

        if key in cache:
            seg["texto"] = cache[key]
            resultados.append(seg)
        else:
            tareas.append((seg, audio_array, sr, modelo))

    print(f"   Nuevos segmentos: {len(tareas)}")

    if tareas:
        with Pool(processes=max(1, cpu_count() - 1)) as pool:
            nuevos = pool.map(transcribir_segmento, tareas)

        for seg in nuevos:
            key = f"{seg['inicio']:.2f}-{seg['fin']:.2f}"
            cache[key] = seg["texto"]

        resultados.extend(nuevos)

    guardar_cache(cache)
    return resultados

def segundos_a_srt(seg):
    h = int(seg // 3600)
    m = int((seg % 3600) // 60)
    s = int(seg % 60)
    ms = int((seg - int(seg)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def generar_srt(segmentos, mapeo):
    os.makedirs(os.path.dirname(OUTPUT_SRT), exist_ok=True)

    with open(OUTPUT_SRT, "w", encoding="utf-8") as f:
        idx = 1

        for seg in segmentos:
            texto = seg.get("texto", "").strip()
            if not texto:
                continue

            inicio = segundos_a_srt(seg["inicio"])
            fin = segundos_a_srt(seg["fin"])

            pid = mapeo.get(seg["speaker"], -1)
            etiqueta = f"[Persona {pid}]" if pid >= 0 else "[?]"

            f.write(f"{idx}\n{inicio} --> {fin}\n{etiqueta}: {texto}\n\n")
            idx += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--modelo", default="base")

    args = parser.parse_args()

    with open(args.json) as f:
        lip_data = json.load(f)

    audio_path = extraer_audio(args.video)

    pipeline = cargar_pipeline(args.token)
    segmentos = diarizar_audio(audio_path, pipeline)

    segmentos, mapeo = fusionar_con_visual(segmentos, lip_data)

    audio_array, sr = cargar_audio_array(audio_path)

    segmentos = transcribir_segmentos_paralelo(
        segmentos, audio_array, sr, args.modelo
    )

    generar_srt(segmentos, mapeo)

    print("\nProceso completo")


if __name__ == "__main__":
    main()