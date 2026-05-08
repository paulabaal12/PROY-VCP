import os
import json
import argparse
import subprocess
import warnings
import numpy as np
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import torch
import whisper
import torchaudio

warnings.filterwarnings("ignore", category=UserWarning)


def _leer_token_env(path=".env"):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("TOKEN"):
                return line.split("=", 1)[1].strip()
    return None

OUTPUT_SRT = "results/srt/subtitulos.srt"
OUTPUT_VIDEO = "results/videos/video_subtitulado.mp4"
AUDIO_TMP = "data/audio/audio_tmp.wav"

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
    print("\n[2/5] Diarizando audio...")
    with ProgressHook() as hook:
        diarizacion = pipeline(audio_path, hook=hook)

    segmentos = []
    for turno, _, hablante in diarizacion.itertracks(yield_label=True):
        segmentos.append({
            "speaker": hablante,
            "inicio": round(turno.start, 3),
            "fin": round(turno.end, 3),
            "duracion": round(turno.end - turno.start, 3)
        })

    print(f"Segmentos: {len(segmentos)}")
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

def transcribir_segmentos(segmentos, audio_array, sr, modelo):
    print("\n[3/5] Transcribiendo...")
    model = whisper.load_model(modelo)

    for i, seg in enumerate(segmentos):
        t0 = int(seg["inicio"] * sr)
        t1 = int(seg["fin"] * sr)

        if (t1 - t0) < sr * 0.5:
            seg["texto"] = ""
            continue

        fragmento = audio_array[t0:t1]
        result = model.transcribe(fragmento, fp16=False)
        seg["texto"] = result["text"].strip()

        if (i + 1) % 10 == 0 or (i + 1) == len(segmentos):
            print(f"{i+1}/{len(segmentos)}")

    return segmentos

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
            etiqueta = f"Persona {pid}:" if pid >= 0 else ""

            f.write(f"{idx}\n{inicio} --> {fin}\n{etiqueta} {texto}\n\n")
            idx += 1

def generar_video_con_subtitulos(video_input):
    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

    srt_path = os.path.abspath(OUTPUT_SRT)

    srt_path = srt_path.replace("\\", "\\\\")
    srt_path = srt_path.replace(":", "\\:")

    estilo = "Fontsize=28,PrimaryColour=&Hffffff&,OutlineColour=&H000000&,BorderStyle=3,Outline=2,Shadow=3,Alignment=2"

    filtro = f"subtitles='{srt_path}':force_style='{estilo}'"

    cmd = [
        "ffmpeg", "-y",
        "-i", video_input,
        "-vf", filtro,
        "-c:a", "copy",
        OUTPUT_VIDEO
    ]

    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--token", default=None)
    parser.add_argument("--modelo", default="base")

    args = parser.parse_args()

    if args.token is None:
        args.token = _leer_token_env()
    if not args.token:
        raise ValueError("No se encontró token. Agrégalo en .env (TOKEN=hf_...) o pásalo con --token")

    with open(args.json) as f:
        lip_data = json.load(f)

    print("\n[1/5] Extrayendo audio...")
    audio_path = extraer_audio(args.video)

    pipeline = cargar_pipeline(args.token)
    segmentos = diarizar_audio(audio_path, pipeline)

    segmentos, mapeo = fusionar_con_visual(segmentos, lip_data)

    audio_array, sr = cargar_audio_array(audio_path)
    segmentos = transcribir_segmentos(segmentos, audio_array, sr, args.modelo)

    print("\n[4/5] Generando SRT...")
    generar_srt(segmentos, mapeo)

    print("\n[5/5] Generando video con subtítulos...")
    generar_video_con_subtitulos(args.video)

    print(f"\nVideo final: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()