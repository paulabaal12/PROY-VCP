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

OUTPUT_SRT   = "results/srt/subtitulos.srt"
OUTPUT_VIDEO = "results/videos/video_subtitulado.mp4"
AUDIO_TMP    = "data/audio/audio_tmp.wav"


def extraer_audio(video_path, salida=AUDIO_TMP):
    os.makedirs(os.path.dirname(salida), exist_ok=True)
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
    from huggingface_hub import login
    login(token=token)
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    pipeline.to(torch.device("cpu"))
    return pipeline


def diarizar_audio(audio_path, pipeline, min_speakers=None, max_speakers=None):
    print("\n[2/5] Diarizando audio...")
    with ProgressHook() as hook:
        kwargs = {}
        if min_speakers: kwargs["min_speakers"] = min_speakers
        if max_speakers: kwargs["max_speakers"] = max_speakers
        diarizacion = pipeline(audio_path, hook=hook, **kwargs)

    segmentos = []
    for turno, _, hablante in diarizacion.itertracks(yield_label=True):
        segmentos.append({
            "speaker":  hablante,
            "inicio":   round(turno.start, 3),
            "fin":      round(turno.end,   3),
            "duracion": round(turno.end - turno.start, 3)
        })

    print(f"Segmentos: {len(segmentos)}")
    return segmentos


def fusionar_con_visual(segmentos_audio, lip_data):
    """
    Para cada segmento de audio, identifica qué persona visual tiene
    el LAR promedio más alto en ese intervalo de tiempo.
    Asignación segmento a segmento — no acumulativa.
    """
    umbral = lip_data.get("umbral_lar", 0.03)

    # Pre-cargar series LAR por persona
    series = {}
    for pid_str, serie in lip_data["lar_series"].items():
        pid = int(pid_str)
        tiempos = np.array(serie["tiempos"])
        valores = np.array(serie["valores"])
        series[pid] = (tiempos, valores)

    # Para cada segmento de audio → encontrar persona con mayor LAR promedio
    asignacion_por_segmento = []

    for seg in segmentos_audio:
        t0, t1 = seg["inicio"], seg["fin"]
        mejor_pid   = -1
        mejor_score = -1.0

        for pid, (tiempos, valores) in series.items():
            mask  = (tiempos >= t0) & (tiempos <= t1)
            total = np.sum(mask)
            if total == 0:
                continue

            # Score = LAR promedio en el intervalo (no solo binario hablando/no)
            media_persona = np.mean(valores)
            std_persona = np.std(valores) + 1e-6
            lar_normalizado = np.mean((valores[mask] - media_persona) / std_persona)
            lar_promedio = lar_normalizado

            if lar_promedio > mejor_score:
                mejor_score = lar_promedio
                mejor_pid   = pid

        asignacion_por_segmento.append(mejor_pid)
        seg["persona_visual"] = mejor_pid

    # Construir mapeo speaker_audio → pid_visual
    # Un speaker de audio puede aparecer en varios segmentos;
    # tomamos el pid visual más frecuente entre sus segmentos
    votos_speaker = {}
    for seg, pid in zip(segmentos_audio, asignacion_por_segmento):
        sp = seg["speaker"]
        votos_speaker.setdefault(sp, {})
        votos_speaker[sp][pid] = votos_speaker[sp].get(pid, 0) + 1

    mapeo = {}
    for sp, conteos in votos_speaker.items():
        # Ignorar pid=-1 si hay otras opciones
        conteos_validos = {p: c for p, c in conteos.items() if p >= 0}
        if conteos_validos:
            mapeo[sp] = max(conteos_validos, key=conteos_validos.get)
        else:
            mapeo[sp] = -1

    # Re-asignar persona_visual usando el mapeo consolidado
    for seg in segmentos_audio:
        seg["persona_visual"] = mapeo.get(seg["speaker"], -1)

    # Debug: mostrar mapeo
    print("\nMapeo speaker audio → persona visual:")
    for sp, pid in sorted(mapeo.items()):
        print(f"  {sp} → Persona {pid}")

    return segmentos_audio, mapeo


def transcribir_segmentos(segmentos, audio_array, sr, modelo):
    print("\n[3/5] Transcribiendo...")
    model = whisper.load_model(modelo)

    for i, seg in enumerate(segmentos):
        t0 = int(seg["inicio"] * sr)
        t1 = int(seg["fin"]    * sr)

        if (t1 - t0) < sr * 0.5:
            seg["texto"] = ""
            continue

        fragmento = audio_array[t0:t1]
        result    = model.transcribe(fragmento, fp16=False)
        seg["texto"] = result["text"].strip()

        if (i + 1) % 10 == 0 or (i + 1) == len(segmentos):
            print(f"{i+1}/{len(segmentos)}")

    return segmentos


def segundos_a_srt(seg):
    h  = int(seg // 3600)
    m  = int((seg % 3600) // 60)
    s  = int(seg % 60)
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
            fin    = segundos_a_srt(seg["fin"])
            pid    = mapeo.get(seg["speaker"], -1)
            etiqueta = f"Persona {pid}:" if pid >= 0 else ""

            f.write(f"{idx}\n{inicio} --> {fin}\n{etiqueta} {texto}\n\n")
            idx += 1


def generar_video_con_subtitulos(video_input):
    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

    srt_path = os.path.abspath(OUTPUT_SRT)
    srt_path = srt_path.replace("\\", "\\\\")
    srt_path = srt_path.replace(":", "\\:")

    estilo  = "Fontsize=28,PrimaryColour=&Hffffff&,OutlineColour=&H000000&,BorderStyle=3,Outline=2,Shadow=3,Alignment=2"
    filtro  = f"subtitles='{srt_path}':force_style='{estilo}'"

    cmd = [
        "ffmpeg", "-y",
        "-i", video_input,
        "-vf", filtro,
        "-c:a", "aac",
        OUTPUT_VIDEO
    ]
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",  required=True)
    parser.add_argument("--json",   required=True)
    parser.add_argument("--token",  default=None)
    parser.add_argument("--modelo", default="base")
    parser.add_argument("--min_speakers", type=int, default=None)
    parser.add_argument("--max_speakers", type=int, default=None)
    args = parser.parse_args()

    if args.token is None:
        args.token = _leer_token_env()
    if not args.token:
        raise ValueError("No se encontró token. Agrégalo en .env (TOKEN=hf_...) o pásalo con --token")

    with open(args.json) as f:
        lip_data = json.load(f)

    print("\n[1/5] Extrayendo audio...")
    audio_path = extraer_audio(args.video)

    pipeline  = cargar_pipeline(args.token)
    segmentos = diarizar_audio(audio_path, pipeline, 
                           min_speakers=args.min_speakers,
                           max_speakers=args.max_speakers)

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