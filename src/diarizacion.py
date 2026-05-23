import os
import json
import argparse
import subprocess
import warnings
import numpy as np
from scipy.ndimage import median_filter
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

# ── MEJORA 1: parámetros de suavizado y duración mínima ──────────────────────
LAR_SMOOTH_FRAMES   = 7    # ventana del filtro de mediana (frames)
MIN_SPEECH_DURATION = 0.3  # segundos mínimos para considerar que alguien habla


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


# ── MEJORA 2: umbral adaptativo por persona ───────────────────────────────────
def calcular_umbral_adaptativo(valores, percentil_base=70, factor=0.4, umbral_min=0.05):
    """
    Calcula un umbral LAR personalizado para cada persona basado en
    la distribución de sus propios valores.

    Idea: el umbral es una fracción del percentil 70 de sus valores LAR.
    Esto adapta el detector al rango de movimiento natural de cada persona.
    Se impone un mínimo global de 0.03 para evitar falsos positivos en silencio.
    """
    if len(valores) == 0:
        return umbral_min
    p70 = np.percentile(valores, percentil_base)
    umbral = max(umbral_min, p70 * factor)
    return round(umbral, 4)


def suavizar_lar(valores, ventana=LAR_SMOOTH_FRAMES):
    """
    Aplica un filtro de mediana sobre la serie LAR para eliminar
    picos espúreos frame a frame (micro-movimientos de labios en silencio).
    """
    if len(valores) < ventana:
        return valores
    return median_filter(valores, size=ventana)

def fusionar_con_visual(segmentos_audio, lip_data):
    series = {}
    umbrales = {}
    for pid_str, serie in lip_data["lar_series"].items():
        pid = int(pid_str)
        tiempos = np.array(serie["tiempos"])
        valores_raw = np.array(serie["valores"])
        valores = suavizar_lar(valores_raw)
        umbral = calcular_umbral_adaptativo(valores)
        umbrales[pid] = umbral
        series[pid] = (tiempos, valores)

    print("\nUmbrales LAR adaptativos por persona:")
    for pid, u in sorted(umbrales.items()):
        print(f"  Persona {pid} → umbral = {u:.4f}")

    # Score acumulado por (speaker_audio, pid_visual)
    scores_acum = {}
    conteos     = {}

    for seg in segmentos_audio:
        t0, t1 = seg["inicio"], seg["fin"]
        sp = seg["speaker"]

        for pid, (tiempos, valores) in series.items():
            mask  = (tiempos >= t0) & (tiempos <= t1)
            if np.sum(mask) == 0:
                continue
            media = np.mean(valores)
            std   = np.std(valores) + 1e-6
            score = np.mean((valores[mask] - media) / std)
            key   = (sp, pid)
            scores_acum[key] = scores_acum.get(key, 0.0) + score
            conteos[key]     = conteos.get(key, 0) + 1

    # Normalizar scores
    scores_norm = {k: v / conteos[k] for k, v in scores_acum.items()}

    speakers  = sorted(set(seg["speaker"] for seg in segmentos_audio))
    pids_vis  = sorted(series.keys())

    # Asignación greedy: el mejor par (speaker, pid) que no repita pid
    asignados_pid = set()
    mapeo = {}

    pares_ordenados = sorted(scores_norm.items(), key=lambda x: x[1], reverse=True)
    for (sp, pid), score in pares_ordenados:
        if sp not in mapeo and pid not in asignados_pid:
            mapeo[sp] = pid
            asignados_pid.add(pid)

    # Si algún speaker quedó sin asignar, darle el pid restante
    for sp in speakers:
        if sp not in mapeo:
            restantes = [p for p in pids_vis if p not in asignados_pid]
            if restantes:
                mapeo[sp] = restantes[0]
                asignados_pid.add(restantes[0])
            else:
                mapeo[sp] = -1

    for seg in segmentos_audio:
        seg["persona_visual"] = mapeo.get(seg["speaker"], -1)

    print("\nMapeo speaker audio → persona visual:")
    for sp, pid in sorted(mapeo.items()):
        print(f"  {sp} → Persona {pid}")

    lip_data["umbrales_adaptativos"] = {str(k): v for k, v in umbrales.items()}
    return segmentos_audio, mapeo


def filtrar_segmentos_cortos(segmentos, duracion_min=MIN_SPEECH_DURATION):
    """
    Elimina del SRT los segmentos donde la detección LAR activa dura
    menos de `duracion_min` segundos — probablemente son ruido.
    """
    antes = len(segmentos)
    filtrados = [s for s in segmentos if s.get("duracion", 0) >= duracion_min]
    descartados = antes - len(filtrados)
    if descartados:
        print(f"\n  Filtrados {descartados} segmentos < {duracion_min}s (ruido)")
    return filtrados


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

            inicio   = segundos_a_srt(seg["inicio"])
            fin      = segundos_a_srt(seg["fin"])
            pid      = mapeo.get(seg["speaker"], -1)
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
        "-c:a", "aac",
        OUTPUT_VIDEO
    ]
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",        required=True)
    parser.add_argument("--json",         required=True)
    parser.add_argument("--token",        default=None)
    parser.add_argument("--modelo",       default="base")
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

    # Filtrar segmentos muy cortos antes de fusionar
    segmentos = filtrar_segmentos_cortos(segmentos)

    segmentos, mapeo = fusionar_con_visual(segmentos, lip_data)

    audio_array, sr = cargar_audio_array(audio_path)
    segmentos = transcribir_segmentos(segmentos, audio_array, sr, args.modelo)

    print("\n[4/5] Generando SRT...")
    generar_srt(segmentos, mapeo)

    # Guardar lip_data actualizado (con umbrales adaptativos)
    with open(args.json, "w") as f:
        json.dump(lip_data, f, indent=2)

    print("\n[5/5] Generando video con subtítulos...")
    generar_video_con_subtitulos(args.video)

    print(f"\nVideo final: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()