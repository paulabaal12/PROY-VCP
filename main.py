import argparse
import subprocess
import sys
import os
import shutil
from datetime import datetime

PYTHON = sys.executable


def correr(cmd, descripcion):
    print(f"\n{'='*55}")
    print(f"  {descripcion}")
    print(f"{'='*55}")
    resultado = subprocess.run(cmd, check=True, env={**os.environ})
    if resultado.returncode != 0:
        print(f"Error en: {descripcion}")
        sys.exit(1)


def guardar_corrida(video_path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destino   = f"results/runs/{timestamp}"

    os.makedirs(f"{destino}/videos", exist_ok=True)
    os.makedirs(f"{destino}/logs",   exist_ok=True)
    os.makedirs(f"{destino}/plots",  exist_ok=True)
    os.makedirs(f"{destino}/srt",    exist_ok=True)
    os.makedirs(f"{destino}/json",   exist_ok=True)

    archivos = [
        ("results/videos/video_final.mp4",  f"{destino}/videos/video_final.mp4"),
        ("results/logs/metricas.txt",        f"{destino}/logs/metricas.txt"),
        ("results/logs/xai_report.html",     f"{destino}/logs/xai_report.html"),
        ("results/srt/subtitulos.srt",       f"{destino}/srt/subtitulos.srt"),
        ("data/json/lip_tracking_data.json", f"{destino}/json/lip_tracking_data.json"),
    ]
    for origen, dest in archivos:
        if os.path.exists(origen):
            shutil.copy2(origen, dest)

    if os.path.exists("results/plots"):
        for png in os.listdir("results/plots"):
            if png.endswith(".png"):
                shutil.copy2(f"results/plots/{png}", f"{destino}/plots/{png}")

    print(f"\n  Corrida guardada en: {destino}")
    return destino


def main():
    parser = argparse.ArgumentParser(description="Pipeline completo de identificacion de hablantes")
    parser.add_argument("--video",             required=True,          help="Ruta al video de entrada")
    parser.add_argument("--token",             default=None,           help="Token de HuggingFace (hf_...)")
    parser.add_argument("--n_personas",        type=int, default=None, help="Numero esperado de personas (K-Means)")
    parser.add_argument("--modelo",            default="base",         help="Modelo Whisper: tiny, base, small, medium")
    parser.add_argument("--min_speakers",      type=int, default=None, help="Minimo de hablantes para pyannote")
    parser.add_argument("--max_speakers",      type=int, default=None, help="Maximo de hablantes para pyannote")
    parser.add_argument("--skip_lip_tracking", action="store_true",    help="Omitir lip tracking si ya tienes el .json")
    args = parser.parse_args()

    json_path = "data/json/lip_tracking_data.json"
    srt_path  = "results/srt/subtitulos.srt"

    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")

    # ── Paso 1: Lip tracking ─────────────────────────────────────
    if args.skip_lip_tracking:
        print("\n" + "="*55)
        print("  Paso 1/3: Lip tracking omitido")
        print("="*55)
    else:
        cmd1 = [PYTHON, os.path.join(src, "lip_tracking.py"), "--video", args.video]
        if args.n_personas:
            cmd1 += ["--n_personas", str(args.n_personas)]
        correr(cmd1, "Paso 1/3: Deteccion de labios y tracking facial")

    # ── Paso 2: Diarizacion + transcripcion ──────────────────────
    cmd2 = [
        PYTHON, os.path.join(src, "diarizacion.py"),
        "--video",  args.video,
        "--json",   json_path,
        "--token",  args.token,
        "--modelo", args.modelo,
    ]
    if args.min_speakers:
        cmd2 += ["--min_speakers", str(args.min_speakers)]
    if args.max_speakers:
        cmd2 += ["--max_speakers", str(args.max_speakers)]
    correr(cmd2, "Paso 2/3: Diarizacion de audio y transcripcion")

    # ── Paso 3: Render final + métricas + gráficas ───────────────
    cmd3 = [
        PYTHON, os.path.join(src, "render_final.py"),
        "--video", args.video,
        "--json",  json_path,
        "--srt",   srt_path,
    ]
    correr(cmd3, "Paso 3/4: Generacion del video final + metricas + graficas")

    # ── Paso 4: XAI ──────────────────────────────────────────────
    cmd4 = [
        PYTHON, os.path.join(src, "explainability.py"),
        "--json", json_path,
        "--srt",  srt_path,
    ]
    correr(cmd4, "Paso 4/4: Explainable AI — analisis de decisiones")

    # ── Guardar trazabilidad ──────────────────────────────────────
    destino = guardar_corrida(args.video)

    print("\n" + "="*55)
    print("  Pipeline completo.")
    print(f"  Video final  : results/videos/video_final.mp4")
    print(f"  Métricas     : results/logs/metricas.txt")
    print(f"  Reporte XAI  : results/logs/xai_report.html")
    print(f"  Gráficas     : results/plots/  (10 archivos PNG)")
    print(f"  Trazabilidad : {destino}")
    print("="*55)


if __name__ == "__main__":
    main()