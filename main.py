import argparse
import subprocess
import sys
import os

PYTHON = sys.executable

def correr(cmd, descripcion):
    print(f"\n{'='*55}")
    print(f"  {descripcion}")
    print(f"{'='*55}")
    resultado = subprocess.run(cmd, check=True, env={**os.environ})
    if resultado.returncode != 0:
        print(f"Error en: {descripcion}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Pipeline completo de identificacion de hablantes")
    parser.add_argument("--video",        required=True,          help="Ruta al video de entrada")
    parser.add_argument("--token",        required=True,          help="Token de HuggingFace (hf_...)")
    parser.add_argument("--n_personas",   type=int, default=None, help="Numero esperado de personas (K-Means)")
    parser.add_argument("--modelo",       default="base",         help="Modelo Whisper: tiny, base, small, medium")
    # BUG FIX: estos argumentos existían en diarizacion.py pero no se pasaban
    parser.add_argument("--min_speakers", type=int, default=None, help="Minimo de hablantes para pyannote")
    parser.add_argument("--max_speakers", type=int, default=None, help="Maximo de hablantes para pyannote")
    args = parser.parse_args()

    json_path = "data/json/lip_tracking_data.json"
    srt_path  = "results/srt/subtitulos.srt"

    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")

    # ── Paso 1: Lip tracking ─────────────────────────────────────
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
    # BUG FIX: antes se referenciaban sin haber definido los args
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
    correr(cmd3, "Paso 3/3: Generacion del video final + metricas + graficas")

    print("\n" + "="*55)
    print("  Pipeline completo.")
    print(f"  Video final  : results/videos/video_final.mp4")
    print(f"  Métricas     : results/logs/metricas.txt")
    print(f"  Gráficas     : results/plots/  (6 archivos PNG)")
    print("="*55)

if __name__ == "__main__":
    main()