import cv2
import json
import argparse
import subprocess
import numpy as np
import os
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

OUTPUT_VIDEO_NOAUDIO = "results/videos/video_final_tmp.mp4"
OUTPUT_VIDEO_FINAL   = "results/videos/video_final.mp4"
OUTPUT_METRICAS      = "results/logs/metricas.txt"
OUTPUT_PLOTS_DIR     = "results/plots"

COLORS = [(0,255,100),(0,180,255),(255,80,80),(200,0,255),(255,200,0),(200,0,200)]
COLORS_MPL = ["#00FF64", "#00B4FF", "#FF5050", "#C800FF", "#FFC800", "#C800C8"]

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


def obtener_umbral_persona(lip_data, pid):
    """Usa umbral adaptativo si existe, si no cae al global."""
    umbrales = lip_data.get("umbrales_adaptativos", {})
    return umbrales.get(str(pid), LAR_THRESHOLD)


def obtener_info_frame(lip_data, t):
    resultados = []
    for pid_str, serie in lip_data["lar_series"].items():
        pid     = int(pid_str)
        tiempos = np.array(serie["tiempos"])
        valores = np.array(serie["valores"])
        bboxes  = serie.get("bboxes", [])
        umbral  = obtener_umbral_persona(lip_data, pid)

        idx = np.searchsorted(tiempos, t, side="right") - 1
        if 0 <= idx < len(valores):
            lar      = valores[idx]
            hablando = lar > umbral
            bbox     = bboxes[idx] if idx < len(bboxes) else None
            resultados.append({"pid": pid, "lar": lar, "hablando": hablando,
                                "bbox": bbox, "umbral": umbral})
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
    serie = lip_data["lar_series"].get(str(pid))
    if serie is None:
        return np.array([]), np.array([])
    tiempos = np.array(serie["tiempos"])
    valores = np.array(serie["valores"])
    umbral  = obtener_umbral_persona(lip_data, pid)
    ticks   = np.arange(0, duracion, paso)
    señal   = np.zeros(len(ticks), dtype=int)
    lar_muestreado = np.zeros(len(ticks))
    for k, t in enumerate(ticks):
        idx = np.searchsorted(tiempos, t, side="right") - 1
        if 0 <= idx < len(valores):
            lar_muestreado[k] = valores[idx]
            señal[k] = int(valores[idx] > umbral)
    return señal, ticks, lar_muestreado, umbral


def muestrear_ground_truth(segmentos_srt, pid, duracion, paso=EVAL_STEP):
    ticks = np.arange(0, duracion, paso)
    gt    = np.zeros(len(ticks), dtype=int)
    for seg in segmentos_srt:
        if seg["pid"] == pid:
            mask = (ticks >= seg["inicio"]) & (ticks <= seg["fin"])
            gt[mask] = 1
    return gt


def calcular_metricas(gt, pred):
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


# ─────────────────────────────────────────────
#  GRÁFICAS
# ─────────────────────────────────────────────

def graficar_señales_temporales(lip_data, segmentos_gt, duracion, personas, json_path):
    """
    Gráfica 1: Señal LAR vs Ground Truth a lo largo del tiempo para cada persona.
    Muestra claramente dónde el modelo acierta, falla por exceso o por defecto.
    """
    os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)
    n = len(personas)
    fig, axes = plt.subplots(n, 1, figsize=(16, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    fig.suptitle("Señal LAR vs Ground Truth por persona", fontsize=14, fontweight="bold", y=1.01)

    for ax, pid_str in zip(axes, personas):
        pid = int(pid_str)
        color = COLORS_MPL[pid % len(COLORS_MPL)]

        resultado = muestrear_señal(lip_data, pid, duracion)
        if len(resultado[0]) == 0:
            continue
        pred, ticks, lar_vals, umbral = resultado
        gt = muestrear_ground_truth(segmentos_gt, pid, duracion)

        # Área de ground truth
        ax.fill_between(ticks, 0, gt.astype(float) * umbral * 2,
                         alpha=0.2, color="gray", label="Ground truth (SRT)")

        # Curva LAR
        ax.plot(ticks, lar_vals, color=color, linewidth=0.8, alpha=0.9, label="LAR")

        # Línea de umbral
        ax.axhline(umbral, color="red", linewidth=1.2, linestyle="--",
                   label=f"Umbral ({umbral:.4f})")

        # Resaltar TP, FP, FN con bandas de color
        tp_mask = (pred == 1) & (gt == 1)
        fp_mask = (pred == 1) & (gt == 0)
        fn_mask = (pred == 0) & (gt == 1)

        for mask, col, label in [
            (tp_mask, "#00C851", "TP"),
            (fp_mask, "#FF4444", "FP"),
            (fn_mask, "#FF8800", "FN"),
        ]:
            ax.fill_between(ticks, 0, umbral * 1.5,
                            where=mask, alpha=0.25, color=col, label=label)

        ax.set_ylabel("LAR", fontsize=10)
        ax.set_title(f"Persona {pid}", fontsize=11, color=color, fontweight="bold")
        ax.legend(loc="upper right", fontsize=7, ncol=6)
        ax.set_ylim(0, max(lar_vals.max() * 1.1, umbral * 2.5))
        ax.grid(axis="x", alpha=0.3)

    axes[-1].set_xlabel("Tiempo (s)", fontsize=10)
    plt.tight_layout()
    path = os.path.join(OUTPUT_PLOTS_DIR, "01_señales_temporales.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Guardada: {path}")


def graficar_barras_metricas(resultados):
    """
    Gráfica 2: Barras de Precisión, Recall, F1 y Accuracy por persona.
    Fácil de mostrar en la presentación.
    """
    os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)
    metricas_nombres = ["precision", "recall", "f1", "accuracy"]
    etiquetas        = ["Precisión", "Recall", "F1-score", "Accuracy"]
    personas         = sorted(resultados.keys())
    x = np.arange(len(metricas_nombres))
    ancho = 0.8 / len(personas)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    for i, pid in enumerate(personas):
        vals = [resultados[pid][m] for m in metricas_nombres]
        offset = (i - len(personas) / 2 + 0.5) * ancho
        bars = ax.bar(x + offset, vals, ancho * 0.9,
                      label=f"Persona {pid}",
                      color=COLORS_MPL[pid % len(COLORS_MPL)],
                      edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Promedio macro
    prec_avg = np.mean([resultados[p]["precision"] for p in personas])
    rec_avg  = np.mean([resultados[p]["recall"]    for p in personas])
    f1_avg   = np.mean([resultados[p]["f1"]        for p in personas])
    acc_avg  = np.mean([resultados[p]["accuracy"]  for p in personas])
    promedios = [prec_avg, rec_avg, f1_avg, acc_avg]
    for xi, v in zip(x, promedios):
        ax.axhline(v, xmin=(xi + 0.1) / len(x), xmax=(xi + 0.9) / len(x),
                   color="black", linewidth=2, linestyle=":")

    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Valor", fontsize=11)
    ax.set_title("Métricas de detección por persona", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUTPUT_PLOTS_DIR, "02_barras_metricas.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Guardada: {path}")


def graficar_confusion_matrices(resultados):
    """
    Gráfica 3: Matrices de confusión lado a lado para cada persona.
    """
    os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)
    personas = sorted(resultados.keys())
    n = len(personas)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    fig.suptitle("Matrices de Confusión", fontsize=13, fontweight="bold")

    for ax, pid in zip(axes, personas):
        m = resultados[pid]
        matriz = np.array([[m["tn"], m["fp"]],
                            [m["fn"], m["tp"]]])
        total = matriz.sum()
        im = ax.imshow(matriz, cmap="Blues", vmin=0, vmax=total)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred: Silencio", "Pred: Habla"], fontsize=9)
        ax.set_yticklabels(["Real: Silencio", "Real: Habla"], fontsize=9)
        ax.set_title(f"Persona {pid}\nF1={m['f1']:.3f}", fontsize=11,
                     color=COLORS_MPL[pid % len(COLORS_MPL)], fontweight="bold")

        etiquetas_cm = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                val = matriz[i, j]
                pct = 100 * val / total if total > 0 else 0
                color_txt = "white" if val > total * 0.4 else "black"
                ax.text(j, i, f"{etiquetas_cm[i][j]}\n{val}\n({pct:.1f}%)",
                        ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color_txt)

    plt.tight_layout()
    path = os.path.join(OUTPUT_PLOTS_DIR, "03_confusion_matrices.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Guardada: {path}")


def graficar_distribucion_lar(lip_data, personas):
    """
    Gráfica 4: Distribución (histograma + KDE) del LAR por persona,
    con línea del umbral global y adaptativo.
    """
    os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)
    n = len(personas)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    fig.suptitle("Distribución del LAR por persona", fontsize=13, fontweight="bold")
    umbrales_adaptativos = lip_data.get("umbrales_adaptativos", {})

    for ax, pid_str in zip(axes, personas):
        pid = int(pid_str)
        color = COLORS_MPL[pid % len(COLORS_MPL)]
        valores = np.array(lip_data["lar_series"][pid_str]["valores"])

        ax.hist(valores, bins=60, color=color, alpha=0.7, edgecolor="white",
                linewidth=0.3, density=True, label="Distribución LAR")

        # Umbral global
        ax.axvline(LAR_THRESHOLD, color="gray", linewidth=1.5,
                   linestyle="--", label=f"Umbral global ({LAR_THRESHOLD})")

        # Umbral adaptativo si existe
        if pid_str in umbrales_adaptativos:
            u_adapt = umbrales_adaptativos[pid_str]
            ax.axvline(u_adapt, color="red", linewidth=2,
                       linestyle="-", label=f"Umbral adaptativo ({u_adapt:.4f})")

        pct_habla_global = 100 * np.mean(valores > LAR_THRESHOLD)
        ax.set_title(f"Persona {pid}\n{pct_habla_global:.1f}% frames sobre umbral global",
                     fontsize=11, color=color, fontweight="bold")
        ax.set_xlabel("LAR", fontsize=10)
        ax.set_ylabel("Densidad", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUTPUT_PLOTS_DIR, "04_distribucion_lar.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Guardada: {path}")


def graficar_radar_metricas(resultados):
    """
    Gráfica 5: Radar/Spider chart comparando las 4 métricas entre personas.
    Visualmente impactante para la presentación.
    """
    os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)
    categorias = ["Precisión", "Recall", "F1-score", "Accuracy"]
    N = len(categorias)
    angulos = [n / float(N) * 2 * np.pi for n in range(N)]
    angulos += angulos[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    personas = sorted(resultados.keys())
    for pid in personas:
        m = resultados[pid]
        vals = [m["precision"], m["recall"], m["f1"], m["accuracy"]]
        vals += vals[:1]
        color = COLORS_MPL[pid % len(COLORS_MPL)]
        ax.plot(angulos, vals, "o-", linewidth=2, color=color,
                label=f"Persona {pid}", markersize=6)
        ax.fill(angulos, vals, alpha=0.15, color=color)

    # Promedio macro
    prec_avg = np.mean([resultados[p]["precision"] for p in personas])
    rec_avg  = np.mean([resultados[p]["recall"]    for p in personas])
    f1_avg   = np.mean([resultados[p]["f1"]        for p in personas])
    acc_avg  = np.mean([resultados[p]["accuracy"]  for p in personas])
    vals_avg = [prec_avg, rec_avg, f1_avg, acc_avg]
    vals_avg += vals_avg[:1]
    ax.plot(angulos, vals_avg, "s--", linewidth=2, color="white",
            label="Promedio macro", markersize=5, alpha=0.7)

    ax.set_xticks(angulos[:-1])
    ax.set_xticklabels(categorias, fontsize=12, color="white")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"],
                       fontsize=7, color="lightgray")
    ax.grid(color="gray", alpha=0.3)
    ax.spines["polar"].set_color("gray")
    ax.set_title("Comparación de métricas por persona",
                 fontsize=13, fontweight="bold", color="white", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1),
              fontsize=10, facecolor="#1a1a2e", labelcolor="white")

    plt.tight_layout()
    path = os.path.join(OUTPUT_PLOTS_DIR, "05_radar_metricas.png")
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"  Guardada: {path}")


def graficar_actividad_hablante(lip_data, personas, duracion):
    """
    Gráfica 6: Timeline de actividad de habla tipo 'gantt' por persona.
    Muestra quién habla cuándo a lo largo del video.
    """
    os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)
    umbrales = lip_data.get("umbrales_adaptativos", {})

    fig, ax = plt.subplots(figsize=(16, max(3, len(personas) * 1.5)))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    for i, pid_str in enumerate(personas):
        pid    = int(pid_str)
        color  = COLORS_MPL[pid % len(COLORS_MPL)]
        umbral = umbrales.get(pid_str, LAR_THRESHOLD)
        tiempos = np.array(lip_data["lar_series"][pid_str]["tiempos"])
        valores = np.array(lip_data["lar_series"][pid_str]["valores"])

        # Construir bloques de habla contiguos
        hablando = valores > umbral
        en_bloque = False
        t_inicio  = 0.0
        for j, (t, h) in enumerate(zip(tiempos, hablando)):
            if h and not en_bloque:
                t_inicio  = t
                en_bloque = True
            elif not h and en_bloque:
                ax.barh(i, t - t_inicio, left=t_inicio, height=0.6,
                        color=color, alpha=0.8, edgecolor="none")
                en_bloque = False
        if en_bloque:
            ax.barh(i, tiempos[-1] - t_inicio, left=t_inicio, height=0.6,
                    color=color, alpha=0.8, edgecolor="none")

    ax.set_yticks(range(len(personas)))
    ax.set_yticklabels([f"Persona {p}" for p in personas], fontsize=11)
    ax.set_xlabel("Tiempo (s)", fontsize=11)
    ax.set_xlim(0, duracion)
    ax.set_title("Timeline de actividad de habla por persona", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    # Leyenda con tiempo total hablado
    handles = []
    for pid_str in personas:
        pid    = int(pid_str)
        umbral = umbrales.get(pid_str, LAR_THRESHOLD)
        valores = np.array(lip_data["lar_series"][pid_str]["valores"])
        tiempos = np.array(lip_data["lar_series"][pid_str]["tiempos"])
        dt      = np.diff(tiempos, prepend=tiempos[0])
        t_habla = float(np.sum(dt[valores > umbral]))
        pct     = 100 * t_habla / duracion
        patch = mpatches.Patch(color=COLORS_MPL[pid % len(COLORS_MPL)],
                               label=f"P{pid}: {t_habla:.1f}s ({pct:.1f}%)")
        handles.append(patch)
    ax.legend(handles=handles, loc="upper right", fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_PLOTS_DIR, "06_timeline_actividad.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Guardada: {path}")

def graficar_resumen_macro(resultados):
    os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)
    personas  = sorted(resultados.keys())

    prec_avg = np.mean([resultados[p]["precision"] for p in personas])
    rec_avg  = np.mean([resultados[p]["recall"]    for p in personas])
    f1_avg   = np.mean([resultados[p]["f1"]        for p in personas])
    acc_avg  = np.mean([resultados[p]["accuracy"]  for p in personas])

    etiquetas = ["Precisión", "Recall", "F1-score", "Accuracy"]
    valores   = [prec_avg, rec_avg, f1_avg, acc_avg]
    colores   = ["#00B4FF", "#00FF64", "#FFC800", "#FF5050"]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    bars = ax.bar(etiquetas, valores, color=colores, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, valores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Valor", fontsize=11)
    ax.set_title("Rendimiento general del sistema (Promedio Macro)", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUTPUT_PLOTS_DIR, "07_resumen_macro.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Guardada: {path}")


def evaluar_metricas(json_path, srt_path):
    print("\n" + "=" * 60)
    print("  EVALUACIÓN DE MÉTRICAS")
    print("=" * 60)

    with open(json_path) as f:
        lip_data = json.load(f)

    subtitulos   = leer_srt(srt_path)
    segmentos_gt = extraer_personas_srt(subtitulos)
    duracion     = lip_data.get("duracion_seg", 0)
    personas     = sorted(lip_data["lar_series"].keys(), key=int)

    if not segmentos_gt:
        print("\n[!] No se encontraron etiquetas 'Persona X:' en el SRT.")
        print("    Asegurate de que diarizacion.py generó el archivo con ese formato.")
        return

    personas_gt = sorted(set(s["pid"] for s in segmentos_gt))
    umbrales    = lip_data.get("umbrales_adaptativos", {})

    print(f"\nDuración del video : {duracion:.1f}s")
    print(f"Paso de muestreo   : {EVAL_STEP}s  ({int(duracion/EVAL_STEP)} muestras)")
    print(f"Personas en SRT    : {personas_gt}")
    print(f"Personas en JSON   : {[int(p) for p in personas]}")
    if umbrales:
        print(f"Umbrales adaptativos: { {k: v for k, v in umbrales.items()} }")

    resultados = {}

    for pid_str in personas:
        pid = int(pid_str)

        resultado = muestrear_señal(lip_data, pid, duracion)
        if len(resultado[0]) == 0:
            continue
        pred, ticks, _, umbral_usado = resultado

        gt  = muestrear_ground_truth(segmentos_gt, pid, duracion)
        met = calcular_metricas(gt, pred)
        resultados[pid] = met

        print(f"\n  Persona {pid}  [umbral={umbral_usado:.4f}]")
        print(f"    Ground truth (SRT) habla : {np.sum(gt) * EVAL_STEP:.1f}s")
        print(f"    Detección LAR habla      : {np.sum(pred) * EVAL_STEP:.1f}s")
        print(f"    TP={met['tp']}  FP={met['fp']}  FN={met['fn']}  TN={met['tn']}")
        print(f"    Precisión  : {met['precision']:.3f}")
        print(f"    Recall     : {met['recall']:.3f}")
        print(f"    F1-score   : {met['f1']:.3f}")
        print(f"    Accuracy   : {met['accuracy']:.3f}")

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

    # ── Guardar reporte de texto ──
    os.makedirs(os.path.dirname(OUTPUT_METRICAS), exist_ok=True)
    with open(OUTPUT_METRICAS, "w", encoding="utf-8") as f:
        f.write("REPORTE DE MÉTRICAS — Lip Tracking vs Ground Truth (SRT)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Duración        : {duracion:.1f}s\n")
        f.write(f"Paso muestreo   : {EVAL_STEP}s\n")
        f.write(f"Umbral global   : {LAR_THRESHOLD}\n")
        if umbrales:
            f.write(f"Umbrales adapt. : {umbrales}\n")
        f.write("\n")
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

    # ── Generar todas las gráficas ──
    if resultados:
        print(f"\nGenerando gráficas en: {OUTPUT_PLOTS_DIR}/")
        graficar_señales_temporales(lip_data, segmentos_gt, duracion, personas, json_path)
        graficar_barras_metricas(resultados)
        graficar_confusion_matrices(resultados)
        graficar_distribucion_lar(lip_data, personas)
        graficar_radar_metricas(resultados)
        graficar_actividad_hablante(lip_data, personas, duracion)
        graficar_resumen_macro(resultados)
        print(f"  ✓ 7 gráficas guardadas en {OUTPUT_PLOTS_DIR}/")

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
                        help="Solo calcular métricas y gráficas, sin recomponer el video")
    args = parser.parse_args()

    if args.metricas:
        evaluar_metricas(args.json, args.srt)
    else:
        procesar(args.video, args.json, args.srt)
        agregar_audio(args.video)
        evaluar_metricas(args.json, args.srt)


if __name__ == "__main__":
    main()