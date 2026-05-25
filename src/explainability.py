"""
XAI (Explainable AI) module for PROY-VCP.
Generates 3 explanatory graphs and a self-contained HTML report.

Graphs:
  08_xai_fusion_scores.png   — WHY each audio speaker was assigned to each visual person
  09_xai_lar_evidence.png    — HOW confident the LAR signal is at each moment
  10_xai_embeddings_tsne.png — WHERE face identities sit in ArcFace embedding space
"""

import os
import json
import base64
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

OUTPUT_PLOTS_DIR = "results/plots"
OUTPUT_XAI_HTML  = "results/logs/xai_report.html"
LAR_THRESHOLD    = 0.03

COLORS_MPL = ["#00FF64", "#00B4FF", "#FF5050", "#C800FF", "#FFC800", "#C800C8"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _srt_a_seg(s):
    h, m, rest = s.split(":")
    seg, ms = rest.split(",")
    return int(h)*3600 + int(m)*60 + int(seg) + int(ms)/1000


def leer_srt_personas(srt_path):
    segmentos = []
    if not srt_path or not os.path.exists(srt_path):
        return segmentos
    with open(srt_path, encoding="utf-8") as f:
        contenido = f.read().strip()
    for bloque in contenido.split("\n\n"):
        lineas = bloque.strip().split("\n")
        if len(lineas) < 3:
            continue
        partes = lineas[1].split(" --> ")
        if len(partes) < 2:
            continue
        inicio = _srt_a_seg(partes[0].strip())
        fin    = _srt_a_seg(partes[1].strip())
        texto  = " ".join(lineas[2:]).strip()
        if texto.startswith("Persona "):
            try:
                pid = int(texto.split(":")[0].replace("Persona", "").strip())
                segmentos.append({"pid": pid, "inicio": inicio, "fin": fin})
            except ValueError:
                pass
    return segmentos


def _img_to_b64(path):
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return None


# ── GRAPH 8: Fusion Score Heatmap + Bar Chart ─────────────────────────────────

def graficar_fusion_scores(lip_data):
    """
    Explains WHY each audio speaker was mapped to each visual person.
    Shows the full score matrix so the user can see why the winning assignment was chosen.
    """
    os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)

    xai_scores = lip_data.get("xai_fusion_scores")
    xai_mapeo  = lip_data.get("xai_mapeo", {})

    if not xai_scores:
        print("  [XAI] xai_fusion_scores not in JSON — re-run diarizacion.py to populate it.")
        return None

    speakers = sorted(xai_scores.keys())
    personas = sorted({int(p) for d in xai_scores.values() for p in d})

    matrix = np.zeros((len(speakers), len(personas)))
    for i, sp in enumerate(speakers):
        for j, pid in enumerate(personas):
            matrix[i, j] = xai_scores[sp].get(str(pid), 0.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(4, len(speakers) * 1.8 + 2)))
    fig.suptitle("XAI — Fusión Audio↔Visual: ¿por qué esta asignación?",
                 fontsize=14, fontweight="bold")

    # Left: heatmap
    cmap = LinearSegmentedColormap.from_list("xai", ["#1a1a2e", "#0f3460", "#00B4FF", "#00FF64"])
    im = ax1.imshow(matrix, cmap=cmap, aspect="auto", vmin=0)
    plt.colorbar(im, ax=ax1, label="Score de fusión normalizado")
    ax1.set_xticks(range(len(personas)))
    ax1.set_xticklabels([f"Persona {p}" for p in personas], fontsize=11)
    ax1.set_yticks(range(len(speakers)))
    ax1.set_yticklabels(speakers, fontsize=11)
    ax1.set_xlabel("Persona Visual (lip tracking)", fontsize=11)
    ax1.set_ylabel("Speaker Audio (pyannote)", fontsize=11)
    ax1.set_title("Matriz de scores\n(mayor = más evidencia LAR de que habla esa persona)", fontsize=10, fontweight="bold")

    for i, sp in enumerate(speakers):
        pid_ganador = xai_mapeo.get(sp, -1)
        for j, pid in enumerate(personas):
            val = matrix[i, j]
            color_txt = "white" if val < matrix.max() * 0.5 else "black"
            marker = " ★" if pid == pid_ganador else ""
            ax1.text(j, i, f"{val:.3f}{marker}", ha="center", va="center",
                     fontsize=11, fontweight="bold", color=color_txt)

    # Right: bar chart
    x     = np.arange(len(personas))
    width = 0.8 / max(len(speakers), 1)
    for i, sp in enumerate(speakers):
        scores_sp = [xai_scores[sp].get(str(pid), 0.0) for pid in personas]
        offset    = (i - len(speakers)/2 + 0.5) * width
        pid_gan   = xai_mapeo.get(sp, -1)
        bars = ax2.bar(x + offset, scores_sp, width * 0.9,
                       label=sp, color=COLORS_MPL[i % len(COLORS_MPL)],
                       edgecolor="white", linewidth=0.5)
        for bar, pid in zip(bars, personas):
            if pid == pid_gan:
                bar.set_edgecolor("yellow")
                bar.set_linewidth(3)
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.0005,
                     f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Persona {p}" for p in personas], fontsize=11)
    ax2.set_ylabel("Score de fusión", fontsize=11)
    ax2.set_title("Scores por speaker\n(borde amarillo ★ = asignado)", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=9, title="Speaker audio")
    ax2.grid(axis="y", alpha=0.3)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.text(0.5, -0.03,
             "Score = Σ_seg [ frac_LAR_activa × (1 + intensidad) × duración ]  ÷  duración_total_speaker",
             ha="center", fontsize=9, style="italic", color="#555555")

    plt.tight_layout()
    path = os.path.join(OUTPUT_PLOTS_DIR, "08_xai_fusion_scores.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Guardada: {path}")
    return path


# ── GRAPH 9: LAR Evidence + Landmark Sensitivity ─────────────────────────────

def graficar_lar_evidence(lip_data, segmentos_gt):
    """
    Per-person: shows confidence zones around the LAR threshold,
    annotates each SRT segment with its confidence percentage,
    and adds an analytical landmark sensitivity subplot.
    """
    os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)
    personas = sorted(lip_data["lar_series"].keys(), key=int)
    duracion = lip_data.get("duracion_seg", 0)
    umbrales = lip_data.get("umbrales_adaptativos", {})
    n = len(personas)

    fig = plt.figure(figsize=(16, 3.5 * n + 4))
    fig.suptitle("XAI — Evidencia LAR: confianza en detección de habla",
                 fontsize=14, fontweight="bold", y=1.01)
    gs = GridSpec(n + 1, 1, figure=fig, hspace=0.55,
                  height_ratios=[3.0] * n + [2.5])

    ax_list = []
    for idx, pid_str in enumerate(personas):
        pid    = int(pid_str)
        color  = COLORS_MPL[pid % len(COLORS_MPL)]
        umbral = float(umbrales.get(pid_str, LAR_THRESHOLD))

        tiempos = np.array(lip_data["lar_series"][pid_str]["tiempos"])
        valores = np.array(lip_data["lar_series"][pid_str]["valores"])

        ax = fig.add_subplot(gs[idx])
        ax_list.append(ax)

        # Confidence zones (filled backgrounds)
        ax.fill_between(tiempos, umbral * 1.2, float(valores.max()) * 1.15,
                        alpha=0.06, color="green")
        ax.fill_between(tiempos, umbral * 0.8, umbral * 1.2,
                        alpha=0.12, color="orange",
                        label="Zona borderline ±20% umbral")

        # LAR signal + threshold
        ax.plot(tiempos, valores, color=color, linewidth=0.7, alpha=0.85, label="LAR")
        ax.axhline(umbral, color="red", linewidth=1.5, linestyle="--",
                   label=f"Umbral adaptativo = {umbral:.4f}")

        # Per-segment confidence annotations from SRT
        segs_pid = [s for s in segmentos_gt if s["pid"] == pid]
        for seg in segs_pid:
            t0, t1 = seg["inicio"], seg["fin"]
            ax.axvspan(t0, t1, alpha=0.07, color="gray")
            mask = (tiempos >= t0) & (tiempos <= t1)
            if np.sum(mask) > 0:
                vals_seg   = valores[mask]
                conf_alta  = float(np.mean(vals_seg > umbral * 1.2))
                conf_borde = float(np.mean(
                    (vals_seg > umbral * 0.8) & (vals_seg <= umbral * 1.2)
                ))
                col = ("#00C851" if conf_alta > 0.5
                       else "#FF8800" if conf_borde > 0.35
                       else "#FF4444")
                mid_t = (t0 + t1) / 2
                y_pos = umbral * 1.9
                ax.text(mid_t, y_pos, f"{conf_alta:.0%}",
                        ha="center", va="bottom", fontsize=7,
                        color=col, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  alpha=0.75, edgecolor=col, linewidth=1))

        # Global confidence breakdown in title
        pct_alta  = float(np.mean(valores > umbral * 1.2))
        pct_borde = float(np.mean((valores > umbral * 0.8) & (valores <= umbral * 1.2)))
        pct_sil   = float(np.mean(valores < umbral * 0.8))
        ax.set_title(
            f"Persona {pid}  |  Alta conf. habla: {pct_alta:.1%}  "
            f"Borderline: {pct_borde:.1%}  Alta conf. silencio: {pct_sil:.1%}",
            fontsize=9, color=color, fontweight="bold"
        )
        ax.set_ylabel("LAR", fontsize=9)
        ax.legend(loc="upper right", fontsize=7, ncol=3)
        ax.set_xlim(0, duracion)
        ax.set_ylim(0, max(float(valores.max()) * 1.2, umbral * 3.5))
        ax.grid(axis="x", alpha=0.3)
        if idx < n - 1:
            ax.set_xticklabels([])

    if ax_list:
        ax_list[-1].set_xlabel("Tiempo (s)", fontsize=10)

    # Bottom subplot: landmark sensitivity
    ax_sens = fig.add_subplot(gs[n])
    _subplot_landmark_sensitivity(ax_sens, lip_data, personas, umbrales)

    path = os.path.join(OUTPUT_PLOTS_DIR, "09_xai_lar_evidence.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Guardada: {path}")
    return path


def _subplot_landmark_sensitivity(ax, lip_data, personas, umbrales):
    """
    Analytical sensitivity of LAR = v/h to each of the 4 lip landmarks.
    dLAR/dv =  1/h  (positive: opening mouth raises LAR)
    dLAR/dh = -v/h² (negative: wider mouth lowers LAR)
    Evaluated at the median and 90th-percentile LAR values per person.
    h_est = 0.25 (typical MediaPipe normalized lip width).
    """
    H = 0.25  # typical normalized horizontal lip gap

    landmark_labels = [
        "LIP_TOP (13)\n+vertical gap",
        "LIP_BOTTOM (14)\n+vertical gap",
        "LIP_LEFT (78)\n+horizontal gap",
        "LIP_RIGHT (308)\n+horizontal gap",
    ]
    x     = np.arange(len(landmark_labels))
    width = 0.8 / max(len(personas), 1)

    for idx, pid_str in enumerate(personas):
        pid    = int(pid_str)
        color  = COLORS_MPL[pid % len(COLORS_MPL)]
        vals   = np.array(lip_data["lar_series"][pid_str]["valores"])
        pos    = vals[vals > 0]
        if len(pos) == 0:
            continue

        lar_med = float(np.median(pos))
        lar_p90 = float(np.percentile(pos, 90))

        sv     = 1.0 / H                      # same for both vertical landmarks
        sh_med = -(lar_med * H) / H**2        # = -lar_med / H
        sh_p90 = -(lar_p90 * H) / H**2        # = -lar_p90 / H

        sens_med = [sv, sv, sh_med, sh_med]
        offset   = (idx - len(personas)/2 + 0.5) * width

        bars = ax.bar(x + offset, sens_med, width * 0.9,
                      label=f"P{pid} (med LAR={lar_med:.3f})",
                      color=color, alpha=0.85, edgecolor="white", linewidth=0.5)

        # Diamond markers at p90 for horizontal landmarks (they differ between persons)
        for j in (2, 3):
            bar = bars[j]
            ax.plot(bar.get_x() + bar.get_width()/2, sh_p90,
                    marker="D", color=color, markersize=5,
                    markeredgecolor="white", markeredgewidth=0.8,
                    label=f"P{pid} (p90 LAR={lar_p90:.3f})" if j == 2 else "")

    ax.axhline(0, color="gray", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(landmark_labels, fontsize=9)
    ax.set_ylabel("∂LAR / ∂coordenada", fontsize=10)
    ax.set_title(
        "Sensibilidad analítica del LAR a cada landmark labial\n"
        "LAR = gap_vertical / gap_horizontal  →  barras = derivada parcial evaluada en LAR mediano",
        fontsize=10, fontweight="bold"
    )
    ax.legend(fontsize=8, ncol=max(len(personas) * 2, 1))
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.5, -0.22,
        "Positivo: ese landmark empuja LAR↑ (favorece detección de habla)  |  "
        "Negativo: empuja LAR↓ (favorece silencio)  |  ◆ = valor en percentil 90",
        transform=ax.transAxes, ha="center", fontsize=8, color="#555555", style="italic"
    )


# ── GRAPH 10: ArcFace Embedding t-SNE ────────────────────────────────────────

def graficar_embeddings_tsne(lip_data):
    """
    t-SNE of ArcFace face embeddings saved during lip tracking.
    Each point = one face track; color = final person ID.
    Also shows a cosine similarity matrix between person centroids.
    """
    os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)

    xai_embs = lip_data.get("xai_embeddings")
    if not xai_embs:
        print("  [XAI] No xai_embeddings in JSON — needs DeepFace/ArcFace active during lip_tracking.")
        return None
    if len(xai_embs) < 3:
        print("  [XAI] Too few embeddings for t-SNE (need ≥ 3).")
        return None

    from sklearn.manifold import TSNE
    from sklearn.preprocessing import normalize

    # Subsample to max 200 for speed
    if len(xai_embs) > 200:
        import random
        random.seed(42)
        xai_embs = random.sample(xai_embs, 200)

    personas_list = [e["persona"]  for e in xai_embs]
    embs          = np.array([e["embedding"] for e in xai_embs], dtype=np.float32)
    embs_norm     = normalize(embs)

    perplexity = min(30, max(2, len(embs) - 1))
    tsne       = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                      max_iter=1000, init="pca")
    coords     = tsne.fit_transform(embs_norm)

    personas_unicas = sorted(set(personas_list))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("XAI — Espacio de embeddings ArcFace (t-SNE 2D)",
                 fontsize=14, fontweight="bold")

    # t-SNE scatter
    for pid in personas_unicas:
        mask = np.array([p == pid for p in personas_list])
        pts  = coords[mask]
        ax1.scatter(pts[:, 0], pts[:, 1],
                    label=f"Persona {pid}",
                    color=COLORS_MPL[pid % len(COLORS_MPL)],
                    s=70, alpha=0.75, edgecolors="white", linewidths=0.4)
        # Centroid star
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        ax1.scatter(cx, cy, marker="*", s=350,
                    color=COLORS_MPL[pid % len(COLORS_MPL)],
                    edgecolors="white", linewidths=1.5, zorder=5)
        ax1.annotate(f"P{pid}", (cx, cy), xytext=(5, 5),
                     textcoords="offset points", fontsize=10, fontweight="bold",
                     color=COLORS_MPL[pid % len(COLORS_MPL)])

    ax1.set_title("Clusters de identidad facial\n(★ = centroide de cada persona)", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.set_xlabel("t-SNE dim 1", fontsize=10)
    ax1.set_ylabel("t-SNE dim 2", fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.spines[["top", "right"]].set_visible(False)

    # Cosine similarity matrix between person centroids
    avg_embs = {}
    for pid in personas_unicas:
        mask = np.array([p == pid for p in personas_list])
        avg_embs[pid] = normalize(embs_norm[mask].mean(axis=0, keepdims=True))[0]

    n_p      = len(personas_unicas)
    sim_mat  = np.zeros((n_p, n_p))
    for i, pi in enumerate(personas_unicas):
        for j, pj in enumerate(personas_unicas):
            sim_mat[i, j] = float(np.dot(avg_embs[pi], avg_embs[pj]))

    im = ax2.imshow(sim_mat, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax2, label="Similitud coseno")
    ax2.set_xticks(range(n_p))
    ax2.set_yticks(range(n_p))
    labels = [f"P{p}" for p in personas_unicas]
    ax2.set_xticklabels(labels, fontsize=12)
    ax2.set_yticklabels(labels, fontsize=12)
    ax2.set_title("Similitud coseno entre personas\n(centroide ArcFace)", fontsize=11, fontweight="bold")

    for i in range(n_p):
        for j in range(n_p):
            col_txt = "black" if abs(sim_mat[i, j]) < 0.5 else "white"
            ax2.text(j, i, f"{sim_mat[i, j]:.3f}",
                     ha="center", va="center", fontsize=13,
                     fontweight="bold", color=col_txt)

    plt.tight_layout()
    path = os.path.join(OUTPUT_PLOTS_DIR, "10_xai_embeddings_tsne.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Guardada: {path}")
    return path


# ── HTML REPORT ───────────────────────────────────────────────────────────────

def generar_html(lip_data, paths):
    os.makedirs(os.path.dirname(OUTPUT_XAI_HTML), exist_ok=True)

    xai_mapeo  = lip_data.get("xai_mapeo", {})
    xai_scores = lip_data.get("xai_fusion_scores", {})
    umbrales   = lip_data.get("umbrales_adaptativos", {})
    n_personas = lip_data.get("n_personas", 0)
    duracion   = lip_data.get("duracion_seg", 0)

    # ── Fusion score table ──
    if xai_scores:
        speakers = sorted(xai_scores.keys())
        personas = sorted({int(p) for d in xai_scores.values() for p in d})
        thead = ("<tr><th>Speaker Audio</th>"
                 + "".join("<th>Persona " + str(p) + "</th>" for p in personas)
                 + "<th>Asignado a</th></tr>")
        tbody = ""
        for sp in speakers:
            pid_gan = xai_mapeo.get(sp, -1)
            row = "<tr><td><b>" + sp + "</b></td>"
            for pid in personas:
                s    = xai_scores[sp].get(str(pid), 0.0)
                win  = pid == pid_gan
                sty  = "background:#00C851;color:#000;font-weight:bold;" if win else ""
                mark = " &#9733;" if win else ""
                row += "<td style='" + sty + "'>" + f"{s:.4f}" + mark + "</td>"
            row += "<td><b>Persona " + str(pid_gan) + "</b></td></tr>"
            tbody += row
        tabla_fusion = (
            "<table border='1' style='border-collapse:collapse;width:100%'>"
            "<thead style='background:#1a1a2e;color:white'>" + thead + "</thead>"
            "<tbody>" + tbody + "</tbody></table>"
        )
    else:
        tabla_fusion = "<p><i>No hay datos de fusion (ejecutar diarizacion.py de nuevo).</i></p>"

    # ── Threshold table ──
    if umbrales:
        u_rows = "".join(
            "<tr><td>Persona " + k + "</td><td>" + f"{v:.4f}" + "</td>"
            "<td>" + f"{v/LAR_THRESHOLD*100:.0f}" + "% del umbral global</td></tr>"
            for k, v in sorted(umbrales.items())
        )
        tabla_umbrales = (
            "<table border='1' style='border-collapse:collapse'>"
            "<thead style='background:#1a1a2e;color:white'>"
            "<tr><th>Persona</th><th>Umbral adaptativo</th><th>vs. global (0.03)</th></tr>"
            "</thead><tbody>" + u_rows + "</tbody></table>"
        )
    else:
        tabla_umbrales = "<p><i>Usando umbral global LAR = 0.03</i></p>"

    # ── Image helpers ──
    def img_tag(p, alt):
        b64 = _img_to_b64(p)
        if b64:
            return ("<img src='data:image/png;base64," + b64 + "' alt='" + alt + "' "
                    "style='max-width:100%;border-radius:6px;"
                    "box-shadow:0 2px 10px rgba(0,0,0,0.18)'>")
        return "<p><i>" + alt + " &mdash; no disponible</i></p>"

    def card(val, lbl):
        return ("<div class='card'><div class='val'>" + str(val)
                + "</div><div class='lbl'>" + lbl + "</div></div>")

    def section(title, content):
        return (
            "<section style='margin-bottom:36px;padding:22px;"
            "background:#f8f9fa;border-radius:10px;border-left:5px solid #00B4FF'>"
            "<h2 style='color:#1a1a2e;margin-top:0'>" + title + "</h2>"
            + content + "</section>"
        )

    def formula(txt):
        return "<div class='formula'>" + txt + "</div>"

    def note(txt):
        return "<p class='note'>" + txt + "</p>"

    # ── Section 1: Fusion ──
    embed_img = (img_tag(paths.get("embeddings"), "t-SNE embeddings ArcFace")
                 if paths.get("embeddings")
                 else "<p><i>No se generaron embeddings en esta corrida. "
                      "Requiere DeepFace/ArcFace activo durante lip_tracking.</i></p>")

    sec1 = section(
        "1 &middot; Fusion Audio&#8596;Visual &mdash; &iquest;Por que esta asignacion?",
        (formula(
            "score(speaker_audio, persona_visual) =<br>"
            "&nbsp;&nbsp;suma_segmentos [ frac_frames_LAR_activa x (1 + intensidad_LAR) x duracion ]<br>"
            "&nbsp;&nbsp;/ duracion_total_speaker"
         )
         + note(
            "<b>frac_frames_LAR_activa</b>: % de frames donde LAR &gt; umbral adaptativo de esa persona.<br>"
            "<b>intensidad_LAR_activa</b>: valor medio del LAR en frames activos (mayor apertura = mayor peso).<br>"
            "<b>Asignacion greedy</b>: se elige el par con mayor score, luego el segundo, etc."
         )
         + "<h3>Tabla de scores (&#9733; = asignacion final)</h3>"
         + tabla_fusion + "<br>"
         + img_tag(paths.get("fusion"), "Heatmap y barras de scores de fusion"))
    )

    # ── Section 2: LAR Evidence ──
    sec2 = section(
        "2 &middot; Evidencia LAR &mdash; &iquest;Cuando esta seguro el sistema?",
        (formula("umbral_adaptativo(persona) = max(0.05,  percentil_70(LAR_persona) x 0.4")
         + note(
            "Cada persona tiene un umbral calibrado a su rango natural de movimiento labial.<br>"
            "Umbral alto &rarr; esa persona mueve mucho los labios (necesita mas apertura para detectarla).<br>"
            "Umbral bajo &rarr; esa persona tiene movimientos mas sutiles."
         )
         + tabla_umbrales
         + "<h3>Zonas de confianza en la senal LAR</h3>"
         + note(
            "<span style='color:#00C851;font-weight:bold'>Verde</span>: LAR &gt; 120% umbral &rarr; alta confianza habla.<br>"
            "<span style='color:#FF8800;font-weight:bold'>Naranja</span>: LAR entre 80% y 120% umbral &rarr; borderline.<br>"
            "El % en cada segmento SRT = fraccion de frames en zona verde."
         )
         + formula(
            "LAR = gap_vertical(LIP_TOP, LIP_BOTTOM) / gap_horizontal(LIP_LEFT, LIP_RIGHT)<br><br>"
            "dLAR/d(gap_vertical)  = +1/h  &rarr;  abrir la boca sube LAR (detecta habla)<br>"
            "dLAR/d(gap_horizontal) = -v/h2 &rarr;  ensanchar baja LAR"
         )
         + img_tag(paths.get("lar_evidence"), "Evidencia LAR y sensibilidad de landmarks"))
    )

    # ── Section 3: Embeddings ──
    sec3 = section(
        "3 &middot; Espacio de Embeddings &mdash; &iquest;Como distingue las identidades?",
        ("<h3>Modelo: ArcFace via DeepFace (512 dimensiones)</h3>"
         + note(
            "ArcFace proyecta cada cara a un vector de 512 dims entrenado para "
            "maximizar la separacion entre identidades.<br>"
            "El t-SNE reduce esas 512 dims a 2D; los clusters deben corresponder con las personas.<br>"
            "El heatmap de similitud coseno muestra que tan bien el sistema separa las identidades."
         )
         + formula("Re-ID: si similitud_coseno(embedding_nuevo, embedding_banco) &ge; 0.65 &rarr; misma persona")
         + embed_img)
    )

    # ── Meta cards ──
    meta = (
        "<div class='meta'>"
        + card(n_personas, "Personas detectadas")
        + card(f"{duracion:.0f}s", "Duracion del video")
        + card(len(xai_mapeo), "Speakers audio (pyannote)")
        + card(len(umbrales), "Umbrales adaptativos")
        + "</div>"
    )

    # ── CSS ──
    css = (
        "body{font-family:'Segoe UI',Arial,sans-serif;margin:0;padding:24px 44px;"
        "background:#fff;color:#333;max-width:1400px;margin-left:auto;margin-right:auto}"
        "h1{color:#1a1a2e;border-bottom:3px solid #00B4FF;padding-bottom:10px}"
        "h2{color:#0f3460;margin-top:0}h3{color:#16213e}"
        "table{font-size:14px;margin:10px 0}td,th{padding:8px 14px;text-align:center}"
        ".meta{display:flex;gap:18px;flex-wrap:wrap;margin:20px 0}"
        ".card{background:#1a1a2e;color:#fff;padding:16px 22px;border-radius:10px;"
        "min-width:140px;text-align:center}"
        ".card .val{font-size:2.2em;font-weight:bold;color:#00B4FF;line-height:1.1}"
        ".card .lbl{font-size:.8em;color:#aaa;margin-top:4px}"
        ".formula{background:#f0f4ff;border:1px solid #c0d0ff;padding:12px 18px;"
        "border-radius:6px;font-family:monospace;font-size:.95em;margin:10px 0}"
        ".note{font-size:.85em;color:#666;margin:6px 0}"
        "footer{margin-top:40px;padding:18px;background:#1a1a2e;color:#aaa;"
        "border-radius:10px;font-size:.85em}"
    )

    html = (
        "<!DOCTYPE html><html lang='es'><head><meta charset='UTF-8'>"
        "<title>XAI Report - PROY-VCP</title>"
        "<style>" + css + "</style></head><body>"
        "<h1>Reporte XAI &mdash; Sistema de Identificacion de Hablantes</h1>"
        "<p style='color:#888'>Generado por PROY-VCP &middot; Modulo de Explainable AI</p>"
        + meta + sec1 + sec2 + sec3
        + "<footer>PROY-VCP &middot; Pipeline: MediaPipe FaceLandmarker + YuNet + "
          "ArcFace (DeepFace) + pyannote.audio + Whisper<br>"
          "XAI Module: fusion audio&#8596;visual &middot; evidencia LAR &middot; "
          "embedding t-SNE</footer></body></html>"
    )

    with open(OUTPUT_XAI_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Reporte HTML: {OUTPUT_XAI_HTML}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="XAI module for PROY-VCP")
    parser.add_argument("--json", required=True, help="lip_tracking_data.json")
    parser.add_argument("--srt",  default=None,  help="subtitulos.srt (for segment annotations)")
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  XAI — Explainable AI Module")
    print("=" * 55)

    with open(args.json) as f:
        lip_data = json.load(f)
    segmentos_gt = leer_srt_personas(args.srt)

    print(f"\n[1/4] Fusion score heatmap...")
    p8 = graficar_fusion_scores(lip_data)

    print(f"\n[2/4] LAR evidence + landmark sensitivity...")
    p9 = graficar_lar_evidence(lip_data, segmentos_gt)

    print(f"\n[3/4] Embedding t-SNE...")
    p10 = graficar_embeddings_tsne(lip_data)

    print(f"\n[4/4] HTML report...")
    generar_html(lip_data, {"fusion": p8, "lar_evidence": p9, "embeddings": p10})

    print("\n" + "=" * 55)
    print("  XAI completado.")
    if p8:  print(f"  {p8}")
    if p9:  print(f"  {p9}")
    if p10: print(f"  {p10}")
    print(f"  {OUTPUT_XAI_HTML}")
    print("=" * 55)


if __name__ == "__main__":
    main()
