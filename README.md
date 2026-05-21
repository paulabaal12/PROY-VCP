# PROY-VCP

## Estructura de Carpetas

```
PROY-VCP/
│
├── src/                  # Código fuente
│   ├── lip_tracking.py   # Detección de labios y rostros
│   ├── diarizacion.py    # Diarización y fusión audio-visual
│
├── data/                 # Datos de entrada y salida intermedia
│   ├── videos/           # Videos originales
│   ├── audio/            # Audios extraídos
│   └── json/             # Resultados intermedios (ej: lip_tracking_data.json)
│
├── results/              # Resultados finales
│   ├── videos/           # Videos anotados (output_annotated.mp4)
│   ├── plots/            # Gráficas (senal_lar.png, diarizacion.png)
│   ├── srt/              # Subtítulos generados
│   ├── logs/             # Logs o reportes
│   └── frames/           
│
├── README.md
└── requirements.txt
```

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecución

1. Coloca tus videos en `data/videos/`.
2. Ejecuta los scripts desde la carpeta raíz:

```bash
python src/lip_tracking.py --video data/videos/video.mp4
python src/diarizacion.py --video data/videos/video.mp4 --json data/json/lip_tracking_data.json --token hf_...
```

## Descripción de los scripts
- **lip_tracking.py**: Detecta rostros y movimiento de labios, exporta resultados a JSON y genera video anotado.
- **diarizacion.py**: Diarización de audio, fusión con señal visual, transcripción y generación de subtítulos.

```bash
python main.py --video data/videos/<nombre del archivo>.mp4 --token hf_cXXX
```



python main.py --video data/videos/video.mp4 --token hf_ze

python src/render_final.py --video data/videos/video.mp4 --json data/json/lip_tracking_data.json --srt results/srt/subtitulos.srt

python src/render_final.py --video data/videos/video.mp4 --json data/json/lip_tracking_data.json --srt results/srt/subtitulos.srt --metricas