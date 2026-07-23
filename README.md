# PROY-VCP

Ejemplo de captura de habla: https://github.com/paulabaal12/PROY-VCP/blob/main/Videom.mp4

Ejemplo practico: https://github.com/paulabaal12/PROY-VCP/blob/main/Final.mp4

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

python main.py --video data/videos/video.mp4 --token hf_


python main.py --video data/videos/video.mp4 --token hf_TU_TOKEN --n_personas 2 --min_speakers 2 --max_speakers 2 --modelo base

python src/diarizacion.py --video data/videos/video.mp4 --json data/json/lip_tracking_data.json --token hf_TU_TOKEN --modelo base --min_speakers 2 --max_speakers 2

python src/render_final.py --video data/videos/video.mp4 --json data/json/lip_tracking_data.json --srt results/srt/subtitulos.srt

Para correro el pipeline sin el lip_tracking (cuando ya se generó el .json):
python main.py --video data/videos/-nombre_del_archivo-.mp4 --token hf_xxx --skip_lip_tracking


## ⚠️ Disclaimer

Este modelo fue entrenado y optimizado para video en **inglés**. Su desempeño con audio en otros idiomas puede ser inconsistente.

Además, los caracteres acentuados (á, é, í, ó, ú, ñ, ü) pueden aparecer mal interpretados o codificados incorrectamente en la transcripción