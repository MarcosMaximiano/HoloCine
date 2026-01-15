# Imagem base leve de Python
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Dependências do SO (ffmpeg necessário para vídeo)
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependências do projeto
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Instalar Flask/Gunicorn/Firebase/GCS/HuggingFace
RUN pip install --no-cache-dir flask gunicorn firebase-admin google-cloud-storage huggingface_hub

# Copiar todo o código do projeto
COPY . .

# Baixar checkpoints do Hugging Face durante o build da imagem
RUN python - <<'PYCODE'
from huggingface_hub import snapshot_download
import os

os.makedirs("checkpoints/Wan2.2-T2V-A14B", exist_ok=True)
os.makedirs("checkpoints/HoloCine_dit/full", exist_ok=True)

snapshot_download(
    repo_id="Wan-AI/Wan2.2-T2V-A14B",
    local_dir="checkpoints/Wan2.2-T2V-A14B",
    allow_patterns=["models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth"]
)
snapshot_download(
    repo_id="hlwang06/HoloCine",
    local_dir="checkpoints",
    allow_patterns=[
        "HoloCine_dit/full/full_high_noise.safetensors",
        "HoloCine_dit/full/full_low_noise.safetensors"
    ]
)
PYCODE

# Iniciar o servidor Flask via Gunicorn com timeout maior para IA
CMD ["gunicorn", "-b", "0.0.0.0:8080", "--timeout", "600", "main:app"]
