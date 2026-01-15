# Imagem base leve de Python (CPU)
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Dependências do SO
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Dependências Python do projeto
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Servidor + HF hub
RUN pip install --no-cache-dir flask gunicorn huggingface_hub firebase-admin google-cloud-storage

# Copie o seu código
COPY . .

# (Opcional) use variável de ambiente HF_TOKEN no Cloud Build/Run se precisar de modelos privados
# ARG HF_TOKEN
# ENV HUGGINGFACE_HUB_TOKEN=${HF_TOKEN}

# Baixar checkpoints necessários (sem interação)
RUN python - <<EOF
from huggingface_hub import snapshot_download
import os
os.makedirs("checkpoints/Wan2.2-T2V-A14B", exist_ok=True)
os.makedirs("checkpoints/HoloCine_dit/full", exist_ok=True)

snapshot_download(
    repo_id="Wan-AI/Wan2.2-T2V-A14B",
    local_dir="checkpoints/Wan2.2-T2V-A14B",
    allow_patterns=["models_t5_umt5-xxl-enc-bf16.pth","Wan2.1_VAE.pth"]
)
snapshot_download(
    repo_id="hlwang06/HoloCine",
    local_dir="checkpoints",
    allow_patterns=[
        "HoloCine_dit/full/full_high_noise.safetensors",
        "HoloCine_dit/full/full_low_noise.safetensors"
    ]
)
EOF

# Servir Flask via Gunicorn com timeout maior (IA é lenta)
CMD ["gunicorn", "-b", "0.0.0.0:8080", "--timeout", "600", "main:app"]
