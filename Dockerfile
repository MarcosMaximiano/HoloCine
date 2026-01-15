# Base CPU
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

# Servidor + Hugging Face + libs da API
RUN pip install --no-cache-dir flask gunicorn huggingface_hub firebase-admin google-cloud-storage

# Copiar o código do projeto (inclui script de download)
COPY . .

# Limpeza preventiva: se houver arquivos com nomes de pasta, remova-os
RUN rm -rf checkpoints/Wan2.2-T2V-A14B || true && \
    rm -rf checkpoints/HoloCine_dit || true && \
    mkdir -p checkpoints/Wan2.2-T2V-A14B checkpoints/HoloCine_dit/full

# Baixar checkpoints do Hugging Face (sem login interativo)
RUN python scripts/download_checkpoints.py

# Servir Flask via Gunicorn (timeout maior para IA)
CMD ["gunicorn", "-b", "0.0.0.0:8080", "--timeout", "600", "main:app"]
