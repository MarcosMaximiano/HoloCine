# Imagem base leve de Python
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Dependências do SO (ffmpeg costuma ser necessário para vídeo)
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependências do projeto
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Servidor web + SDK Firebase
RUN pip install --no-cache-dir flask gunicorn firebase-admin google-cloud-storage

# Copiar todo o código do repo
COPY . .

# Iniciar servidor Flask via Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "main:app"]
