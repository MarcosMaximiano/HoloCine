import os
import subprocess
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import storage

app = Flask(__name__)
CHECKPOINTS_READY = False
CHECKPOINT_FILES = [
    "checkpoints/Wan2.2-T2V-A14B/models_t5_umt5-xxl-enc-bf16.pth",
    "checkpoints/Wan2.2-T2V-A14B/Wan2.1_VAE.pth",
    "checkpoints/HoloCine_dit/full/full_high_noise.safetensors",
    "checkpoints/HoloCine_dit/full/full_low_noise.safetensors",
]

def missing_checkpoints():
    return [path for path in CHECKPOINT_FILES if not os.path.exists(path)]

def ensure_checkpoints():
    global CHECKPOINTS_READY
    if CHECKPOINTS_READY:
        return
    missing = missing_checkpoints()
    if missing:
        subprocess.run(["python3", "scripts/download_checkpoints.py", "--yes"], check=True)
    missing = missing_checkpoints()
    if missing:
        raise FileNotFoundError(
            f"Missing checkpoint files after download: {', '.join(missing)}"
        )
    CHECKPOINTS_READY = True

# Healthcheck endpoint
@app.route("/healthz", methods=["GET"])
def healthz():
    return "ok", 200

# Endpoint para geração de vídeo
@app.route("/generate-video", methods=["POST"])
def generate_video():
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "prompt é obrigatório"}), 400

    try:
        ensure_checkpoints()
        output_file = "output.mp4"
        cmd = [
            "python3", "HoloCine_inference_full_attention.py",
            "--prompt", prompt,
            "--output", output_file
        ]
        subprocess.run(cmd, check=True)

        bucket = storage.bucket()
        blob = bucket.blob(f"videos/{output_file}")
        blob.upload_from_filename(output_file)
        blob.make_public()

        return jsonify({"videoUrl": blob.public_url}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"inferência falhou: {e}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
