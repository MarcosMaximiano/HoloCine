from huggingface_hub import snapshot_download
import os

WAN_DIR = "checkpoints/Wan2.2-T2V-A14B"
HOLO_DIR = "checkpoints/HoloCine_dit/full"

os.makedirs(WAN_DIR, exist_ok=True)
os.makedirs(HOLO_DIR, exist_ok=True)

# Wan 2.2 T2V
snapshot_download(
    repo_id="Wan-AI/Wan2.2-T2V-A14B",
    local_dir=WAN_DIR,
    allow_patterns=["models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth"],
)

# HoloCine DIT (full)
snapshot_download(
    repo_id="hlwang06/HoloCine",
    local_dir="checkpoints",
    allow_patterns=[
        "HoloCine_dit/full/full_high_noise.safetensors",
        "HoloCine_dit/full/full_low_noise.safetensors",
    ],
)

print("Checkpoints baixados/confirmados em:", WAN_DIR, "e", HOLO_DIR)
