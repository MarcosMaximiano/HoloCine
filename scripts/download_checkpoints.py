import os
from huggingface_hub import snapshot_download

def ensure_dir(path: str):
    # Se existe e é arquivo, remove; se é dir, ok; caso contrário, cria
    if os.path.exists(path) and not os.path.isdir(path):
        os.remove(path)
    os.makedirs(path, exist_ok=True)

# Pastas alvo
WAN_DIR = "checkpoints/Wan2.2-T2V-A14B"
HOLO_FULL_DIR = "checkpoints/HoloCine_dit/full"

# Garantir estrutura
ensure_dir(WAN_DIR)
ensure_dir(HOLO_FULL_DIR)

# Baixar Wan 2.2 T2V (arquivos necessários)
snapshot_download(
    repo_id="Wan-AI/Wan2.2-T2V-A14B",
    local_dir=WAN_DIR,
    allow_patterns=[
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.1_VAE.pth",
    ],
)

# Baixar HoloCine DIT (full)
snapshot_download(
    repo_id="hlwang06/HoloCine",
    local_dir="checkpoints",
    allow_patterns=[
        "HoloCine_dit/full/full_high_noise.safetensors",
        "HoloCine_dit/full/full_low_noise.safetensors",
    ],
)

print("Checkpoints prontos em:", WAN_DIR, "e", HOLO_FULL_DIR)
