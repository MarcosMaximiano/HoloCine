import shutil
import subprocess
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    pip_command = [sys.executable, "-m", "pip", "install", "huggingface-hub"]
    pip_display = " ".join(pip_command)
    response = input(
        f"huggingface_hub is required. Run '{pip_display}' now? [y/N] "
    ).strip().lower()
    if response not in {"y", "yes"}:
        raise SystemExit("Install huggingface-hub to continue.")
    subprocess.check_call(pip_command)
    from huggingface_hub import hf_hub_download

def ensure_dir(path: Path) -> Path:
    for parent in path.parents:
        if parent.exists() and parent.is_file():
            raise SystemExit(
                f"Expected '{parent}' to be a directory. Remove or rename the file "
                f"(e.g. 'rm {parent}') to continue."
            )
    if path.exists() and path.is_file():
        path.unlink()
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_dir(path: Path, expected_files: set[str]) -> None:
    """Remove any files or directories not listed in expected_files."""
    if not path.exists():
        return
    unexpected_files = []
    unexpected_dirs = []
    for entry in path.iterdir():
        if entry.is_file():
            if entry.name not in expected_files:
                unexpected_files.append(entry)
        else:
            unexpected_dirs.append(entry)
    unexpected = unexpected_files + unexpected_dirs
    if not unexpected:
        return
    response = input(
        f"Remove {len(unexpected)} unexpected item(s) from '{path}'? [y/N] "
    ).strip().lower()
    if response not in {"y", "yes"}:
        raise SystemExit("Cleanup aborted. Remove unexpected files to continue.")
    for entry in unexpected:
        if entry.is_dir():
            print(f"Removing unexpected directory: {entry}")
            shutil.rmtree(entry)
        else:
            print(f"Removing unexpected file: {entry}")
            entry.unlink()

# Target folders
WAN_DIR = ensure_dir(Path("checkpoints/Wan2.2-T2V-A14B"))
HOLO_DIR = ensure_dir(Path("checkpoints/HoloCine_dit"))
HOLO_FULL_DIR = ensure_dir(HOLO_DIR / "full")

WAN_FILES = {
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.1_VAE.pth",
}
HOLO_FILES = {
    "full_high_noise.safetensors",
    "full_low_noise.safetensors",
}

# Ensure clean directories
clean_dir(WAN_DIR, WAN_FILES)
clean_dir(HOLO_FULL_DIR, HOLO_FILES)

# Download Wan 2.2 T2V (required files)
for filename in sorted(WAN_FILES):
    hf_hub_download(
        repo_id="Wan-AI/Wan2.2-T2V-A14B",
        filename=filename,
        local_dir=WAN_DIR,
        local_dir_use_symlinks=False,
    )

# Download HoloCine DIT (full)
for filename in sorted(HOLO_FILES):
    hf_hub_download(
        repo_id="hlwang06/HoloCine",
        filename=f"HoloCine_dit/full/{filename}",
        local_dir="checkpoints",
        local_dir_use_symlinks=False,
    )

print("Checkpoints ready in:", WAN_DIR, "and", HOLO_FULL_DIR)
