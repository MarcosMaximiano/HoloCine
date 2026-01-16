import argparse
import shutil
import subprocess
import sys
from pathlib import Path

def confirm_action(message: str, assume_yes: bool) -> None:
    if assume_yes:
        print(f"{message}\nAuto-confirmed (--yes).")
        return
    try:
        response = input(message).strip().lower()
    except EOFError as exc:
        raise SystemExit(
            "Non-interactive mode detected. Re-run with --yes to automatically confirm all prompts."
        ) from exc
    if response not in {"y", "yes"}:
        raise SystemExit("Operation aborted by user.")

def ensure_dir(path: Path) -> Path:
    for parent in path.parents:
        if parent.exists() and parent.is_file():
            raise SystemExit(
                f"Conflicting parent path detected: '{parent}'. Remove or rename the file "
                f"(e.g. 'rm {parent}') to continue."
            )
    if path.exists() and path.is_file():
        path.unlink()
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_dir(
    path: Path,
    expected_files: set[str],
    expected_dirs: set[str] | None = None,
    assume_yes: bool = False,
) -> None:
    """Remove any files or directories not listed in expected_files/expected_dirs."""
    if not path.exists():
        return
    expected_dirs = expected_dirs or set()
    entries = list(path.iterdir())
    unexpected_files = [
        entry for entry in entries if entry.is_file() and entry.name not in expected_files
    ]
    unexpected_dirs = [entry for entry in entries if entry.is_dir() and entry.name not in expected_dirs]
    unexpected = unexpected_files + unexpected_dirs
    if not unexpected:
        return
    unexpected_list = "\n".join(f"- {entry}" for entry in unexpected)
    confirm_action(
        f"Remove the following {len(unexpected)} item(s) from '{path}'?\n"
        f"{unexpected_list}\n\nProceed? [y/N] ",
        assume_yes,
    )
    for entry in unexpected:
        if entry.is_dir():
            print(f"Removing unexpected directory: {entry}")
            shutil.rmtree(entry)
        else:
            print(f"Removing unexpected file: {entry}")
            entry.unlink()

def main() -> None:
    parser = argparse.ArgumentParser(description="Download required HoloCine checkpoints.")
    parser.add_argument("--yes", action="store_true", help="Automatically confirm all prompts.")
    args = parser.parse_args()
    assume_yes = args.yes

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        pip_package = "huggingface_hub>=0.20.0,<1.0.0"
        pip_command = [sys.executable, "-m", "pip", "install", pip_package]
        pip_display = " ".join(pip_command)
        confirm_action(
            "This will install the latest version of "
            f"'{pip_package}' from PyPI. Run '{pip_display}' now? [y/N] ",
            assume_yes,
        )
        subprocess.check_call(pip_command)
        from huggingface_hub import hf_hub_download

    # Target folders
    WAN_DIR = ensure_dir(Path("checkpoints/Wan2.2-T2V-A14B"))
    HOLO_FULL_DIR = ensure_dir(Path("checkpoints/HoloCine_dit/full"))

    WAN_FILES = {
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.1_VAE.pth",
    }
    HOLO_FILES = {
        "full_high_noise.safetensors",
        "full_low_noise.safetensors",
    }
    HOLO_REPO_DIR = "HoloCine_dit/full"

    # Ensure clean directories
    clean_dir(WAN_DIR, WAN_FILES, assume_yes=assume_yes)
    clean_dir(HOLO_FULL_DIR, HOLO_FILES, assume_yes=assume_yes)

    # Download Wan 2.2 T2V (required files)
    for filename in sorted(WAN_FILES):
        print(f"Downloading {filename} to {WAN_DIR}...")
        hf_hub_download(
            repo_id="Wan-AI/Wan2.2-T2V-A14B",
            filename=filename,
            local_dir=WAN_DIR,
            local_dir_use_symlinks=False,
        )

    # Download HoloCine DIT (full)
    for filename in sorted(HOLO_FILES):
        print(f"Downloading {filename} to {HOLO_FULL_DIR}...")
        hf_hub_download(
            repo_id="hlwang06/HoloCine",
            filename=f"{HOLO_REPO_DIR}/{filename}",
            local_dir="checkpoints",
            local_dir_use_symlinks=False,
        )

    print("Checkpoints ready in:", WAN_DIR, "and", HOLO_FULL_DIR)


if __name__ == "__main__":
    main()
