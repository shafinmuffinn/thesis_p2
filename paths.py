"""Central path config for the thesis project.

The same code runs locally on the Mac and on Colab — both environments
override these defaults via environment variables in their setup cell.

Local (Mac) defaults assume the layout in /Users/shafin/Desktop/thesis_p2/.
Colab overrides them to point at /content/drive/MyDrive/Thesis_EAV/...

Usage:
    from paths import EAV_PICKLES, CHECKPOINTS, RESULTS, pretrained
"""
import os
from pathlib import Path


def _is_colab() -> bool:
    """True if running inside a Google Colab kernel."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


# Defaults differ between local Mac and Colab. Env vars override either.
if _is_colab():
    _DEFAULT_ROOT = Path("/content/thesis_p2")
    _DEFAULT_DATA = Path("/content/drive/MyDrive/Thesis_EAV")
else:
    _DEFAULT_ROOT = Path("/Users/shafin/Desktop/thesis_p2")
    # Local Mac has no real data tree; we put it next to the repo so paths
    # remain printable, but no actual pickles live here.
    _DEFAULT_DATA = _DEFAULT_ROOT / "data" / "EAV"

# -- root --
PROJECT_ROOT = Path(os.environ.get("THESIS_ROOT", _DEFAULT_ROOT))

# -- data: never in git, lives on Drive in Colab --
EAV_DATA_ROOT = Path(os.environ.get("EAV_DATA_ROOT", _DEFAULT_DATA))
EAV_PICKLES = Path(os.environ.get("EAV_PICKLES", EAV_DATA_ROOT / "Input_images"))
EAV_RAW = Path(os.environ.get("EAV_RAW", EAV_DATA_ROOT / "raw"))

# -- outputs: also on Drive in Colab so they survive disconnects --
CHECKPOINTS = Path(os.environ.get("CHECKPOINTS", _DEFAULT_DATA / "checkpoints"))
RESULTS = Path(os.environ.get("RESULTS", _DEFAULT_DATA / "results"))
LOGS = Path(os.environ.get("LOGS", _DEFAULT_DATA / "logs"))

# -- pretrained model cache (HF downloads); on Drive to avoid re-downloading --
PRETRAINED = Path(os.environ.get("PRETRAINED", _DEFAULT_DATA / "pretrained"))

# Hugging Face model IDs used by the EAV repo
HF_AUDIO_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
HF_VISION_MODEL = "dima806/facial_emotions_image_detection"

# EAV emotion mapping (must match Dataload_audio.py and Dataload_vision.py)
EMOTION_TO_IDX = {"Neutral": 0, "Sadness": 1, "Anger": 2, "Happiness": 3, "Calmness": 4}
IDX_TO_EMOTION = {v: k for k, v in EMOTION_TO_IDX.items()}


def pretrained(name: str) -> Path:
    """Path to a local cache of a HF model, e.g. pretrained('ast-finetuned-audioset')."""
    return PRETRAINED / name


def ensure_dirs():
    for p in (EAV_DATA_ROOT, EAV_PICKLES, CHECKPOINTS, RESULTS, LOGS, PRETRAINED):
        p.mkdir(parents=True, exist_ok=True)


def summary() -> str:
    return "\n".join([
        f"PROJECT_ROOT = {PROJECT_ROOT}",
        f"EAV_PICKLES  = {EAV_PICKLES}",
        f"CHECKPOINTS  = {CHECKPOINTS}",
        f"RESULTS      = {RESULTS}",
        f"PRETRAINED   = {PRETRAINED}",
    ])


if __name__ == "__main__":
    ensure_dirs()
    print(summary())
