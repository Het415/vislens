"""The one definition of preprocessing.

`CLAUDE.md` section 9: training, eval, the ONNX export and the parity fixtures
all import from here, and a second definition anywhere is the bug — it does not
fail, it silently degrades the product. A model trained on one resize and served
on another is still a model; it is just a worse one, and nothing in the metrics
says so.

Three decisions worth stating, because each is a place two implementations
drift apart:

*   **Interpolation is pinned to PIL `BICUBIC`.** PIL, cv2 and
    torchvision-without-antialias produce visibly different pixels for the same
    resize, and CLIP's published weights were trained on the PIL path. Getting
    this wrong costs a couple of points of recall and looks like a bad
    fine-tune.
*   **No normalization here.** Mean/std and the final L2 go *inside* the ONNX
    graph, so a consumer owns only resize + crop + scale and cannot forget the
    rest. That is why this module stops at [0, 1].
*   **No torch.** This is importable by the audit service, which must never
    grow a torch dependency (`CLAUDE.md` section 2), and by the ONNX export,
    which should not need the training stack to describe its own input. PIL and
    numpy only.

The parameters are serialized by `preprocess_config()` and written next to any
exported model, so a consumer reads them rather than reimplementing them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# CLIP ViT-B/32's input geometry. Kept as constants rather than arguments: a
# caller free to pass 256 is a caller free to silently mismatch the exported
# graph, which is the failure this module exists to prevent.
IMAGE_SIZE = 224
RESIZE_SHORTEST = 224

# OpenAI CLIP's published statistics. Recorded here so the ONNX graph and any
# reader of `preprocess.json` agree on them, even though this module does not
# apply them.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

INTERPOLATION = "bicubic"
_RESAMPLE = Image.Resampling.BICUBIC


@dataclass(frozen=True)
class PreprocessConfig:
    """Everything a consumer needs to reproduce the model's input exactly."""

    image_size: int = IMAGE_SIZE
    resize_shortest: int = RESIZE_SHORTEST
    interpolation: str = INTERPOLATION
    scale: float = 1 / 255.0
    mean: tuple[float, float, float] = CLIP_MEAN
    std: tuple[float, float, float] = CLIP_STD
    # Stated so a consumer knows the graph already did it and must not repeat
    # it. Normalizing twice is a silent 20-point recall loss.
    normalization_in_graph: bool = True
    l2_normalize_in_graph: bool = True
    channel_order: str = "RGB"
    layout: str = "CHW"


def preprocess_config() -> PreprocessConfig:
    return PreprocessConfig()


def write_preprocess_json(path: str | Path) -> Path:
    """Write `preprocess.json` beside an exported model."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(preprocess_config()), indent=2) + "\n")
    return out


def resize_and_crop(img: Image.Image) -> Image.Image:
    """Shortest side to `RESIZE_SHORTEST`, then a centre crop to a square.

    Aspect ratio is preserved by the resize and then discarded by the crop,
    which is what CLIP expects. Cropping rather than squashing matters on this
    dataset: ABO product photos are frequently not square, and a squashed
    product is a different-looking product.
    """
    width, height = img.size
    if width == 0 or height == 0:  # pragma: no cover - defensive
        raise ValueError("image has a zero dimension")

    scale = RESIZE_SHORTEST / min(width, height)
    # `round` rather than `int`: truncation can leave the shortest side one
    # pixel under the crop size, and the crop then pads with black.
    new_size = (
        max(round(width * scale), RESIZE_SHORTEST),
        max(round(height * scale), RESIZE_SHORTEST),
    )
    resized = img.resize(new_size, _RESAMPLE)

    left = (new_size[0] - IMAGE_SIZE) // 2
    top = (new_size[1] - IMAGE_SIZE) // 2
    return resized.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))


def to_array(img: Image.Image) -> np.ndarray:
    """A decoded PIL image to the model's input array, float32 CHW in [0, 1].

    Alpha is composited onto WHITE, not dropped. A transparent PNG dropped to
    RGB gives black wherever it was transparent, which on product photography
    means a black silhouette where the background should be — the same trap the
    audit's `had_alpha` check exists to surface.
    """
    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        flat = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        flat.alpha_composite(rgba)
        img = flat.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    array = np.asarray(img, dtype=np.float32) / 255.0
    return np.ascontiguousarray(array.transpose(2, 0, 1))


def preprocess(img: Image.Image) -> np.ndarray:
    """The whole path: PIL image to model input."""
    return to_array(resize_and_crop(img))


def preprocess_bytes(data: bytes) -> np.ndarray:
    """The same, from encoded bytes — what the shard pipeline holds."""
    import io

    with Image.open(io.BytesIO(data)) as img:
        img.load()
        return preprocess(img)


def describe() -> dict[str, Any]:
    """The config as a plain dict, for embedding in a run record."""
    return asdict(preprocess_config())
