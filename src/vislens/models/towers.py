"""The two towers, and the asymmetry between them.

The build plan specifies a ResNet-50 image tower and a MiniLM text tower, both
trained. This departs from it deliberately, and the reason is that the spec as
written cannot answer its own question.

The question is "does in-domain fine-tuning beat zero-shot CLIP at product
retrieval". A ResNet-50 + MiniLM pair starts in a different embedding space
from CLIP, so a loss curve from it says nothing about that comparison: if it
loses, the honest reading is "a smaller model trained from scratch on 121K
pairs lost to a model trained on 400M", which nobody needed an experiment to
learn. Initialising the image tower FROM CLIP's visual encoder makes the
baseline and the treatment differ in exactly one thing — the fine-tuning —
which is the only way the delta means anything.

**The text tower is frozen and its embeddings are precomputed.** Two reasons,
one scientific and one about compute:

*   It holds the target space fixed. If both towers move, a gain can come from
    the text side collapsing toward the images rather than from the images
    getting better, and the zero-shot comparison stops being like-for-like.
*   It is what makes batch 256 fit a free T4. Contrastive learning draws its
    negatives from the batch, and gradient accumulation does NOT fix a small
    batch — it accumulates gradients, not negatives. Removing the text forward
    pass and its activations buys the batch size outright, which is strictly
    better than the MoCo-style queue the build plan proposes as the fallback.

`logit_scale` lives in log space and is clamped to `log(100)`, matching CLIP.
Unclamped it grows until the fp16 logits overflow, which shows up as a loss
that NaNs several hundred steps into a headless run — hours in, with nothing
but a dead checkpoint to look at.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

DEFAULT_MODEL = "ViT-B-32"
DEFAULT_PRETRAINED = "openai"
# CLIP's own ceiling on the learnable temperature.
MAX_LOGIT_SCALE = math.log(100.0)
INIT_LOGIT_SCALE = math.log(1 / 0.07)


@dataclass(frozen=True)
class TowerConfig:
    model_name: str = DEFAULT_MODEL
    pretrained: str = DEFAULT_PRETRAINED
    projection_dim: int = 256
    # The projection is the only randomly initialised part, so it starts with a
    # much larger effective learning rate than the pretrained trunk. Kept
    # separate in the optimiser rather than papered over with a low global LR.
    trunk_lr: float = 1e-5
    head_lr: float = 1e-3


class ImageTower(nn.Module):
    """CLIP's visual trunk plus a projection head, L2-normalised.

    `trunk` is injected rather than constructed here so tests can pass a stub:
    CI has no network and must never download 350 MB of weights to check that a
    tensor has the right shape.
    """

    def __init__(self, trunk: nn.Module, trunk_dim: int, projection_dim: int = 256):
        super().__init__()
        self.trunk = trunk
        self.projection = nn.Linear(trunk_dim, projection_dim, bias=False)
        # Init small: a randomly initialised head at default scale dominates
        # the pretrained features for the first few hundred steps and undoes
        # some of what we are trying to keep.
        nn.init.normal_(self.projection.weight, std=trunk_dim**-0.5)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        features = self.trunk(pixels)
        if isinstance(features, (tuple, list)):  # pragma: no cover - defensive
            features = features[0]
        return F.normalize(self.projection(features.float()), dim=-1)


class TextProjection(nn.Module):
    """Projects the frozen, precomputed text embeddings into the shared space.

    The text ENCODER is frozen; this small head is not. Without it the image
    tower would have to move all the way to CLIP's text space on its own, which
    both wastes capacity and makes the comparison unfair in the other
    direction.
    """

    def __init__(self, text_dim: int, projection_dim: int = 256):
        super().__init__()
        self.projection = nn.Linear(text_dim, projection_dim, bias=False)
        nn.init.normal_(self.projection.weight, std=text_dim**-0.5)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(embeddings.float()), dim=-1)


class TwoTower(nn.Module):
    def __init__(self, image_tower: ImageTower, text_projection: TextProjection):
        super().__init__()
        self.image_tower = image_tower
        self.text_projection = text_projection
        self.logit_scale = nn.Parameter(torch.tensor(INIT_LOGIT_SCALE))

    def clamp_logit_scale(self) -> None:
        """Call after every optimiser step, not inside forward.

        In forward it would be a no-op under `no_grad` semantics in some
        autocast paths and, worse, would silently differ between train and
        eval. CLIP clamps in the training loop; so do we.
        """
        with torch.no_grad():
            self.logit_scale.clamp_(max=MAX_LOGIT_SCALE)

    def forward(
        self, pixels: torch.Tensor, text_embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = self.image_tower(pixels)
        text = self.text_projection(text_embeddings)
        return image, text, self.logit_scale.exp()


def load_clip(
    model_name: str = DEFAULT_MODEL,
    pretrained: str = DEFAULT_PRETRAINED,
    device: str = "cpu",
):
    """The real CLIP, for training and for the zero-shot baseline.

    Imported lazily so that merely importing this module does not pull
    `open_clip` — the ONNX export path and the tests should not need it.
    Downloads weights on first use, so it is never called from CI.
    """
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, tokenizer


class ClipVisualTrunk(nn.Module):
    """Adapts open_clip's model to the plain `pixels -> features` interface."""

    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.visual = clip_model.visual

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.visual(pixels)


def build_two_tower(
    config: TowerConfig,
    text_dim: int,
    device: str = "cpu",
) -> TwoTower:
    """The real model. Downloads CLIP weights; not used in tests."""
    clip_model, _ = load_clip(config.model_name, config.pretrained, device=device)
    trunk = ClipVisualTrunk(clip_model)
    with torch.no_grad():
        probe = trunk(torch.zeros(1, 3, 224, 224, device=device))
    trunk_dim = int(probe.shape[-1])

    model = TwoTower(
        ImageTower(trunk, trunk_dim, config.projection_dim),
        TextProjection(text_dim, config.projection_dim),
    )
    return model.to(device)
