"""Symmetric InfoNCE, with the false negatives taken out.

Contrastive training treats every other item in the batch as a negative. On
this catalogue that assumption is wrong often enough to matter, in two ways:

*   **The same ASIN appears more than once.** ABO lists a product per
    marketplace, so `pairs.parquet` holds 121,307 rows over 120,626 distinct
    `product_id`s. Two rows for one ASIN are the same product.
*   **Different ASINs share a title.** "Amazon Brand - Solimo Designer …" style
    titles repeat across variants, and the frozen text tower maps identical
    strings to identical embeddings — so the model is asked to push apart two
    points that are, in the target space, the same point.

Either way the gradient is asking for something unachievable, and the effect is
not noise: it is a ceiling on the metric that more training cannot lift. The
build plan does not mention it. This is the cheapest thing on the list to get
wrong and the hardest to notice afterwards, because the loss still goes down.

**The loss is computed in fp32 even under autocast.** On a T4 the logits are
fp16, and `logit_scale` up to 100 pushes a normalised dot product to a
magnitude where the exponential in `log_softmax` overflows. That surfaces as a
NaN several hundred steps in — i.e. hours into a headless Kaggle commit, with
a dead checkpoint as the only evidence.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def false_negative_mask(
    product_ids: torch.Tensor,
    title_hashes: torch.Tensor,
) -> torch.Tensor:
    """`True` where a cell is a false negative and must be excluded.

    Both arguments are int64 tensors of shape (B,) — ids hashed upstream so the
    comparison is a tensor op rather than a Python loop over strings.

    The diagonal is explicitly cleared: it is the positive, and masking it
    would remove the only term the numerator has.
    """
    same_product = product_ids[:, None] == product_ids[None, :]
    same_title = title_hashes[:, None] == title_hashes[None, :]
    # A zero hash means "no title" (an `index`-role sample). Two of those are
    # not the same product, so do not let 0 == 0 merge them.
    has_title = title_hashes != 0
    same_title &= has_title[:, None] & has_title[None, :]

    mask = same_product | same_title
    mask.fill_diagonal_(False)
    return mask


def info_nce(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    logit_scale: torch.Tensor,
    *,
    product_ids: torch.Tensor | None = None,
    title_hashes: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Symmetric InfoNCE. Returns the loss and a few scalars worth logging.

    Both inputs are expected already L2-normalised — the towers do it, so doing
    it again here would be a second definition of the same thing.
    """
    # fp32 from here down, whatever autocast set up above. See the module note.
    image = image_embeddings.float()
    text = text_embeddings.float()
    scale = logit_scale.float()

    logits = scale * image @ text.t()

    if product_ids is not None and title_hashes is not None:
        mask = false_negative_mask(product_ids, title_hashes)
        # -inf rather than a large negative: a large negative still contributes
        # to the denominator, and with scale up to 100 "large" is relative.
        logits = logits.masked_fill(mask, float("-inf"))
        n_masked = int(mask.sum())
    else:
        n_masked = 0

    targets = torch.arange(logits.shape[0], device=logits.device)
    loss_i2t = F.cross_entropy(logits, targets)
    loss_t2i = F.cross_entropy(logits.t(), targets)
    loss = (loss_i2t + loss_t2i) / 2

    with torch.no_grad():
        acc_i2t = (logits.argmax(dim=1) == targets).float().mean()
        stats = {
            "loss": float(loss),
            "loss_i2t": float(loss_i2t),
            "loss_t2i": float(loss_t2i),
            "batch_acc_i2t": float(acc_i2t),
            "logit_scale": float(scale),
            # Published per step: if this climbs, the batch is full of
            # duplicates and the effective number of negatives is not the
            # batch size, which changes how the loss should be read.
            "false_negatives_masked": n_masked,
        }
    return loss, stats
