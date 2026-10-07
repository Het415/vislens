"""What each check means to a seller, and how to reshoot the photo to pass it.

This exists because rule text written for an auditor reached the seller
unchanged. Given `"product bounding box ≥85% of the frame"` and
`"longest side >= 1000px for zoom"`, ListingLens' synthesizer produced next
actions like "crop or adjust the shot so the product fills ≥ 85 % of the frame"
— accurate, and useless to someone holding a phone over their product. Neither
the seller nor the developer could say what to physically do differently.

The fix is data, not a prompt instruction, for the same reason the advisory
findings are absent rather than labelled (see `build_audit_payload`): an LLM
paraphrases what it is given. So the payload now carries, per breached rule, a
neutral `title` and a `fix` written as photography steps. The synthesizer and
the UI card both read the same sentence, so the advice cannot differ between
them.

Two constraints on the wording:

*   **A fix describes how to reshoot, never what the check saw.** The artifact
    check deliberately cannot tell a logo from a prop from a stray speck, so
    its fix lists what to remove in general rather than naming what was found.
*   **Every number comes from `rules`.** A fix that says "1,000 pixels" while
    the check measures something else is the drift `rules_v1.json` exists to
    prevent.

Kept short on purpose: the whole payload has a 2400-character budget, and a fix
is sent once per breached rule, not once per image.
"""

from __future__ import annotations

from typing import Any

# Neutral names, not problem statements. The legend also carries entries for
# advisory checks, and a title like "Photo is the wrong shape" on one of those
# would read as a verdict that was never made.
_TITLES: dict[str, str] = {
    "resolution_and_format": "Photo size",
    "white_background": "White background",
    "frame_occupancy": "Product size in the photo",
    "background_artifacts": "Anything else on the background",
    "aspect_ratio": "Photo shape",
    "image_count": "Number of photos",
}


def title_for(check_id: str) -> str:
    """Plain-language name for a check, falling back to the id itself."""
    return _TITLES.get(check_id, check_id)


def fix_for(check_id: str, rules: dict[str, Any]) -> str | None:
    """How to reshoot or re-export the photo so it passes `check_id`.

    `None` for checks that are never verdicts (aspect ratio is a
    recommendation, so a fix for it would assert a problem that does not
    exist).
    """
    checks = rules["checks"]
    if check_id == "resolution_and_format":
        zoom = checks["resolution_and_format"]["zoom_min_longest_side"]
        return (
            f"Phone cameras shoot well over {zoom:,} pixels, so a small photo has "
            "usually been shrunk: upload the original JPEG or PNG, not a "
            "screenshot or chat-app copy, and move closer instead of zooming."
        )
    if check_id == "white_background":
        return (
            "Shoot on white poster board curved up behind the product, lit from "
            "both sides so it doesn't look grey, then make the background pure "
            "white with a background-remover app."
        )
    if check_id == "frame_occupancy":
        return (
            "Move the phone closer until the product nearly touches the edges, "
            "leaving a thin white margin — or crop off the empty space."
        )
    if check_id == "background_artifacts":
        return (
            "Remove everything except the product being sold: props, extra items, "
            "and added badges, text, borders or watermarks (fine on other photos, "
            "not the main one)."
        )
    if check_id == "image_count":
        return (
            "Add photos buyers need: other angles, a close-up of the details, the "
            "product in use, one showing its size (in a hand), and what's in the "
            "box."
        )
    return None


DUPLICATE_FIX = (
    "Replace one of the repeated photos with a different shot: another angle, a "
    "close-up, or the product in use."
)
