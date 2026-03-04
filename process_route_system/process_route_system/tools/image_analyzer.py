"""
tools/image_analyzer.py — JPG Image Analysis Tool for the Feature Extraction Agent (FEA).

Sends part multi-view images to the GLM-4V multimodal LLM and returns a
structured textual description of the part geometry, feature locations, and
surface characteristics.

Mirrors the paper's use of GLM-4V (Section 3.3, tool 2).

Dependencies: requests, base64 (stdlib), pathlib (stdlib)
"""

import base64
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests

logger = logging.getLogger(__name__)

# ── GLM-4V API wrapper ─────────────────────────────────────────────────────────

GLM_API_ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

SYSTEM_PROMPT = """\
You are a professional mechanical engineer specialising in CNC machining and \
process planning. You will receive one or more orthographic views (front, side, \
top, isometric) of a machined part. Analyse all visible geometric features and \
produce a structured description containing:

1. General Shape: (e.g. rectangular block, cylindrical body, L-bracket …)
2. Feature Description: list every identifiable machining feature with its \
   approximate location, orientation, and any visible dimension cues. Include:
   - Through holes / blind holes (diameter, depth if visible, location)
   - Rectangular channels / slots (width, depth, orientation)
   - Rectangular grooves / pockets (size, location)
   - Flat reference surfaces
   - Chamfers, fillets, rounded corners
3. Surface Treatment: (e.g. matte, polished, anodised …)
4. Colour
5. Other Features: anything not covered above

Be concise but thorough. Use bullet points for the Feature Description section.\
"""

IMAGE_PROMPT = """\
Please analyse the attached part views and provide the structured description \
as specified. This will be used as input to a downstream process planning agent.\
"""


def _encode_image(image_path: str) -> str:
    """Return base64-encoded content of an image file."""
    data = Path(image_path).read_bytes()
    return base64.b64encode(data).decode("utf-8")


def _detect_mime(image_path: str) -> str:
    ext = Path(image_path).suffix.lower()
    return {"jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png", "bmp": "image/bmp",
            "webp": "image/webp"}.get(ext.lstrip("."), "image/jpeg")


def analyse_images(image_paths: List[str],
                   api_key: str,
                   model: str = "glm-4v",
                   max_tokens: int = 1024) -> Dict[str, Any]:
    """
    Send part images to GLM-4V and return the analysis.

    Args:
        image_paths : List of local paths to JPG/PNG part view images.
        api_key     : GLM API key (from config).
        model       : Model identifier (default "glm-4v").
        max_tokens  : Maximum response tokens.

    Returns:
        {
          "image_text":   "Full textual description from GLM-4V",
          "feature_hints": ["hint1", "hint2", …]   # parsed bullet points
        }
    """
    if not image_paths:
        raise ValueError("At least one image path must be provided.")

    # Build multimodal message content
    content: List[Dict] = []

    for img_path in image_paths:
        path = Path(img_path)
        if not path.exists():
            logger.warning(f"Image not found, skipping: {img_path}")
            continue
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{_detect_mime(img_path)};base64,{_encode_image(img_path)}"
            }
        })

    content.append({"type": "text", "text": IMAGE_PROMPT})

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": content},
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    logger.info(f"Sending {len(image_paths)} image(s) to GLM-4V …")
    response = requests.post(GLM_API_ENDPOINT, json=payload,
                              headers=headers, timeout=120)
    response.raise_for_status()

    data       = response.json()
    image_text = data["choices"][0]["message"]["content"]
    logger.info("GLM-4V response received.")

    # Parse bullet-point feature hints for downstream agents
    feature_hints = _extract_feature_hints(image_text)

    return {
        "image_text":    image_text,
        "feature_hints": feature_hints,
    }


def _extract_feature_hints(text: str) -> List[str]:
    """Extract bullet-point lines from the LLM description."""
    lines = text.splitlines()
    hints = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("-", "•", "*", "·")) or re.match(r"^\d+\.", stripped):
            hints.append(stripped.lstrip("-•*·").strip())
    return hints


# ── Mock for testing without API access ───────────────────────────────────────

def analyse_images_mock(image_paths: List[str]) -> Dict[str, Any]:
    """
    Returns a canned response matching the paper's aerospace part (Fig. 9).
    Use this when GLM_API_KEY is not yet configured.
    """
    mock_text = """\
1. General Shape: Rectangular block with multiple machined features symmetrically
   distributed along the central axes of the long and short sides.

2. Feature Description:
   - Two through holes: Located along the central axis of the short sides,
     symmetrically positioned on both sides of the central axis.
   - Two rectangular channels: Located on both sides of the central axis of the
     long sides, symmetrically placed. Pass through the entire part length.
   - Four rectangular grooves: Positioned at the right angle formed by the
     intersection of the two rectangular channels. Used for fixing during assembly.
   - Flat reference surface: Main top and bottom faces ensuring tight integration
     with mating components.
   - Chamfers / rounded corners on all external edges.

3. Surface Treatment: Matte
4. Colour: White (likely engineering plastic or light alloy)
5. Other Features: Symmetric layout suggests the part functions as a sliding-fit
   assembly bracket.
"""
    return {
        "image_text":    mock_text,
        "feature_hints": _extract_feature_hints(mock_text),
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from config.config import GLM_API_KEY
    if len(sys.argv) < 2:
        print("Usage: python image_analyzer.py img1.jpg [img2.jpg …]")
        sys.exit(1)
    result = analyse_images(sys.argv[1:], api_key=GLM_API_KEY)
    print(result["image_text"])
