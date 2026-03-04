"""
agents/fea.py — Feature Extraction Agent (FEA)

Sub-task: Machining Feature Recognition  (paper Section 3.3)

Input  : Part STP file  +  multi-view JPG images
Output : F (machining feature set)  +  D (part description)

Equation (1) from paper:
    FEA : (S, I) → (F, D)

Tools used:
  1. STP File Analysis Tool  (tools/stp_parser.py)
  2. JPG Image Analysis Tool (tools/image_analyzer.py)
  3. General Tool / GPT-4o   (tools/llm_tool.py)

Three-stage prompt:
  Stage 1 – Role:  professional engineer, file parsing & data analysis
  Stage 2 – Input: STP geometry + image analysis results
  Stage 3 – Task:  extract machining features, produce part description
"""

import json
import logging
from typing import List, Dict, Any, Optional

from tools.stp_parser      import parse_stp_file
from tools.image_analyzer  import analyse_images, analyse_images_mock
from tools.llm_tool        import LLMTool

logger = logging.getLogger(__name__)


# ── Three-Stage Prompt  (from Fig. 8 of paper) ────────────────────────────────

FEA_SYSTEM_PROMPT = """\
You are a professional engineer specialising in file parsing and data analysis, \
with deep expertise in STP file parsing, CNC machining feature recognition, and \
image analysis. Your knowledge covers geometric feature taxonomy, manufacturing \
tolerances, and process planning conventions.

Using the provided tools, complete the following THREE tasks:

(1) Parse the STP file data: the geometric analysis has already been performed \
    and the face/vertex/normal vector data is provided to you. Interpret and \
    summarise the 3D geometry.

(2) Analyse the multi-view images: the image analysis has already been performed \
    by GLM-4V and the textual description is provided. Integrate it with the \
    geometric data.

(3) Produce a comprehensive, unified machining feature summary that includes:
    - A brief overall part description (shape, material context, dimensions if inferable)
    - A numbered list of all identified machining features, each with:
        * feature_type (through_hole | blind_hole | rectangular_through_slot |
                        rectangular_pocket | flat_surface | chamfer | fillet)
        * feature_id (integer)
        * face_ids (list of face IDs from STP analysis)
        * description (location, symmetry, estimated size)

Output the final result as valid JSON with this exact schema:
{
  "part_description": "<overall textual description>",
  "features": [
    {
      "feature_id": 0,
      "feature_type": "through_hole",
      "face_ids": [10, 11],
      "description": "..."
    },
    ...
  ]
}
Output ONLY the JSON — no preamble, no markdown fences.\
"""


def _build_fea_user_prompt(stp_result: Dict, image_result: Dict) -> str:
    """Compose the user message with STP + image data."""
    stp_summary  = stp_result.get("summary", "STP parsing produced no summary.")
    image_text   = image_result.get("image_text", "No image analysis available.")

    # Truncate large face lists to avoid context overflow
    faces = stp_result.get("faces", [])
    if len(faces) > 50:
        face_snippet = json.dumps(faces[:50], indent=2) + f"\n... ({len(faces)} faces total)"
    else:
        face_snippet = json.dumps(faces, indent=2)

    raw_features = stp_result.get("features", [])

    return f"""\
=== STP FILE ANALYSIS RESULTS ===
{stp_summary}

Raw face data (first 50 of {len(faces)} faces):
{face_snippet}

Rule-based feature candidates detected:
{json.dumps(raw_features, indent=2)}

=== IMAGE ANALYSIS RESULTS (GLM-4V) ===
{image_text}

=== YOUR TASK ===
Integrate the STP geometry and image description above.
Produce the unified machining feature JSON as specified in your system prompt.\
"""


# ── FEA Agent class ────────────────────────────────────────────────────────────

class FeatureExtractionAgent:
    """
    Feature Extraction Agent (FEA).

    Usage:
        fea = FeatureExtractionAgent(llm_tool, openai_api_key, glm_api_key)
        result = fea.run(stp_path="part.stp", image_paths=["front.jpg","iso.jpg"])
        # result = {"part_description": "...", "features": [...]}
    """

    def __init__(self,
                 llm_tool:      LLMTool,
                 glm_api_key:   str = "",
                 use_mock_images: bool = False,
                 use_neo4j:     bool = False):
        """
        Args:
            llm_tool       : Configured LLMTool instance (GPT-4o).
            glm_api_key    : API key for GLM-4V image analysis.
            use_mock_images: If True, use canned image analysis response
                             (for testing without GLM API access).
        """
        self.llm           = llm_tool
        self.glm_api_key   = glm_api_key
        self.use_mock_images = use_mock_images

    def run(self,
            stp_path:    Optional[str] = None,
            image_paths: Optional[List[str]] = None,
            stp_result:  Optional[Dict] = None,
            image_result: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute the FEA sub-task.

        You may either supply file paths (stp_path / image_paths) for the tools
        to process, or supply pre-computed results directly (stp_result /
        image_result) to skip the tool calls.

        Returns:
            {
              "part_description": str,
              "features":         list[dict],
              "raw_stp":          dict,   # raw STP parser output
              "raw_image":        dict,   # raw image analysis output
            }
        """
        # ── Step 1: STP File Analysis Tool ──────────────────────────────────
        if stp_result is None:
            if stp_path is None:
                # Demo/mock mode: use a canned STP summary
                logger.info("[FEA] No STP file provided — using mock STP result.")
                stp_result = {
                    "faces": [],
                    "features": [
                        {"feature_id": 0, "feature_type": "flat_surface",
                         "face_ids": [0,1,2,3], "bottom_face_id": 0,
                         "description": "Main flat reference surface"},
                        {"feature_id": 1, "feature_type": "through_hole",
                         "face_ids": [10,11], "bottom_face_id": 10,
                         "description": "Through hole, left"},
                        {"feature_id": 2, "feature_type": "through_hole",
                         "face_ids": [10,11], "bottom_face_id": 10,
                         "description": "Through hole, right"},
                        {"feature_id": 3, "feature_type": "rectangular_through_slot",
                         "face_ids": [13,14,15], "bottom_face_id": 14,
                         "description": "Rectangular slot, left long side"},
                        {"feature_id": 4, "feature_type": "rectangular_through_slot",
                         "face_ids": [13,14,15], "bottom_face_id": 14,
                         "description": "Rectangular slot, right long side"},
                        {"feature_id": 5, "feature_type": "rectangular_pocket",
                         "face_ids": list(range(16,32)), "bottom_face_id": 18,
                         "description": "Four rectangular pockets at slot intersections"},
                    ],
                    "summary": (
                        "This part has 32 faces.\n"
                        "This part has 4 feature sets: "
                        "flat_surface, through_hole×2, rectangular_through_slot×2, "
                        "rectangular_pocket×4."
                    ),
                }
            else:
                logger.info(f"[FEA] Running STP File Analysis Tool on: {stp_path}")
                stp_result = parse_stp_file(stp_path, use_gnn=False)
        else:
            logger.info("[FEA] Using pre-computed STP result.")

        # ── Step 2: JPG Image Analysis Tool ─────────────────────────────────
        if image_result is None:
            if self.use_mock_images or not image_paths:
                logger.info("[FEA] Using mock image analysis (no GLM-4V call).")
                image_result = analyse_images_mock(image_paths or [])
            else:
                logger.info(f"[FEA] Running JPG Image Analysis Tool on "
                             f"{len(image_paths)} images …")
                image_result = analyse_images(image_paths,
                                               api_key=self.glm_api_key)
        else:
            logger.info("[FEA] Using pre-computed image result.")

        # ── Step 3: General Tool (GPT-4o) — integrate and summarise ─────────
        logger.info("[FEA] Running General Tool (GPT-4o) for feature integration …")
        user_prompt = _build_fea_user_prompt(stp_result, image_result)

        raw_reply = self.llm.complete(FEA_SYSTEM_PROMPT, user_prompt,
                                       json_mode=True)
        try:
            parsed = json.loads(raw_reply)
        except json.JSONDecodeError:
            cleaned = raw_reply.strip().removeprefix("```json").removesuffix("```").strip()
            parsed  = json.loads(cleaned)

        logger.info(f"[FEA] Identified {len(parsed.get('features', []))} features.")

        return {
            "part_description": parsed.get("part_description", ""),
            "features":         parsed.get("features", []),
            "raw_stp":          stp_result,
            "raw_image":        image_result,
        }
