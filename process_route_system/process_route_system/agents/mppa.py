"""
agents/mppa.py — Macro Process Planning Agent (MPPA)

Sub-task: Machining Feature Sorting  (paper Section 3.4)

Input  : F (machining feature set) + D (part description) — from FEA
Output : R = {r1, r2, …, rk} — ordered sequence of machining features

Equation (2) from paper:
    MPPA : (F, D) → R

Tools used:
  - General Tool / GPT-4o only (no external data tools)

Three-stage prompt:
  Stage 1 – Role  : Senior process planner, macro-level route decision-making
  Stage 2 – Input : Feature information + part description from FEA
  Stage 3 – Task  : Determine machining sequence considering:
                    - Machining stage division (rough / semi-finish / finish)
                    - Cost, ease of machining, fixturing constraints
                    - "Base surface first" and "reference surface first" principles
"""

import json
import logging
from typing import List, Dict, Any

from tools.llm_tool import LLMTool

logger = logging.getLogger(__name__)


# ── Three-Stage Prompt (from Fig. 8 of paper) ─────────────────────────────────

MPPA_SYSTEM_PROMPT = """\
You are a senior process planner with 20+ years of experience in CNC machining \
for aerospace precision components. Your expertise covers:
  - Process stage division (rough machining → semi-finish → finish)
  - Fixturing and datum surface selection
  - Optimal sequencing to minimise setup changes and machining distortion
  - Cost-effectiveness analysis of process routes

You will be given a set of machining features extracted from a part drawing. \
Your job is to determine the optimal MACRO process sequence — i.e., the order \
in which the machining features should be processed, WITHOUT yet selecting \
specific machines, tools, or process chains.

Key sequencing principles to apply:
  1. Basic shape / reference surfaces first (ensures stable fixturing)
  2. Through holes before slots/pockets when they serve as positioning references
  3. Roughing for all features before semi-finishing (grouped by stage)
  4. Features on the same setup/face should be grouped when possible
  5. Surface treatment / deburring / chamfering last
  6. Symmetric features can often be batched in a single setup

Output your result as valid JSON with this exact schema:
{
  "macro_sequence": [
    {
      "step": 1,
      "feature_id": <int>,
      "feature_type": "<string>",
      "reason": "<concise justification for this position in the sequence>"
    },
    ...
  ],
  "sequence_notes": "<any overall comments about the macro strategy>"
}

Output ONLY the JSON — no preamble, no markdown fences.\
"""


def _build_mppa_user_prompt(features: List[Dict],
                             part_description: str) -> str:
    """Compose the user message for MPPA."""
    feature_block = json.dumps(features, indent=2)
    return f"""\
=== PART DESCRIPTION (from FEA) ===
{part_description}

=== MACHINING FEATURES (from FEA) ===
{feature_block}

=== YOUR TASK ===
Analyse the part description and the machining feature list above.
Determine the macro machining sequence following the principles in your \
system prompt.

For each feature, output:
  - The step number in the planned sequence
  - The feature_id and feature_type
  - A concise reason for its position

Return valid JSON only.\
"""


# ── MPPA Agent class ───────────────────────────────────────────────────────────

class MacroProcessPlanningAgent:
    """
    Macro Process Planning Agent (MPPA).

    Usage:
        mppa = MacroProcessPlanningAgent(llm_tool)
        result = mppa.run(fea_output)
        # result = {"macro_sequence": [...], "sequence_notes": "..."}
    """

    def __init__(self, llm_tool: LLMTool):
        self.llm = llm_tool

    def run(self, fea_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the MPPA sub-task.

        Args:
            fea_output : Dict returned by FeatureExtractionAgent.run()
                         Must contain 'features' and 'part_description'.

        Returns:
            {
              "macro_sequence":  list[dict],   # ordered feature steps
              "sequence_notes":  str,
              "ordered_features": list[dict],  # feature dicts in planned order
            }
        """
        features         = fea_output.get("features", [])
        part_description = fea_output.get("part_description", "")

        if not features:
            logger.warning("[MPPA] No features received from FEA.")
            return {"macro_sequence": [], "sequence_notes": "No features to sequence.",
                    "ordered_features": []}

        logger.info(f"[MPPA] Sequencing {len(features)} features …")
        user_prompt = _build_mppa_user_prompt(features, part_description)

        raw_reply = self.llm.complete(MPPA_SYSTEM_PROMPT, user_prompt,
                                       json_mode=True)
        try:
            parsed = json.loads(raw_reply)
        except json.JSONDecodeError:
            cleaned = raw_reply.strip().removeprefix("```json").removesuffix("```").strip()
            parsed  = json.loads(cleaned)

        macro_sequence = parsed.get("macro_sequence", [])
        logger.info(f"[MPPA] Macro sequence has {len(macro_sequence)} steps.")

        # Build ordered feature list for downstream agents
        feature_map = {f["feature_id"]: f for f in features}
        ordered_features = []
        for step in macro_sequence:
            fid = step.get("feature_id")
            if fid in feature_map:
                ordered_features.append({
                    **feature_map[fid],
                    "macro_step":   step["step"],
                    "macro_reason": step.get("reason", ""),
                })

        return {
            "macro_sequence":   macro_sequence,
            "sequence_notes":   parsed.get("sequence_notes", ""),
            "ordered_features": ordered_features,
        }
