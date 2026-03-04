"""
agents/poea.py — Process Optimization and Evaluation Agent (POEA)

Sub-task: Process Route Merging & Optimization  (paper Section 3.6)

Input  : RM (macro route from MPPA)  +  RS (detailed route from SPPA)
Output : R* — final optimized merged process route

Equation (5) from paper:
    POEA : (RM, RS) → R*

Tools used:
  1. Decision Evaluation and Analysis Tool (tools/decision_evaluator.py)
       → Y = Y0 − α·a − β·b   (Eq. 10)
  2. General Tool / GPT-4o (tools/llm_tool.py)
       → Merges routes, adds auxiliary operations, refines output

Three-stage prompt:
  Stage 1 – Role  : Senior process planner performing merging & evaluation
  Stage 2 – Input : Macro route + detailed route + tool/machine resource info
  Stage 3 – Task  : Generate multiple merged route candidates; evaluate via
                    efficiency model; select best; add auxiliary operations
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple

from tools.llm_tool          import LLMTool
from tools.decision_evaluator import (DecisionEvaluator, ProcessRoute,
                                       ProcessStep, build_route_from_sppa)

logger = logging.getLogger(__name__)


# ── Three-Stage Prompt (from Fig. 8 of paper) ─────────────────────────────────

POEA_MERGE_SYSTEM_PROMPT = """\
You are a senior process planner responsible for MERGING and OPTIMISING a \
machining process route for a CNC-machined precision component.

You will receive:
  1. The MACRO process route (feature sequence and grouping rationale from MPPA)
  2. The DETAILED process route (specific operations, machines, tools per feature from SPPA)

Your task is to generate TWO alternative merged process routes by applying the \
"rough first, fine later" (先粗后精) principle:
  - Group all rough machining operations together (regardless of feature type)
  - Then all semi-finish operations
  - Then all finish operations
  - Finally drilling/boring for holes (after milling is complete)
  - Add essential auxiliary operations: deburring, cleaning, final inspection

Merging rules:
  1. Operations on the SAME machine with the SAME tool should be BATCHED together
  2. Minimise machine changes (α = 8 efficiency penalty each)
  3. Minimise tool changes (β = 3 efficiency penalty each)
  4. Both routes must be feasible and cover ALL features

Output your result as valid JSON with this schema:
{
  "route_1": {
    "description": "<brief strategy description>",
    "steps": [
      {
        "feature": "<feature_type or 'all'>",
        "operation": "<operation name>",
        "machine_name":  "<machine name>",
        "machine_model": "<machine model>",
        "cutter_name":   "<cutter name>",
        "cutter_model":  "<cutter model>",
        "notes":         "<optional>"
      },
      ...
    ]
  },
  "route_2": {
    "description": "<brief strategy description>",
    "steps": [ ... ]
  }
}

Output ONLY the JSON — no preamble, no markdown fences.\
"""

POEA_REFINE_SYSTEM_PROMPT = """\
You are a senior process planner. You have already selected the best merged \
process route based on an efficiency evaluation. Now you must REFINE and \
FINALIZE this route by:

1. Adding any missing auxiliary operations:
   - Deburring after rough machining stages
   - Cleaning between milling and hole operations
   - Final dimensional inspection
   - Surface treatment if specified
2. Ensuring operation descriptions are detailed and professional
3. Verifying logical consistency (no finish before rough, etc.)
4. Numbering all steps sequentially

Output the final process route as valid JSON:
{
  "final_route": [
    {
      "step_number": 1,
      "feature":       "<feature_type or 'all'>",
      "operation":     "<detailed operation name>",
      "machine_name":  "<machine name>",
      "machine_model": "<machine model>",
      "cutter_name":   "<cutter name>",
      "cutter_model":  "<cutter model>",
      "notes":         "<rationale or special instructions>"
    },
    ...
  ],
  "route_summary": "<2–3 sentence summary of the final route strategy>"
}

Output ONLY the JSON.\
"""


def _build_merge_user_prompt(macro_output: Dict, detailed_output: Dict) -> str:
    macro_str    = json.dumps(macro_output.get("macro_sequence",   []), indent=2)
    detailed_str = json.dumps(detailed_output.get("detailed_route", []), indent=2)
    return f"""\
=== MACRO PROCESS ROUTE (from MPPA) ===
{macro_str}

=== DETAILED PROCESS ROUTE (from SPPA) ===
{detailed_str}

=== YOUR TASK ===
Generate TWO alternative merged routes following the "rough first, fine later" \
principle and the merging rules in your system prompt.
Return valid JSON only.\
"""


def _build_refine_user_prompt(best_route: ProcessRoute,
                               efficiency_report: str) -> str:
    steps_str = json.dumps([
        {"feature": s.feature, "operation": s.operation,
         "machine_name": s.machine_name, "machine_model": s.machine_model,
         "cutter_name": s.cutter_name,   "cutter_model":  s.cutter_model}
        for s in best_route.steps
    ], indent=2)
    return f"""\
=== EFFICIENCY EVALUATION REPORT ===
{efficiency_report}

=== SELECTED BEST ROUTE STEPS ===
{steps_str}

=== YOUR TASK ===
Refine and finalize the best route above.
Add any missing auxiliary operations, number all steps, and produce the \
polished final process route JSON as specified in your system prompt.\
"""


# ── POEA Agent class ───────────────────────────────────────────────────────────

class ProcessOptimizationEvaluationAgent:
    """
    Process Optimization and Evaluation Agent (POEA).

    Usage:
        poea = ProcessOptimizationEvaluationAgent(llm_tool)
        result = poea.run(mppa_output, sppa_output)
    """

    def __init__(self,
                 llm_tool:  LLMTool,
                 evaluator: Optional[DecisionEvaluator] = None):
        self.llm       = llm_tool
        self.evaluator = evaluator or DecisionEvaluator()

    def run(self,
            mppa_output: Dict[str, Any],
            sppa_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the POEA sub-task.

        Args:
            mppa_output : Output of MacroProcessPlanningAgent.run()
            sppa_output : Output of SpecificProcessPlanningAgent.run()

        Returns:
            {
              "final_route":        list[dict],   # numbered, refined steps
              "route_summary":      str,
              "efficiency_report":  str,           # evaluation scorecard
              "best_route_id":      str,
              "best_score":         float,
            }
        """
        # ── Step 1: General Tool — generate two merged route candidates ──────
        logger.info("[POEA] Generating merged route candidates …")
        merge_prompt = _build_merge_user_prompt(mppa_output, sppa_output)
        raw_merge    = self.llm.complete(POEA_MERGE_SYSTEM_PROMPT, merge_prompt,
                                          json_mode=True)
        try:
            merged = json.loads(raw_merge)
        except json.JSONDecodeError:
            cleaned = raw_merge.strip().removeprefix("```json").removesuffix("```").strip()
            merged  = json.loads(cleaned)

        # ── Step 2: Decision Evaluation and Analysis Tool ────────────────────
        logger.info("[POEA] Evaluating route candidates with efficiency model …")
        routes: List[ProcessRoute] = []
        for key in ("route_1", "route_2"):
            rd = merged.get(key, {})
            if not rd:
                continue
            route = build_route_from_sppa(
                route_id    = key.upper(),
                description = rd.get("description", key),
                sppa_output = rd.get("steps", [])
            )
            routes.append(route)

        if not routes:
            logger.error("[POEA] No route candidates generated.")
            return {"final_route": [], "route_summary": "No candidates.",
                    "efficiency_report": "", "best_route_id": "",
                    "best_score": 0.0}

        efficiency_report = self.evaluator.report(routes)
        best_route        = self.evaluator.best_route(routes)
        logger.info(f"[POEA] Best route: {best_route.route_id}"
                     f"  (Y={best_route.efficiency_score})")

        # ── Step 3: General Tool — refine and finalize the best route ────────
        logger.info("[POEA] Refining best route with GPT-4o …")
        refine_prompt = _build_refine_user_prompt(best_route, efficiency_report)
        raw_refine    = self.llm.complete(POEA_REFINE_SYSTEM_PROMPT, refine_prompt,
                                           json_mode=True)
        try:
            refined = json.loads(raw_refine)
        except json.JSONDecodeError:
            cleaned = raw_refine.strip().removeprefix("```json").removesuffix("```").strip()
            refined = json.loads(cleaned)

        final_route   = refined.get("final_route",   [])
        route_summary = refined.get("route_summary", "")
        logger.info(f"[POEA] Final route has {len(final_route)} steps.")

        return {
            "final_route":       final_route,
            "route_summary":     route_summary,
            "efficiency_report": efficiency_report,
            "best_route_id":     best_route.route_id,
            "best_score":        best_route.efficiency_score,
        }
