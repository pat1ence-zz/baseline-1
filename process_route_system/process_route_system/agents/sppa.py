"""
agents/sppa.py — Specific Process Planning Agent (SPPA)

Sub-task: Machining Feature Process Chain & Resource Selection  (paper Section 3.5)

Input  : F (features from FEA) + RM (macro sequence from MPPA) + RE (part requirements)
Output : RS — detailed process route with process chains and machining resources

Equation (3) from paper:
    SPPA : (F, RM, RE) → RS

Tools used:
  1. Process Knowledge Graph Query Tool (tools/knowledge_graph.py) — RAG
  2. General Tool / GPT-4o             (tools/llm_tool.py)

Three-stage prompt:
  Stage 1 – Role  : Senior process planner focused on detailed process plans
  Stage 2 – Input : STP analysis + machining requirements + macro route + KG triplets
  Stage 3 – Task  : Select process chains from KG; specify machines and tools per step
"""

import json
import logging
from typing import List, Dict, Any, Optional

from tools.llm_tool       import LLMTool
from tools.knowledge_graph import InMemoryKnowledgeGraph, get_knowledge_graph

logger = logging.getLogger(__name__)


# ── Three-Stage Prompt (from Fig. 8 of paper) ─────────────────────────────────

SPPA_SYSTEM_PROMPT = """\
You are a senior process planner specialising in DETAILED process route design \
for CNC-machined aerospace components. Your knowledge spans:
  - Selection of appropriate process chains for specific machining features
  - Machine tool and cutting tool specification
  - Retrieval-Augmented Generation from process knowledge graphs
  - Tolerance stack-up and surface finish requirements

You will receive:
  1. The part's machining feature information (from FEA)
  2. The macro-level feature processing sequence (from MPPA)
  3. The part's machining precision requirements (Ra, IT grade)
  4. Relevant process chain triplets retrieved from the process knowledge graph

Your task is to produce a DETAILED process route by:
  - Mapping each machining feature to its optimal process chain (from the KG data)
  - Specifying the exact machine and cutting tool for EACH operation in the chain
  - Following the macro sequence order from MPPA
  - Ensuring Ra and IT grade requirements are satisfied

Output your result as valid JSON with this exact schema:
{
  "detailed_route": [
    {
      "feature_id": <int>,
      "feature_type": "<string>",
      "macro_step": <int>,
      "chain_id": "<string>",
      "operations": [
        {
          "operation": "<e.g. Rough Milling>",
          "machine_name":  "<e.g. Gantry Milling Machine>",
          "machine_model": "<e.g. Doosan BM2740>",
          "cutter_name":   "<e.g. Carbide End Mill>",
          "cutter_model":  "<e.g. Mitsubishi APX4000>"
        },
        ...
      ]
    },
    ...
  ],
  "notes": "<any overall comments on the detailed plan>"
}

Output ONLY the JSON — no preamble, no markdown fences.\
"""


def _build_sppa_user_prompt(ordered_features: List[Dict],
                             part_description: str,
                             machining_requirements: Dict,
                             kg_context: str) -> str:
    """Compose the user message for SPPA."""
    req_str = json.dumps(machining_requirements, indent=2)
    feat_str = json.dumps(ordered_features, indent=2)

    return f"""\
=== PART DESCRIPTION ===
{part_description}

=== MACHINING REQUIREMENTS ===
{req_str}

=== MACRO FEATURE SEQUENCE (from MPPA) ===
{feat_str}

=== PROCESS KNOWLEDGE GRAPH — RETRIEVED CHAINS ===
{kg_context}

=== YOUR TASK ===
Using the knowledge graph process chains above as your primary reference, \
produce the detailed process route for each machining feature in the macro \
sequence order.

For each feature:
  1. Select the most appropriate process chain from the KG data (note its chain_id)
  2. List every operation in that chain with its specific machine and cutting tool
  3. Ensure the Ra and IT grade requirements from the machining requirements are met

Return valid JSON only.\
"""


# ── SPPA Agent class ───────────────────────────────────────────────────────────

class SpecificProcessPlanningAgent:
    """
    Specific Process Planning Agent (SPPA).

    Usage:
        sppa = SpecificProcessPlanningAgent(llm_tool, kg)
        result = sppa.run(fea_output, mppa_output, machining_requirements)
    """

    def __init__(self,
                 llm_tool: LLMTool,
                 knowledge_graph=None,
                 use_neo4j: bool = False,
                 neo4j_uri: str = "",
                 neo4j_user: str = "",
                 neo4j_password: str = ""):
        self.llm = llm_tool

        # Initialise knowledge graph backend
        if knowledge_graph is not None:
            self.kg = knowledge_graph
        elif use_neo4j:
            self.kg = get_knowledge_graph(
                use_neo4j=True, uri=neo4j_uri,
                user=neo4j_user, password=neo4j_password)
        else:
            self.kg = InMemoryKnowledgeGraph()
            logger.info("[SPPA] Using in-memory knowledge graph (no Neo4j).")

    def run(self,
            fea_output:  Dict[str, Any],
            mppa_output: Dict[str, Any],
            machining_requirements: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute the SPPA sub-task.

        Args:
            fea_output   : Output of FeatureExtractionAgent.run()
            mppa_output  : Output of MacroProcessPlanningAgent.run()
            machining_requirements : Dict with keys:
                             'Ra'  (e.g. "0.8-1.6 μm"),
                             'IT'  (e.g. "IT7-IT8"),
                             'material' (e.g. "15CrMo"),
                             'production_volume' (e.g. "mass production")

        Returns:
            {
              "detailed_route": list[dict],
              "notes":          str,
              "kg_context":     str,   # raw KG retrieval text (for audit)
            }
        """
        features         = fea_output.get("features", [])
        part_description = fea_output.get("part_description", "")
        ordered_features = mppa_output.get("ordered_features", features)

        if machining_requirements is None:
            machining_requirements = {
                "Ra":               "0.8-1.6 μm",
                "IT":               "IT7-IT8",
                "material":         "15CrMo (chromium-molybdenum alloy steel)",
                "production_volume": "mass production",
                "notes":            "Medium-precision assembly system bracket"
            }

        ra  = machining_requirements.get("Ra",  "0.8-1.6")
        it  = machining_requirements.get("IT",  "IT7-IT8")

        # ── Step 1: Knowledge Graph Query Tool ───────────────────────────────
        logger.info("[SPPA] Querying process knowledge graph …")
        kg_results  = self.kg.query_all_features(ordered_features, ra=ra,
                                                   dim_accuracy=it)
        kg_context  = self.kg.format_for_agent(kg_results)
        logger.info(f"[SPPA] KG retrieved {len(kg_results)} chain(s).")

        # ── Step 2: General Tool (GPT-4o) — generate detailed route ─────────
        logger.info("[SPPA] Running General Tool (GPT-4o) for detailed route …")
        user_prompt = _build_sppa_user_prompt(ordered_features,
                                               part_description,
                                               machining_requirements,
                                               kg_context)

        raw_reply = self.llm.complete(SPPA_SYSTEM_PROMPT, user_prompt,
                                       json_mode=True)
        try:
            parsed = json.loads(raw_reply)
        except json.JSONDecodeError:
            cleaned = raw_reply.strip().removeprefix("```json").removesuffix("```").strip()
            parsed  = json.loads(cleaned)

        detailed_route = parsed.get("detailed_route", [])
        logger.info(f"[SPPA] Detailed route has {len(detailed_route)} feature entries.")

        return {
            "detailed_route": detailed_route,
            "notes":          parsed.get("notes", ""),
            "kg_context":     kg_context,
        }
