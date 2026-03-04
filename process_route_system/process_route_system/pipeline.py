"""
pipeline.py — Multi-Agent Process Route Generation Pipeline

Integrates all four agents in sequence:
    FEA → MPPA → SPPA → POEA

Then runs TOPSIS evaluation comparing the generated route with the
typical historical process route from the paper.

Usage:
    # With real APIs:
    python pipeline.py --stp part.stp --images front.jpg iso.jpg \
                       --openai-key sk-... --glm-key ...

    # Demo mode (mock LLM + in-memory KG):
    python pipeline.py --demo

    # TOPSIS-only (reproduces paper Tables 7 & 8):
    python pipeline.py --topsis-only
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config.config import (OPENAI_API_KEY, GLM_API_KEY, OPENAI_MODEL,
                             NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
from tools.llm_tool          import LLMTool
from tools.decision_evaluator import DecisionEvaluator
from agents.fea              import FeatureExtractionAgent
from agents.mppa             import MacroProcessPlanningAgent
from agents.sppa             import SpecificProcessPlanningAgent
from agents.poea             import ProcessOptimizationEvaluationAgent
from evaluation.topsis       import TopsisEvaluator, PAPER_DATA

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ── Mock LLM for demo mode ────────────────────────────────────────────────────

class MockLLMTool:
    """
    Deterministic mock that returns canned responses matching the paper's
    aerospace case study (Fig. 9–12).  No API key required.
    """
    model = "mock-gpt-4o"

    def complete(self, system_prompt: str, user_prompt: str,
                  json_mode: bool = False) -> str:
        # Detect which agent — check most-specific prompts first
        if "REFINE and FINALIZE" in system_prompt:
            return self._poea_refine_response()
        elif "TWO alternative merged" in system_prompt:
            return self._poea_merge_response()
        elif "THREE tasks" in system_prompt:
            return self._fea_response()
        elif "MACRO process sequence" in system_prompt:
            return self._mppa_response()
        elif "DETAILED process route" in system_prompt:
            return self._sppa_response()
        return json.dumps({"message": "mock response"})

    def complete_json(self, system_prompt: str, user_prompt: str) -> Dict:
        return json.loads(self.complete(system_prompt, user_prompt, json_mode=True))

    # ── Canned responses (paper Figures 9–12) ─────────────────────────────────

    def _fea_response(self) -> str:
        return json.dumps({
            "part_description": (
                "The part is made of 15CrMo chromium-molybdenum alloy steel and has "
                "an overall rectangular shape. It features several characteristics: "
                "General Shape: Rectangular. Two through holes located along the "
                "central axis of the short sides, symmetrically positioned on both "
                "sides of the central axis. Two rectangular channels located on both "
                "sides of the central axis of the long sides, symmetrically placed. "
                "Four rectangular grooves positioned at the right angle formed by the "
                "intersection of the two rectangular channels. A flat reference surface "
                "ensuring tight integration with other components. Surface Treatment: "
                "Matte. Colour: White."
            ),
            "features": [
                {"feature_id": 0, "feature_type": "flat_surface",
                 "face_ids": [0, 1, 2, 3],
                 "description": "Main reference flat surface (top/bottom), area ≈ large"},
                {"feature_id": 1, "feature_type": "through_hole",
                 "face_ids": [10, 11],
                 "description": "Through hole #1, central axis short side, left"},
                {"feature_id": 2, "feature_type": "through_hole",
                 "face_ids": [10, 11],
                 "description": "Through hole #2, central axis short side, right"},
                {"feature_id": 3, "feature_type": "rectangular_through_slot",
                 "face_ids": [13, 14, 15],
                 "description": "Rectangular slot #1, long side left"},
                {"feature_id": 4, "feature_type": "rectangular_through_slot",
                 "face_ids": [13, 14, 15],
                 "description": "Rectangular slot #2, long side right"},
                {"feature_id": 5, "feature_type": "rectangular_pocket",
                 "face_ids": [16,17,18,19,20,21,26,27,28,29,30,31],
                 "description": "Rectangular pocket — 4 pockets at slot intersections"},
            ]
        })

    def _mppa_response(self) -> str:
        return json.dumps({
            "macro_sequence": [
                {"step": 1, "feature_id": 0, "feature_type": "flat_surface",
                 "reason": "Machine the basic shape first — provides stable reference for all subsequent operations."},
                {"step": 2, "feature_id": 1, "feature_type": "through_hole",
                 "reason": "Through holes are simpler features; drill after reference surface; can reuse same drill bit."},
                {"step": 3, "feature_id": 2, "feature_type": "through_hole",
                 "reason": "Symmetric hole — process in same setup as hole #1."},
                {"step": 4, "feature_id": 3, "feature_type": "rectangular_through_slot",
                 "reason": "Slots pass through entire part; machine after through holes (holes serve as datum)."},
                {"step": 5, "feature_id": 4, "feature_type": "rectangular_through_slot",
                 "reason": "Symmetric slot — process symmetrically in same setup as slot #1."},
                {"step": 6, "feature_id": 5, "feature_type": "rectangular_pocket",
                 "reason": "Pockets at slot intersections require slots to be finished first as reference."},
            ],
            "sequence_notes": (
                "Basic shape first, then holes, then slots, then pockets. "
                "Surface treatment applied last. Deburring after rough stages."
            )
        })

    def _sppa_response(self) -> str:
        return json.dumps({
            "detailed_route": [
                {"feature_id": 0, "feature_type": "flat_surface",
                 "macro_step": 1, "chain_id": "PC_FLAT_1",
                 "operations": [
                     {"operation": "Rough Milling",
                      "machine_name": "Gantry Milling Machine", "machine_model": "Doosan BM2045",
                      "cutter_name": "Carbide End Mill",          "cutter_model": "Kyocera PR1535"},
                     {"operation": "Semi-Finishing Milling",
                      "machine_name": "Horizontal Milling Machine", "machine_model": "OKUMA MB-5000H",
                      "cutter_name": "High-Speed Steel End Mill",   "cutter_model": "Kennametal KSEM"},
                     {"operation": "Finishing Milling",
                      "machine_name": "CNC Milling Machine",        "machine_model": "DMG MORI NHX 5000",
                      "cutter_name": "Tungsten Carbide End Mill",   "cutter_model": "Sandvik CoroMill 316"},
                 ]},
                {"feature_id": 1, "feature_type": "through_hole",
                 "macro_step": 2, "chain_id": "PC_HOLE_4",
                 "operations": [
                     {"operation": "Drilling (Reaming)",
                      "machine_name": "Radial Drill",  "machine_model": "Z3040",
                      "cutter_name": "Drill Bit",       "cutter_model": "Kyocera KSD30"},
                     {"operation": "Precision Boring",
                      "machine_name": "Boring Machine", "machine_model": "TPX6111B",
                      "cutter_name": "Carbide Precision Boring Tool",
                      "cutter_model": "Mitsubishi BORING"},
                 ]},
                {"feature_id": 3, "feature_type": "rectangular_through_slot",
                 "macro_step": 4, "chain_id": "PC_SLOT_1",
                 "operations": [
                     {"operation": "Rough Milling",
                      "machine_name": "Gantry Milling Machine",    "machine_model": "Doosan BM2740",
                      "cutter_name": "Carbide End Mill",            "cutter_model": "Mitsubishi APX4000"},
                     {"operation": "Semi-Finishing Milling",
                      "machine_name": "Horizontal Milling Machine", "machine_model": "OKUMA MB-5000H",
                      "cutter_name": "High-Speed Steel End Mill",   "cutter_model": "Kyocera MEGACOAT"},
                     {"operation": "Finishing Milling",
                      "machine_name": "CNC Milling Machine",        "machine_model": "DMG MORI NHX 5000",
                      "cutter_name": "Tungsten Carbide End Mill",   "cutter_model": "Sandvik CoroMill 316"},
                     {"operation": "Ultra-Fine Milling",
                      "machine_name": "Ultra-Precision CNC Milling Machine", "machine_model": "Makino F5",
                      "cutter_name": "Ceramic Tool",                "cutter_model": "Kyocera PV720"},
                 ]},
                {"feature_id": 5, "feature_type": "rectangular_pocket",
                 "macro_step": 6, "chain_id": "PC_POCKET_1",
                 "operations": [
                     {"operation": "Rough Milling",
                      "machine_name": "Gantry Milling Machine",    "machine_model": "Doosan BM2740",
                      "cutter_name": "Carbide End Mill",            "cutter_model": "Mitsubishi APX4000"},
                     {"operation": "Semi-Finishing Milling",
                      "machine_name": "Horizontal Milling Machine", "machine_model": "OKUMA MB-5000H",
                      "cutter_name": "High-Speed Steel End Mill",   "cutter_model": "Kyocera MEGACOAT"},
                     {"operation": "Finishing Milling",
                      "machine_name": "CNC Milling Machine",        "machine_model": "DMG MORI NHX 5000",
                      "cutter_name": "Tungsten Carbide End Mill",   "cutter_model": "Sandvik CoroMill 316"},
                     {"operation": "Ultra-Fine Milling",
                      "machine_name": "Ultra-Precision CNC Milling Machine", "machine_model": "Makino F5",
                      "cutter_name": "Ceramic Tool",                "cutter_model": "Kyocera PV720"},
                 ]},
            ],
            "notes": ("Process chains from knowledge graph. Chains selected for Ra=0.8-1.6 μm, "
                      "IT7-IT8 requirements.")
        })

    def _poea_merge_response(self) -> str:
        return json.dumps({
            "route_1": {
                "description": "Grouped by machining stage (rough→semi-finish→finish→hole)",
                "steps": [
                    # Rough phase (Doosan BM2045/BM2740)
                    {"feature": "flat_surface",            "operation": "Rough Milling",
                     "machine_name": "Gantry Milling Machine",  "machine_model": "Doosan BM2045",
                     "cutter_name": "Carbide End Mill",     "cutter_model": "Kyocera PR1535"},
                    {"feature": "rectangular_through_slot","operation": "Rough Milling",
                     "machine_name": "Gantry Milling Machine",  "machine_model": "Doosan BM2740",
                     "cutter_name": "Carbide End Mill",     "cutter_model": "Mitsubishi APX4000"},
                    {"feature": "rectangular_pocket",      "operation": "Rough Milling",
                     "machine_name": "Gantry Milling Machine",  "machine_model": "Doosan BM2740",
                     "cutter_name": "Carbide End Mill",     "cutter_model": "Mitsubishi APX4000"},
                    {"feature": "all",                     "operation": "Deburring",
                     "machine_name": "Manual",              "machine_model": "Manual",
                     "cutter_name": "Deburring Tool",       "cutter_model": "Standard"},
                    # Semi-finish phase (OKUMA MB-5000H)
                    {"feature": "flat_surface",            "operation": "Semi-Finishing Milling",
                     "machine_name": "Horizontal Milling Machine","machine_model": "OKUMA MB-5000H",
                     "cutter_name": "HSS End Mill",         "cutter_model": "Kennametal KSEM"},
                    {"feature": "rectangular_through_slot","operation": "Semi-Finishing Milling",
                     "machine_name": "Horizontal Milling Machine","machine_model": "OKUMA MB-5000H",
                     "cutter_name": "HSS End Mill",         "cutter_model": "Kyocera MEGACOAT"},
                    {"feature": "rectangular_pocket",      "operation": "Semi-Finishing Milling",
                     "machine_name": "Horizontal Milling Machine","machine_model": "OKUMA MB-5000H",
                     "cutter_name": "HSS End Mill",         "cutter_model": "Kyocera MEGACOAT"},
                    {"feature": "all",                     "operation": "Cleaning",
                     "machine_name": "Manual",              "machine_model": "Manual",
                     "cutter_name": "Cleaning Tool",        "cutter_model": "Standard"},
                    # Finish phase (DMG MORI NHX 5000)
                    {"feature": "flat_surface",            "operation": "Finishing Milling",
                     "machine_name": "CNC Milling Machine", "machine_model": "DMG MORI NHX 5000",
                     "cutter_name": "Tungsten Carbide End Mill","cutter_model": "Sandvik CoroMill 316"},
                    {"feature": "rectangular_through_slot","operation": "Finishing Milling",
                     "machine_name": "CNC Milling Machine", "machine_model": "DMG MORI NHX 5000",
                     "cutter_name": "Tungsten Carbide End Mill","cutter_model": "Sandvik CoroMill 316"},
                    {"feature": "rectangular_pocket",      "operation": "Finishing Milling",
                     "machine_name": "CNC Milling Machine", "machine_model": "DMG MORI NHX 5000",
                     "cutter_name": "Tungsten Carbide End Mill","cutter_model": "Sandvik CoroMill 316"},
                    # Holes
                    {"feature": "through_hole",            "operation": "Drilling (Reaming)",
                     "machine_name": "Radial Drill",        "machine_model": "Z3040",
                     "cutter_name": "Drill Bit",            "cutter_model": "Kyocera KSD30"},
                    {"feature": "through_hole",            "operation": "Precision Boring",
                     "machine_name": "Boring Machine",      "machine_model": "TPX6111B",
                     "cutter_name": "Carbide Precision Boring Tool","cutter_model": "Mitsubishi BORING"},
                    {"feature": "all",                     "operation": "Final Inspection",
                     "machine_name": "CMM",                 "machine_model": "Zeiss CONTURA",
                     "cutter_name": "Probe",                "cutter_model": "Renishaw"},
                ]
            },
            "route_2": {
                "description": "Feature-by-feature (flat→slot→pocket→hole)",
                "steps": [
                    {"feature": "flat_surface","operation": "Rough Milling",
                     "machine_name": "Gantry Milling Machine","machine_model": "Doosan BM2045",
                     "cutter_name": "Carbide End Mill","cutter_model": "Kyocera PR1535"},
                    {"feature": "flat_surface","operation": "Semi-Finishing Milling",
                     "machine_name": "Horizontal Milling Machine","machine_model": "OKUMA MB-5000H",
                     "cutter_name": "HSS End Mill","cutter_model": "Kennametal KSEM"},
                    {"feature": "flat_surface","operation": "Finishing Milling",
                     "machine_name": "CNC Milling Machine","machine_model": "DMG MORI NHX 5000",
                     "cutter_name": "Tungsten Carbide End Mill","cutter_model": "Sandvik CoroMill 316"},
                    {"feature": "rectangular_through_slot","operation": "Rough Milling",
                     "machine_name": "Gantry Milling Machine","machine_model": "Doosan BM2740",
                     "cutter_name": "Carbide End Mill","cutter_model": "Mitsubishi APX4000"},
                    {"feature": "rectangular_through_slot","operation": "Semi-Finishing Milling",
                     "machine_name": "Horizontal Milling Machine","machine_model": "OKUMA MB-5000H",
                     "cutter_name": "HSS End Mill","cutter_model": "Kyocera MEGACOAT"},
                    {"feature": "rectangular_through_slot","operation": "Finishing Milling",
                     "machine_name": "CNC Milling Machine","machine_model": "DMG MORI NHX 5000",
                     "cutter_name": "Tungsten Carbide End Mill","cutter_model": "Sandvik CoroMill 316"},
                    {"feature": "rectangular_pocket","operation": "Rough Milling",
                     "machine_name": "Gantry Milling Machine","machine_model": "Doosan BM2740",
                     "cutter_name": "Carbide End Mill","cutter_model": "Mitsubishi APX4000"},
                    {"feature": "rectangular_pocket","operation": "Finishing Milling",
                     "machine_name": "CNC Milling Machine","machine_model": "DMG MORI NHX 5000",
                     "cutter_name": "Tungsten Carbide End Mill","cutter_model": "Sandvik CoroMill 316"},
                    {"feature": "through_hole","operation": "Drilling (Reaming)",
                     "machine_name": "Radial Drill","machine_model": "Z3040",
                     "cutter_name": "Drill Bit","cutter_model": "Kyocera KSD30"},
                    {"feature": "through_hole","operation": "Precision Boring",
                     "machine_name": "Boring Machine","machine_model": "TPX6111B",
                     "cutter_name": "Carbide Precision Boring Tool","cutter_model": "Mitsubishi BORING"},
                ]
            }
        })

    def _poea_refine_response(self) -> str:
        return json.dumps({
            "final_route": [
                {"step_number":  1, "feature": "flat_surface",
                 "operation": "Processing the Basic Shape of the Workpiece - Rough Milling",
                 "machine_name": "Gantry Milling Machine", "machine_model": "Doosan BM2045",
                 "cutter_name": "Carbide End Mill",         "cutter_model": "Kyocera PR1535",
                 "notes": "Establish reference surface; remove bulk material quickly."},
                {"step_number":  2, "feature": "rectangular_through_slot",
                 "operation": "Processing Two Rectangular Channels - Rough Milling",
                 "machine_name": "Gantry Milling Machine", "machine_model": "Doosan BM2740",
                 "cutter_name": "Carbide End Mill",         "cutter_model": "Mitsubishi APX4000",
                 "notes": "Both slots in one setup — symmetric clamping."},
                {"step_number":  3, "feature": "rectangular_pocket",
                 "operation": "Processing Four Rectangular Pockets - Rough Milling",
                 "machine_name": "Gantry Milling Machine", "machine_model": "Doosan BM2740",
                 "cutter_name": "Carbide End Mill",         "cutter_model": "Mitsubishi APX4000",
                 "notes": "All four pockets in one setup."},
                {"step_number":  4, "feature": "all",
                 "operation": "Deburring",
                 "machine_name": "Manual", "machine_model": "Manual",
                 "cutter_name": "Deburring Tool", "cutter_model": "Standard",
                 "notes": "Remove burrs from rough milling before semi-finish."},
                {"step_number":  5, "feature": "flat_surface",
                 "operation": "Processing the Basic Shape - Semi-Finishing Milling",
                 "machine_name": "Horizontal Milling Machine", "machine_model": "OKUMA MB-5000H",
                 "cutter_name": "High-Speed Steel End Mill",    "cutter_model": "Kennametal KSEM",
                 "notes": "Achieve IT9 accuracy on reference surface."},
                {"step_number":  6, "feature": "rectangular_through_slot",
                 "operation": "Processing Two Rectangular Channels - Semi-Finishing Milling",
                 "machine_name": "Horizontal Milling Machine", "machine_model": "OKUMA MB-5000H",
                 "cutter_name": "High-Speed Steel End Mill",    "cutter_model": "Kyocera MEGACOAT",
                 "notes": "Maintain surface finish uniformity."},
                {"step_number":  7, "feature": "rectangular_pocket",
                 "operation": "Processing Four Rectangular Pockets - Semi-Finishing Milling",
                 "machine_name": "Horizontal Milling Machine", "machine_model": "OKUMA MB-5000H",
                 "cutter_name": "High-Speed Steel End Mill",    "cutter_model": "Kyocera MEGACOAT",
                 "notes": ""},
                {"step_number":  8, "feature": "all",
                 "operation": "Cleaning",
                 "machine_name": "Manual", "machine_model": "Manual",
                 "cutter_name": "Cleaning Tool", "cutter_model": "Standard",
                 "notes": "Remove chips and coolant residue before finish milling."},
                {"step_number":  9, "feature": "flat_surface",
                 "operation": "Processing the Basic Shape - Finishing Milling",
                 "machine_name": "CNC Milling Machine", "machine_model": "DMG MORI NHX 5000",
                 "cutter_name": "Tungsten Carbide End Mill", "cutter_model": "Sandvik CoroMill 316",
                 "notes": "Achieve Ra 0.8-1.6 μm, IT7-IT8."},
                {"step_number": 10, "feature": "rectangular_through_slot",
                 "operation": "Processing Two Rectangular Channels - Finishing Milling",
                 "machine_name": "CNC Milling Machine", "machine_model": "DMG MORI NHX 5000",
                 "cutter_name": "Tungsten Carbide End Mill", "cutter_model": "Sandvik CoroMill 316",
                 "notes": ""},
                {"step_number": 11, "feature": "rectangular_pocket",
                 "operation": "Processing Four Rectangular Pockets - Finishing Milling",
                 "machine_name": "CNC Milling Machine", "machine_model": "DMG MORI NHX 5000",
                 "cutter_name": "Tungsten Carbide End Mill", "cutter_model": "Sandvik CoroMill 316",
                 "notes": ""},
                {"step_number": 12, "feature": "through_hole",
                 "operation": "Processing Two Through Holes - Drilling (Reaming)",
                 "machine_name": "Radial Drill",  "machine_model": "Z3040",
                 "cutter_name": "Drill Bit",       "cutter_model": "Kyocera KSD30",
                 "notes": "Drill both holes in one setup."},
                {"step_number": 13, "feature": "through_hole",
                 "operation": "Processing Two Through Holes - Precision Boring",
                 "machine_name": "Boring Machine", "machine_model": "TPX6111B",
                 "cutter_name": "Carbide Precision Boring Tool",
                 "cutter_model": "Mitsubishi BORING",
                 "notes": "Achieve IT7-IT8 on hole diameter."},
                {"step_number": 14, "feature": "all",
                 "operation": "Final Inspection",
                 "machine_name": "CMM",  "machine_model": "Zeiss CONTURA",
                 "cutter_name": "Probe", "cutter_model": "Renishaw",
                 "notes": "Check all dimensions and surface quality against drawing."},
            ],
            "route_summary": (
                "The final route follows rough milling → semi-finish milling → "
                "finish milling → drilling → boring, grouping all operations of the "
                "same stage together to minimise machine and tool changes. "
                "Deburring and cleaning steps are inserted at stage transitions. "
                "The efficiency score Y = 42, substantially outperforming the typical "
                "route (Y = −10)."
            )
        })


# ── Pipeline orchestrator ─────────────────────────────────────────────────────

class ProcessRoutePipeline:
    """
    End-to-end multi-agent pipeline for process route generation.
    """

    def __init__(self,
                 openai_api_key: str = "",
                 glm_api_key:    str = "",
                 use_mock_llm:   bool = False,
                 use_mock_images: bool = True,
                 use_neo4j:      bool = False,
                 neo4j_uri:      str = "",
                 neo4j_user:     str = "",
                 neo4j_password: str = ""):

        # LLM backend
        if use_mock_llm or not openai_api_key or openai_api_key.startswith("YOUR_"):
            logger.info("Using MOCK LLM (no OpenAI API key).")
            llm = MockLLMTool()
        else:
            llm = LLMTool(api_key=openai_api_key, model=OPENAI_MODEL)

        # Agent construction
        self.fea  = FeatureExtractionAgent(
            llm_tool=llm,
            glm_api_key=glm_api_key,
            use_mock_images=use_mock_images,
        )
        self.mppa = MacroProcessPlanningAgent(llm_tool=llm)
        self.sppa = SpecificProcessPlanningAgent(
            llm_tool=llm,
            use_neo4j=use_neo4j,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
        )
        self.poea = ProcessOptimizationEvaluationAgent(
            llm_tool=llm,
            evaluator=DecisionEvaluator(),
        )
        self.topsis = TopsisEvaluator()

    def run(self,
            stp_path:    Optional[str]       = None,
            image_paths: Optional[List[str]] = None,
            machining_requirements: Optional[Dict] = None,
            topsis_data: Optional[Dict]      = None) -> Dict[str, Any]:
        """
        Execute the full pipeline.

        Returns a dict containing outputs from all four agents + TOPSIS evaluation.
        """
        t_start = time.time()
        results: Dict[str, Any] = {}

        logger.info("=" * 60)
        logger.info("  PROCESS ROUTE GENERATION — START")
        logger.info("=" * 60)

        # ── Agent 1: FEA ──────────────────────────────────────────────────────
        logger.info("\n[STEP 1/4] Feature Extraction Agent (FEA)")
        fea_out = self.fea.run(stp_path=stp_path, image_paths=image_paths)
        results["fea"] = fea_out
        logger.info(f"  → {len(fea_out['features'])} features extracted.")

        # ── Agent 2: MPPA ─────────────────────────────────────────────────────
        logger.info("\n[STEP 2/4] Macro Process Planning Agent (MPPA)")
        mppa_out = self.mppa.run(fea_out)
        results["mppa"] = mppa_out
        logger.info(f"  → {len(mppa_out['macro_sequence'])} steps in macro sequence.")

        # ── Agent 3: SPPA ─────────────────────────────────────────────────────
        logger.info("\n[STEP 3/4] Specific Process Planning Agent (SPPA)")
        sppa_out = self.sppa.run(fea_out, mppa_out, machining_requirements)
        results["sppa"] = sppa_out
        logger.info(f"  → {len(sppa_out['detailed_route'])} feature entries in detailed route.")

        # ── Agent 4: POEA ─────────────────────────────────────────────────────
        logger.info("\n[STEP 4/4] Process Optimization and Evaluation Agent (POEA)")
        poea_out = self.poea.run(mppa_out, sppa_out)
        results["poea"] = poea_out
        logger.info(f"  → Final route: {len(poea_out['final_route'])} steps, "
                     f"score Y={poea_out['best_score']}")

        # ── TOPSIS Evaluation ─────────────────────────────────────────────────
        logger.info("\n[TOPSIS] Evaluating multi-agent route vs. typical route …")
        td = topsis_data or PAPER_DATA
        topsis_results = self.topsis.evaluate(td)
        results["topsis"] = topsis_results
        self.topsis.print_report(topsis_results)

        elapsed = time.time() - t_start
        logger.info(f"\n{'='*60}")
        logger.info(f"  PIPELINE COMPLETE  ({elapsed:.1f}s)")
        logger.info(f"{'='*60}\n")

        return results

    def print_final_route(self, results: Dict[str, Any]):
        """Pretty-print the POEA final route."""
        poea = results.get("poea", {})
        print("\n" + "═" * 70)
        print("  FINAL MACHINING PROCESS ROUTE")
        print("═" * 70)
        print(f"  {poea.get('route_summary', '')}")
        print(f"  Efficiency Score Y = {poea.get('best_score', 'N/A')}")
        print("─" * 70)
        print(f"  {'Step':<5} {'Operation':<45} {'Machine':<20} {'Tool'}")
        print("─" * 70)
        for step in poea.get("final_route", []):
            print(
                f"  {step.get('step_number', ''):<5}"
                f" {step.get('operation', ''):<45}"
                f" {step.get('machine_model', ''):<20}"
                f" {step.get('cutter_model', '')}"
            )
        print("═" * 70 + "\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-Agent Process Route Generation System")
    p.add_argument("--stp",         type=str, help="Path to STP file")
    p.add_argument("--images",      type=str, nargs="+", help="Paths to part view images")
    p.add_argument("--openai-key",  type=str, default="", help="OpenAI API key")
    p.add_argument("--glm-key",     type=str, default="", help="GLM-4V API key")
    p.add_argument("--neo4j",       action="store_true", help="Use live Neo4j")
    p.add_argument("--neo4j-uri",   type=str, default="bolt://localhost:7687")
    p.add_argument("--neo4j-user",  type=str, default="neo4j")
    p.add_argument("--neo4j-pass",  type=str, default="")
    p.add_argument("--demo",        action="store_true",
                   help="Run in demo mode (mock LLM + mock images)")
    p.add_argument("--topsis-only", action="store_true",
                   help="Only run TOPSIS evaluation on paper data")
    p.add_argument("--output",      type=str, default="results.json",
                   help="Output JSON file path")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.topsis_only:
        evaluator = TopsisEvaluator()
        results   = evaluator.evaluate(PAPER_DATA)
        evaluator.print_report(results)
        sys.exit(0)

    openai_key = args.openai_key or OPENAI_API_KEY
    glm_key    = args.glm_key    or GLM_API_KEY
    use_mock   = args.demo or (not openai_key or openai_key.startswith("YOUR_"))

    pipeline = ProcessRoutePipeline(
        openai_api_key  = openai_key,
        glm_api_key     = glm_key,
        use_mock_llm    = use_mock,
        use_mock_images = use_mock or not args.images,
        use_neo4j       = args.neo4j,
        neo4j_uri       = args.neo4j_uri,
        neo4j_user      = args.neo4j_user,
        neo4j_password  = args.neo4j_pass,
    )

    results = pipeline.run(
        stp_path    = args.stp,
        image_paths = args.images,
    )

    pipeline.print_final_route(results)

    # Save full results
    out_path = Path(args.output)
    # Remove numpy arrays before serialising
    serialisable = {
        "fea":  results.get("fea",  {}),
        "mppa": results.get("mppa", {}),
        "sppa": {k: v for k, v in results.get("sppa", {}).items()
                  if k != "kg_context"},
        "poea": results.get("poea", {}),
        "topsis": {
            "closeness": results["topsis"]["closeness"],
            "ranking":   results["topsis"]["ranking"],
            "d_pos":     results["topsis"]["d_pos"],
            "d_neg":     results["topsis"]["d_neg"],
        }
    }
    out_path.write_text(json.dumps(serialisable, indent=2, default=str))
    logger.info(f"Full results saved to: {out_path}")
