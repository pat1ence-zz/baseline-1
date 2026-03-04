"""
tools/decision_evaluator.py — Decision Evaluation and Analysis Tool for POEA.

Implements the weighted linear efficiency model from paper Section 3.6 (Eq. 6–10):

    Y = Y0 − α·a − β·b

where:
    a  = number of machine (equipment) changes in the process route
    b  = number of tool (cutter) changes in the process route
    α  = 8   (derived from: ω1·ta + ω2·Ca = 7×10 + 3×6, normalised)
    β  = 3   (derived from: ω1·tb + ω2·Cb = 7×3  + 3×4, normalised)
    Y0 = 100 (baseline score with zero changes)

The tool also counts machine/tool switches from a structured process route
list and returns a full evaluation report.
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class ProcessStep:
    """One operation in the merged process route."""
    step_number:   int
    operation:     str
    feature:       str
    machine_name:  str
    machine_model: str
    cutter_name:   str
    cutter_model:  str
    notes:         str = ""


@dataclass
class ProcessRoute:
    """A complete merged process route (output of POEA merging step)."""
    route_id:     str
    description:  str
    steps:        List[ProcessStep] = field(default_factory=list)

    # Computed by evaluator
    machine_changes: int = 0
    tool_changes:    int = 0
    efficiency_score: float = 0.0


# ── Switch counter ─────────────────────────────────────────────────────────────

def count_switches(steps: List[ProcessStep]) -> Tuple[int, int]:
    """
    Count the number of machine changes and tool changes across a route.

    A "change" is defined as consecutive steps where the machine model OR
    cutter model differs from the previous step.

    Returns:
        (machine_changes, tool_changes)
    """
    machine_changes = 0
    tool_changes    = 0

    prev_machine = None
    prev_cutter  = None

    for step in steps:
        curr_machine = step.machine_model.strip()
        curr_cutter  = step.cutter_model.strip()

        if prev_machine is not None and curr_machine != prev_machine:
            machine_changes += 1
        if prev_cutter is not None and curr_cutter != prev_cutter:
            tool_changes += 1

        prev_machine = curr_machine
        prev_cutter  = curr_cutter

    return machine_changes, tool_changes


# ── Efficiency model ───────────────────────────────────────────────────────────

def compute_efficiency_score(machine_changes: int,
                              tool_changes: int,
                              y0: float = 100.0,
                              alpha: float = 8.0,
                              beta: float = 3.0) -> float:
    """
    Implements Equation (10) from the paper:
        Y = Y0 − α·a − β·b

    Args:
        machine_changes : a — number of machine switches
        tool_changes    : b — number of tool switches
        y0              : baseline score (default 100)
        alpha           : machine-change weight (default 8)
        beta            : tool-change weight (default 3)

    Returns:
        Efficiency score Y (can be negative for highly fragmented routes)
    """
    score = y0 - alpha * machine_changes - beta * tool_changes
    return round(score, 2)


# ── Evaluator ──────────────────────────────────────────────────────────────────

class DecisionEvaluator:
    """
    Evaluates and ranks a set of candidate process route schemes
    using the efficiency model from the paper.
    """

    def __init__(self,
                 y0: float = 100.0,
                 alpha: float = 8.0,
                 beta: float = 3.0):
        self.y0    = y0
        self.alpha = alpha
        self.beta  = beta

    def evaluate_route(self, route: ProcessRoute) -> ProcessRoute:
        """Compute and attach switch counts + efficiency score to a route."""
        a, b = count_switches(route.steps)
        route.machine_changes  = a
        route.tool_changes     = b
        route.efficiency_score = compute_efficiency_score(
            a, b, self.y0, self.alpha, self.beta)
        logger.info(
            f"Route '{route.route_id}': a={a}, b={b}, Y={route.efficiency_score}")
        return route

    def evaluate_all(self, routes: List[ProcessRoute]) -> List[ProcessRoute]:
        """Evaluate all routes and return them sorted by score (best first)."""
        evaluated = [self.evaluate_route(r) for r in routes]
        evaluated.sort(key=lambda r: r.efficiency_score, reverse=True)
        return evaluated

    def best_route(self, routes: List[ProcessRoute]) -> ProcessRoute:
        """Return the highest-scoring route."""
        evaluated = self.evaluate_all(routes)
        best = evaluated[0]
        logger.info(f"Best route: '{best.route_id}' (Y={best.efficiency_score})")
        return best

    def report(self, routes: List[ProcessRoute]) -> str:
        """
        Generate a human-readable evaluation report.
        Matches the output format shown in Fig. 12 of the paper.
        """
        evaluated = self.evaluate_all(routes)
        lines = ["=" * 60, "Process Route Efficiency Evaluation Report", "=" * 60]
        for rank, route in enumerate(evaluated, 1):
            lines.append(
                f"\nRank {rank}: {route.route_id} — {route.description}")
            lines.append(
                f"  Machine changes (a): {route.machine_changes}")
            lines.append(
                f"  Tool changes    (b): {route.tool_changes}")
            lines.append(
                f"  Efficiency score Y = {self.y0} - "
                f"{self.alpha}×{route.machine_changes} - "
                f"{self.beta}×{route.tool_changes} = {route.efficiency_score}")
            lines.append(f"  Steps ({len(route.steps)}):")
            for step in route.steps:
                lines.append(
                    f"    {step.step_number:>2}. {step.operation:<30}"
                    f" | {step.machine_name} ({step.machine_model})"
                    f" | {step.cutter_name} ({step.cutter_model})")
        lines.append("\n" + "=" * 60)
        lines.append(f"  BEST ROUTE: {evaluated[0].route_id}"
                     f"  (Y = {evaluated[0].efficiency_score})")
        lines.append("=" * 60)
        return "\n".join(lines)


# ── Helper: build route from SPPA output dict ──────────────────────────────────

def build_route_from_sppa(route_id: str,
                           description: str,
                           sppa_output: List[Dict]) -> ProcessRoute:
    """
    Convert the structured output of SPPA into a ProcessRoute object.

    Args:
        route_id    : Identifier string (e.g. "Route_1")
        description : Short description (e.g. "Merge method A")
        sppa_output : List of dicts, each with keys:
                        feature, operation, machine_name, machine_model,
                        cutter_name, cutter_model [, notes]
    """
    steps = []
    for i, item in enumerate(sppa_output, 1):
        steps.append(ProcessStep(
            step_number   = i,
            operation     = item.get("operation",     ""),
            feature       = item.get("feature",       ""),
            machine_name  = item.get("machine_name",  ""),
            machine_model = item.get("machine_model", ""),
            cutter_name   = item.get("cutter_name",   ""),
            cutter_model  = item.get("cutter_model",  ""),
            notes         = item.get("notes",         ""),
        ))
    return ProcessRoute(route_id=route_id, description=description, steps=steps)


# ── Demo / CLI ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Reproduce the paper's Fig. 12 evaluation scenario
    # The paper reports:
    #   Route Merging Method 1: Y = 42
    #   Route Merging Method 2: Y = −2
    #   Original typical route: Y = −10

    from tools.decision_evaluator import (
        ProcessStep, ProcessRoute, DecisionEvaluator, build_route_from_sppa)

    # Paper's "Route Merging Method 1" (the best one, Y=42)
    route1_steps = [
        # Rough milling phase  (same machine: Doosan BM2045, same tool: APX4000)
        {"feature": "flat_surface",            "operation": "Rough Milling",
         "machine_name": "Gantry Milling Machine",  "machine_model": "Doosan BM2045",
         "cutter_name": "Carbide End Mill",     "cutter_model": "Kyocera PR1535"},
        {"feature": "rectangular_through_slot","operation": "Rough Milling",
         "machine_name": "Gantry Milling Machine",  "machine_model": "Doosan BM2740",
         "cutter_name": "Carbide End Mill",     "cutter_model": "Mitsubishi APX4000"},
        {"feature": "rectangular_pocket",      "operation": "Rough Milling",
         "machine_name": "Gantry Milling Machine",  "machine_model": "Doosan BM2740",
         "cutter_name": "Carbide End Mill",     "cutter_model": "Mitsubishi APX4000"},
        # Deburring
        {"feature": "all",                     "operation": "Deburring",
         "machine_name": "Manual",              "machine_model": "Manual",
         "cutter_name": "Deburring Tool",       "cutter_model": "Standard"},
        # Semi-finish milling phase (same machine: OKUMA MB-5000H)
        {"feature": "flat_surface",            "operation": "Semi-Finishing Milling",
         "machine_name": "Horizontal Milling Machine", "machine_model": "OKUMA MB-5000H",
         "cutter_name": "HSS End Mill",         "cutter_model": "Kennametal KSEM"},
        {"feature": "rectangular_through_slot","operation": "Semi-Finishing Milling",
         "machine_name": "Horizontal Milling Machine", "machine_model": "OKUMA MB-5000H",
         "cutter_name": "HSS End Mill",         "cutter_model": "Kyocera MEGACOAT"},
        {"feature": "rectangular_pocket",      "operation": "Semi-Finishing Milling",
         "machine_name": "Horizontal Milling Machine", "machine_model": "OKUMA MB-5000H",
         "cutter_name": "HSS End Mill",         "cutter_model": "Kyocera MEGACOAT"},
        # Cleaning
        {"feature": "all",                     "operation": "Cleaning",
         "machine_name": "Manual",              "machine_model": "Manual",
         "cutter_name": "Cleaning Tool",        "cutter_model": "Standard"},
        # Finish milling (DMG MORI NHX 5000)
        {"feature": "flat_surface",            "operation": "Finishing Milling",
         "machine_name": "CNC Milling Machine", "machine_model": "DMG MORI NHX 5000",
         "cutter_name": "Tungsten Carbide End Mill", "cutter_model": "Sandvik CoroMill 316"},
        {"feature": "rectangular_through_slot","operation": "Finishing Milling",
         "machine_name": "CNC Milling Machine", "machine_model": "DMG MORI NHX 5000",
         "cutter_name": "Tungsten Carbide End Mill", "cutter_model": "Sandvik CoroMill 316"},
        {"feature": "rectangular_pocket",      "operation": "Finishing Milling",
         "machine_name": "CNC Milling Machine", "machine_model": "DMG MORI NHX 5000",
         "cutter_name": "Tungsten Carbide End Mill", "cutter_model": "Sandvik CoroMill 316"},
        # Hole drilling
        {"feature": "through_hole",            "operation": "Drilling (Reaming)",
         "machine_name": "Radial Drill",        "machine_model": "Z3040",
         "cutter_name": "Drill Bit",            "cutter_model": "Kyocera KSD30"},
        {"feature": "through_hole",            "operation": "Precision Boring",
         "machine_name": "Boring Machine",      "machine_model": "TPX6111B",
         "cutter_name": "Carbide Precision Boring Tool", "cutter_model": "Mitsubishi BORING"},
        # Final inspection
        {"feature": "all",                     "operation": "Final Inspection",
         "machine_name": "CMM",                 "machine_model": "Zeiss CONTURA",
         "cutter_name": "Probe",                "cutter_model": "Renishaw"},
    ]

    route1 = build_route_from_sppa("Route_MA1", "Multi-agent Merging Method 1", route1_steps)

    evaluator = DecisionEvaluator()
    print(evaluator.report([route1]))
