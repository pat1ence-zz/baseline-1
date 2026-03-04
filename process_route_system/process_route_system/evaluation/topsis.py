"""
evaluation/topsis.py — TOPSIS Multi-Criteria Evaluation Module

Implements the TOPSIS (Technique for Order Preference by Similarity to Ideal
Solution) evaluation method described in paper Section 4.4.

Reproduces:
  - Table 6: Evaluation indicator definitions
  - Table 7: Original index data (both routes)
  - Table 8: Closeness coefficient comparison

Evaluation indicators (6 total):
  Quantitative (3):
    1. Production Time       (H)    — smaller is better
    2. Manufacturing Cost    (RMB)  — smaller is better
    3. Energy Consumption    (kWh)  — smaller is better
  Qualitative (3):
    4. Process Feasibility   (1–5)  — larger is better
    5. Flexibility           (1–5)  — larger is better
    6. Equipment Utilization (1–5)  — larger is better

Usage:
    from evaluation.topsis import TopsisEvaluator, PAPER_DATA
    evaluator = TopsisEvaluator()
    results   = evaluator.evaluate(PAPER_DATA)
    evaluator.print_report(results)
"""

import math
import logging
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Paper data (Tables 7 & 8) ──────────────────────────────────────────────────

PAPER_DATA = {
    "schemes": ["Typical Process Route", "Multi-agent Generated Process Route"],
    "indicators": [
        "Production Time (H)",
        "Manufacturing Cost (RMB)",
        "Energy Consumption (kWh)",
        "Process Feasibility (1-5)",
        "Flexibility (1-5)",
        "Equipment Utilization (1-5)",
    ],
    # Rows = schemes, Cols = indicators (in order above)
    "raw_data": [
        [16.5, 1280, 235, 4.5, 3.2, 3.7],   # Typical
        [14.8, 1325, 193, 3.25, 4.3, 4.25],  # Multi-agent
    ],
    # True = benefit (larger is better), False = cost (smaller is better)
    "benefit": [False, False, False, True, True, True],
    # Equal weights (paper does not specify different weights)
    "weights":  [1/6, 1/6, 1/6, 1/6, 1/6, 1/6],
    # Expected results from Table 8
    "expected_closeness": [0.414, 0.586],
}


# ── TOPSIS implementation ──────────────────────────────────────────────────────

class TopsisEvaluator:
    """
    TOPSIS evaluator.

    Steps:
      1. Normalise the decision matrix (min-max normalisation, Eq. 11 in paper)
      2. Apply weights
      3. Identify positive ideal solution (PIS) and negative ideal solution (NIS)
      4. Compute distances to PIS and NIS
      5. Compute closeness coefficient C = D_neg / (D_pos + D_neg)
      6. Rank schemes by closeness coefficient (higher = better)
    """

    def __init__(self,
                 weights:  Optional[List[float]] = None,
                 benefit:  Optional[List[bool]]  = None):
        """
        Args:
            weights : List of indicator weights (must sum to 1.0).
                      If None, equal weights are used.
            benefit : List of booleans — True = "larger is better".
                      If None, all indicators treated as benefit.
        """
        self.weights = weights
        self.benefit = benefit

    # ── Normalisation (Equation 11 from paper) ────────────────────────────────

    @staticmethod
    def _minmax_normalise(matrix: np.ndarray,
                           benefit: List[bool]) -> np.ndarray:
        """
        Min-max normalise so that a higher normalised value always means better.

        x'_ij = (x_ij - min_j) / (max_j - min_j)        for benefit indicators
        x'_ij = (max_j - x_ij) / (max_j - min_j)        for cost indicators
        """
        norm = np.zeros_like(matrix, dtype=float)
        for j in range(matrix.shape[1]):
            col     = matrix[:, j]
            col_min = col.min()
            col_max = col.max()
            spread  = col_max - col_min

            if spread < 1e-12:          # all values identical — treat as 0.5
                norm[:, j] = 0.5
                continue

            if benefit[j]:
                norm[:, j] = (col - col_min) / spread
            else:
                norm[:, j] = (col_max - col) / spread

        return norm

    # ── Core TOPSIS ───────────────────────────────────────────────────────────

    def evaluate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run TOPSIS on the supplied data dict.

        Args:
            data : Dict with keys:
                     schemes    — list of scheme names
                     indicators — list of indicator names
                     raw_data   — 2D list (schemes × indicators)
                     benefit    — list[bool] per indicator
                     weights    — list[float] per indicator

        Returns:
            {
              "scheme_names":  list[str],
              "indicators":    list[str],
              "raw_data":      ndarray,
              "normalised":    ndarray,
              "weighted":      ndarray,
              "pis":           ndarray,
              "nis":           ndarray,
              "d_pos":         list[float],
              "d_neg":         list[float],
              "closeness":     list[float],
              "ranking":       list[int],   # 1-based rank for each scheme
            }
        """
        schemes    = data["schemes"]
        indicators = data["indicators"]
        raw        = np.array(data["raw_data"], dtype=float)
        benefit    = data.get("benefit",  self.benefit  or [True]*raw.shape[1])
        weights    = data.get("weights",  self.weights  or [1/raw.shape[1]]*raw.shape[1])

        n_schemes, n_indicators = raw.shape
        assert len(benefit) == n_indicators, "benefit list length mismatch"
        assert len(weights) == n_indicators, "weights list length mismatch"
        assert abs(sum(weights) - 1.0) < 1e-9, "weights must sum to 1.0"

        w = np.array(weights)

        # Step 1 — normalise
        normalised = self._minmax_normalise(raw, benefit)

        # Step 2 — weight
        weighted = normalised * w

        # Step 3 — PIS and NIS
        # After normalisation, higher = better for all indicators
        pis = weighted.max(axis=0)
        nis = weighted.min(axis=0)

        # Step 4 — Euclidean distances
        d_pos = np.sqrt(((weighted - pis) ** 2).sum(axis=1))
        d_neg = np.sqrt(((weighted - nis) ** 2).sum(axis=1))

        # Step 5 — Closeness coefficient
        closeness = d_neg / (d_pos + d_neg + 1e-12)

        # Step 6 — Ranking (1 = best)
        order   = closeness.argsort()[::-1]           # descending
        ranking = np.empty_like(order)
        for rank, idx in enumerate(order, 1):
            ranking[idx] = rank

        logger.info("TOPSIS evaluation complete.")
        for i, name in enumerate(schemes):
            logger.info(
                f"  {name}: D+={d_pos[i]:.4f}  D-={d_neg[i]:.4f}"
                f"  C={closeness[i]:.4f}  Rank={ranking[i]}")

        return {
            "scheme_names": schemes,
            "indicators":   indicators,
            "raw_data":     raw,
            "normalised":   normalised,
            "weighted":     weighted,
            "pis":          pis,
            "nis":          nis,
            "d_pos":        d_pos.tolist(),
            "d_neg":        d_neg.tolist(),
            "closeness":    closeness.tolist(),
            "ranking":      ranking.tolist(),
        }

    # ── Report printing ───────────────────────────────────────────────────────

    def print_report(self, results: Dict[str, Any]):
        """Print a formatted TOPSIS report matching the paper's Table 8."""
        schemes    = results["scheme_names"]
        indicators = results["indicators"]
        raw        = results["raw_data"]
        norm       = results["normalised"]
        d_pos      = results["d_pos"]
        d_neg      = results["d_neg"]
        closeness  = results["closeness"]
        ranking    = results["ranking"]

        sep = "─" * 80

        print("\n" + "═" * 80)
        print("  TOPSIS EVALUATION REPORT")
        print("  Paper Reference: Section 4.4, Tables 7 & 8")
        print("═" * 80)

        # Raw data table
        print(f"\n{'Indicator':<35}  " + "  ".join(f"{s:<25}" for s in schemes))
        print(sep)
        for j, ind in enumerate(indicators):
            row_vals = "  ".join(f"{raw[i, j]:<25.4g}" for i in range(len(schemes)))
            print(f"{ind:<35}  {row_vals}")

        # Normalised data
        print(f"\n{'Normalised Indicator':<35}  "
              + "  ".join(f"{s:<25}" for s in schemes))
        print(sep)
        for j, ind in enumerate(indicators):
            row_vals = "  ".join(f"{norm[i, j]:<25.4f}" for i in range(len(schemes)))
            print(f"{ind:<35}  {row_vals}")

        # Closeness table (Table 8)
        print("\n" + "=" * 80)
        print("  TABLE 8: Closeness Coefficient Comparison")
        print("=" * 80)
        header = f"{'Scheme':<40} {'D+ (to PIS)':>12} {'D- (to NIS)':>12} {'C (closeness)':>14} {'Rank':>6}"
        print(header)
        print(sep)
        for i, name in enumerate(schemes):
            print(
                f"{name:<40} {d_pos[i]:>12.4f} {d_neg[i]:>12.4f}"
                f" {closeness[i]:>14.4f} {int(ranking[i]):>6}"
            )

        # Paper's expected values for verification
        expected = PAPER_DATA.get("expected_closeness", [])
        if expected:
            print(f"\n  Paper reported closeness: {expected}")
            diffs = [abs(closeness[i] - expected[i]) for i in range(len(expected))]
            print(f"  Absolute differences:     {[round(d, 4) for d in diffs]}")
            match = all(d < 0.01 for d in diffs)
            print(f"  ✓ Results match paper." if match
                  else f"  ⚠ Results differ from paper (check weights/normalisation).")

        print("\n" + "=" * 80)
        best_idx = int(np.argmax(closeness))
        print(f"  CONCLUSION: '{schemes[best_idx]}' is the superior route")
        print(f"  (Closeness coefficient = {closeness[best_idx]:.4f})")
        print("=" * 80 + "\n")


# ── Sensitivity analysis ──────────────────────────────────────────────────────

def sensitivity_analysis(data: Dict[str, Any],
                          weight_variations: int = 10) -> List[Dict]:
    """
    Test TOPSIS robustness by varying indicator weights.
    Returns a list of results under different weight configurations.
    """
    n = len(data["indicators"])
    results = []
    evaluator = TopsisEvaluator()

    # Vary each indicator's weight from 0.05 to 0.50, redistribute remainder equally
    for j in range(n):
        for alpha in np.linspace(0.05, 0.50, weight_variations):
            w          = np.full(n, (1 - alpha) / (n - 1))
            w[j]       = alpha
            data_copy  = {**data, "weights": w.tolist()}
            res        = evaluator.evaluate(data_copy)
            results.append({
                "varied_indicator": data["indicators"][j],
                "weight":           round(alpha, 3),
                "closeness":        [round(c, 4) for c in res["closeness"]],
                "ranking":          res["ranking"],
            })

    # Summarise: count how often multi-agent route ranks first
    ma_idx  = 1  # Multi-agent is scheme index 1
    n_first = sum(1 for r in results if r["ranking"][ma_idx] == 1)
    print(f"\nSensitivity: Multi-agent route ranks #1 in "
          f"{n_first}/{len(results)} weight scenarios "
          f"({100*n_first/len(results):.1f}%)")
    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    evaluator = TopsisEvaluator()
    results   = evaluator.evaluate(PAPER_DATA)
    evaluator.print_report(results)

    print("\nRunning sensitivity analysis …")
    sensitivity_analysis(PAPER_DATA)
