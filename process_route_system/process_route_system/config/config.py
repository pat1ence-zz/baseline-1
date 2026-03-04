"""
config.py — Central configuration for the Process Route Generation System.

Set your API keys either:
  1. In a .env file at the project root (recommended), or
  2. Directly in the KEYS dict below (not recommended for production).

Required API keys:
  - OPENAI_API_KEY   : GPT-4o access (MPPA, SPPA, POEA, General Tool)
  - GLM_API_KEY      : GLM-4V access (FEA image analysis)
  - NEO4J_URI        : e.g. bolt://localhost:7687
  - NEO4J_USER       : e.g. neo4j
  - NEO4J_PASSWORD   : your Neo4j password
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env file in project root

# ── Model identifiers ──────────────────────────────────────────────────────────
OPENAI_MODEL   = "gpt-4o"
GLM_MODEL      = "glm-4v"

# ── API credentials (override via .env) ───────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")
GLM_API_KEY    = os.getenv("GLM_API_KEY",    "YOUR_GLM_API_KEY_HERE")

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "YOUR_NEO4J_PASSWORD_HERE")

# ── POEA efficiency model parameters (from paper Section 3.6) ─────────────────
# Y = Y0 - alpha * a - beta * b
# a = number of machine changes, b = number of tool changes
POEA_Y0    = 100   # baseline score
POEA_ALPHA = 8     # weight for machine changes  (α = ω1·ta + ω2·Ca = 7×10 + 3×6 = 88 → normalised to 8)
POEA_BETA  = 3     # weight for tool changes     (β = ω1·tb + ω2·Cb = 7×3  + 3×4 = 33 → normalised to 3)

# ── TOPSIS evaluation weights (equal weighting across 6 indicators) ───────────
# Indicators: production_time, manufacturing_cost, energy_consumption,
#             process_feasibility, flexibility, equipment_utilization
TOPSIS_WEIGHTS = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]

# ── Indicator direction: True = "larger is better", False = "smaller is better"
TOPSIS_BENEFIT = [False, False, False, True, True, True]

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
