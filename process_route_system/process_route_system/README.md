# Multi-Agent Process Route Generation System

Implementation of the paper:

> **"Rapid generation method of process routes based on multi-agent collaboration with LLMs"**  
> Xie et al., *Advanced Engineering Informatics* 68 (2025) 103733  
> DOI: 10.1016/j.aei.2025.103733

---

## System Architecture

```
Part STP File + Multi-view Images
          │
          ▼
┌─────────────────────────────────┐
│  Agent 1: FEA                   │  ← STP Parser + GLM-4V + GPT-4o
│  Feature Extraction Agent       │
│  (S, I) → (F, D)               │
└──────────────┬──────────────────┘
               │ F (features) + D (description)
               ▼
┌─────────────────────────────────┐
│  Agent 2: MPPA                  │  ← GPT-4o
│  Macro Process Planning Agent   │
│  (F, D) → R                     │
└──────────────┬──────────────────┘
               │ R (macro sequence)
               ▼
┌─────────────────────────────────┐
│  Agent 3: SPPA                  │  ← Neo4j/In-Memory KG + GPT-4o
│  Specific Process Planning Agent│  (RAG)
│  (F, RM, RE) → RS               │
└──────────────┬──────────────────┘
               │ RS (detailed route with machines + tools)
               ▼
┌─────────────────────────────────┐
│  Agent 4: POEA                  │  ← Y=100-8a-3b + GPT-4o
│  Process Optimization &         │
│  Evaluation Agent               │
│  (RM, RS) → R*                  │
└──────────────┬──────────────────┘
               │
               ▼
    Final Process Route R*
          +
    TOPSIS Evaluation (Section 4.4)
```

---

## Project Structure

```
process_route_system/
├── pipeline.py                  ← Main entry point / orchestrator
├── config/
│   └── config.py                ← API keys, model settings, TOPSIS weights
├── agents/
│   ├── fea.py                   ← Feature Extraction Agent
│   ├── mppa.py                  ← Macro Process Planning Agent
│   ├── sppa.py                  ← Specific Process Planning Agent
│   └── poea.py                  ← Process Optimization & Evaluation Agent
├── tools/
│   ├── stp_parser.py            ← STP File Analysis Tool (rule-based + GNN stub)
│   ├── image_analyzer.py        ← JPG Image Analysis Tool (GLM-4V)
│   ├── knowledge_graph.py       ← Process Knowledge Graph (Neo4j + in-memory)
│   ├── decision_evaluator.py    ← Decision Evaluation Tool (Y = 100 - 8a - 3b)
│   └── llm_tool.py              ← General Tool (GPT-4o wrapper)
└── evaluation/
    └── topsis.py                ← TOPSIS evaluator (Tables 7 & 8)
```

---

## Installation

```bash
pip install openai langchain langchain-openai neo4j requests numpy pandas python-dotenv
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
GLM_API_KEY=...
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

---

## Usage

### 1. Demo mode (no API keys required)
Uses canned responses matching the paper's aerospace case study (Figs. 9–12):

```bash
python pipeline.py --demo
```

### 2. TOPSIS-only (reproduces paper Tables 7 & 8)
No API keys needed; runs purely on paper data:

```bash
python pipeline.py --topsis-only
```

Expected output:
```
Typical Process Route          D+=0.3333  D-=0.2357  C=0.4142  Rank=2
Multi-agent Generated Route    D+=0.2357  D-=0.3333  C=0.5858  Rank=1
✓ Results match paper.
```

### 3. Full pipeline with real APIs

```bash
python pipeline.py \
  --stp path/to/part.stp \
  --images front.jpg top.jpg iso.jpg \
  --openai-key sk-... \
  --glm-key ...
```

### 4. With live Neo4j knowledge graph

```bash
# First populate Neo4j (run once):
python -c "
from config.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from tools.knowledge_graph import Neo4jKnowledgeGraph
kg = Neo4jKnowledgeGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
kg.populate_from_dict()
"

# Then run with Neo4j:
python pipeline.py --stp part.stp --images *.jpg --neo4j
```

### 5. Python API

```python
from pipeline import ProcessRoutePipeline

pipeline = ProcessRoutePipeline(
    openai_api_key  = "sk-...",
    glm_api_key     = "...",
    use_mock_images = False,   # True = skip GLM-4V
    use_neo4j       = False,   # True = live Neo4j
)

results = pipeline.run(
    stp_path    = "part.stp",
    image_paths = ["front.jpg", "top.jpg", "iso.jpg"],
    machining_requirements = {
        "Ra":               "0.8-1.6 μm",
        "IT":               "IT7-IT8",
        "material":         "15CrMo",
        "production_volume": "mass production",
    }
)

pipeline.print_final_route(results)
```

---

## Paper Reproduction Checklist

| Component | Status | Notes |
|-----------|--------|-------|
| FEA — STP parsing | ✅ | Rule-based; GNN stub in `stp_parser._gnn_segment()` |
| FEA — Image analysis | ✅ | GLM-4V API; mock available |
| FEA — GPT-4o integration | ✅ | Three-stage prompt (Fig. 8) |
| MPPA — Macro sequencing | ✅ | Three-stage prompt (Fig. 8) |
| SPPA — KG query (RAG) | ✅ | In-memory (Tables 4/5) + Neo4j backend |
| SPPA — Detailed route | ✅ | Three-stage prompt (Fig. 8) |
| POEA — Route merging | ✅ | Two candidates + "rough first" principle |
| POEA — Efficiency model | ✅ | Y = 100 − 8a − 3b (Eq. 10) |
| POEA — Route refinement | ✅ | Auxiliary operations added |
| TOPSIS evaluation | ✅ | Matches paper Table 8 to 4 decimal places |
| Process Knowledge Graph | ✅ | Tables 4 & 5 encoded; Neo4j Cypher in `knowledge_graph.py` |

---

## Plugging In Your GNN Model (FEA)

The paper uses a deep-learning GNN for semantic/instance/bottom-face segmentation.
To integrate your model, replace the stub in `tools/stp_parser.py`:

```python
def _gnn_segment(face_dicts: List[Dict]) -> List[MachiningFeature]:
    # Your model inference here:
    # 1. Build graph from face_dicts
    # 2. Run model(graph)
    # 3. Convert predictions to MachiningFeature list
    ...
```

Then call:
```python
result = parse_stp_file("part.stp", use_gnn=True)
```

---

## Efficiency Model (POEA)

From paper Eq. (6)–(10):

```
Y = Y₀ − α·a − β·b

Y₀ = 100 (baseline)
α  = 8   (machine-change weight: ω₁·tₐ + ω₂·Cₐ = 7×10 + 3×6)
β  = 3   (tool-change weight:    ω₁·t_b + ω₂·C_b = 7×3  + 3×4)
a  = number of machine changes
b  = number of tool changes
```

Paper reports for the aerospace case:
- Route Merging Method 1: **Y = 42**
- Route Merging Method 2: **Y = −2**
- Original typical route:  **Y = −10**

---

## TOPSIS Indicators

| Indicator | Type | Unit | Direction | Source |
|-----------|------|------|-----------|--------|
| Production Time | Quantitative | H | Smaller better | Production stats |
| Manufacturing Cost | Quantitative | RMB | Smaller better | Financial system |
| Energy Consumption | Quantitative | kWh | Smaller better | Energy monitoring |
| Process Feasibility | Qualitative | 1–5 | Larger better | Expert evaluation |
| Flexibility | Qualitative | 1–5 | Larger better | Expert evaluation |
| Equipment Utilization | Qualitative | 1–5 | Larger better | Expert evaluation |

---

## Dependencies

| Package | Purpose | Required for |
|---------|---------|--------------|
| `openai` or `requests` | GPT-4o API | All agents (live mode) |
| `neo4j` | Graph database | SPPA (optional; in-memory fallback) |
| `numpy` | TOPSIS matrix ops | Evaluation |
| `pandas` | Data handling | Optional |
| `python-dotenv` | .env loading | Config |
| `requests` | GLM-4V API | FEA image analysis |
