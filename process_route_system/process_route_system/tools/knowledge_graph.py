"""
tools/knowledge_graph.py — Process Knowledge Graph Query Tool for SPPA.

Wraps a Neo4j graph database containing typical machining feature process chains
and associated machining resources (machines + cutting tools).

Graph ontology (from paper Fig. 4 / Section 3.5):
  (MainProcessFeature) -[:have]-> (ProcessChain)
  (ProcessChain)       -[:have]-> (ProcessX)         [individual operations]
  (ProcessX)           -[:machiner_is]-> (Machine)
  (ProcessX)           -[:cutter_is]->  (Cutter)
  (Machine) -[:have]-> (MachineResource)
  (Cutter)  -[:have]-> (CuttingToolResource)

Attributes on ProcessChain:
  Ra, Dimensional_accuracy, Materials, Describe

This module provides:
  1. Neo4jKnowledgeGraph  — live Neo4j backend
  2. InMemoryKnowledgeGraph — self-contained fallback with paper's Table 4/5 data
"""

import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  In-Memory Knowledge Graph  (paper Tables 4 & 5, no Neo4j required)
# ══════════════════════════════════════════════════════════════════════════════

# Process knowledge base built from the paper's Table 4 (hole chains) and
# generalised to the rectangular slot / pocket / flat surface features used
# in the aerospace case study.

KNOWLEDGE_BASE: Dict[str, Any] = {
    "through_hole": {
        "process_chains": [
            {
                "chain_id": "PC_HOLE_1",
                "Ra": "6.3–25", "Dimensional_accuracy": "IT11-IT12",
                "Materials": "ferrous and non-ferrous metals",
                "Describe": "Basic hole drilling for low-precision requirements",
                "steps": [
                    {"operation": "Drilling",
                     "machine": {"name": "Vertical Drilling Machine", "model": "Z3040"},
                     "cutter":  {"name": "Twist Drill", "model": "Sandvik R840-5"}},
                ]
            },
            {
                "chain_id": "PC_HOLE_2",
                "Ra": "6.3–12.5", "Dimensional_accuracy": "IT9-IT10",
                "Materials": "ferrous and non-ferrous metals",
                "Describe": "Drilling + reaming for medium precision",
                "steps": [
                    {"operation": "Drilling",
                     "machine": {"name": "Vertical Drilling Machine", "model": "Z5140"},
                     "cutter":  {"name": "Twist Drill", "model": "Kyocera KSD35"}},
                    {"operation": "Reaming",
                     "machine": {"name": "Radial Drilling Machine",   "model": "Z3041"},
                     "cutter":  {"name": "Reamer", "model": "Mitsubishi MWE080"}},
                ]
            },
            {
                "chain_id": "PC_HOLE_3",
                "Ra": "3.2–6.3", "Dimensional_accuracy": "IT8-IT9",
                "Materials": "ferrous and non-ferrous metals",
                "Describe": "Drilling + reaming + honing",
                "steps": [
                    {"operation": "Drilling",
                     "machine": {"name": "Vertical Drilling Machine", "model": "Z5140"},
                     "cutter":  {"name": "Twist Drill", "model": "Kyocera KSD35"}},
                    {"operation": "Reaming",
                     "machine": {"name": "Radial Drilling Machine",   "model": "Z3041"},
                     "cutter":  {"name": "Reamer", "model": "Mitsubishi MWE080"}},
                    {"operation": "Honing",
                     "machine": {"name": "Honing Machine", "model": "M4215"},
                     "cutter":  {"name": "Honing Stone", "model": "GD-80"}},
                ]
            },
            {
                "chain_id": "PC_HOLE_4",
                "Ra": "0.8–3.2", "Dimensional_accuracy": "IT7-IT8",
                "Materials": "ferrous metals",
                "Describe": "Drilling + reaming + rough/finish honing (high precision)",
                "steps": [
                    {"operation": "Drilling (Reaming)",
                     "machine": {"name": "Radial Drill",   "model": "Z3040"},
                     "cutter":  {"name": "Drill Bit",       "model": "Kyocera KSD30"}},
                    {"operation": "Precision Boring",
                     "machine": {"name": "Boring Machine",  "model": "TPX6111B"},
                     "cutter":  {"name": "Carbide Precision Boring Tool",
                                  "model": "Mitsubishi BORING"}},
                ]
            },
            {
                "chain_id": "PC_HOLE_5",
                "Ra": "0.2–0.8", "Dimensional_accuracy": "IT6-IT7",
                "Materials": "ferrous metals",
                "Describe": "Full precision — drilling through carbide boring",
                "steps": [
                    {"operation": "Rough Boring",
                     "machine": {"name": "Boring Machine",  "model": "TPX6111B"},
                     "cutter":  {"name": "Carbide Rough Boring Tool",
                                  "model": "Sandvik CoroBore BR20"}},
                    {"operation": "Semi-finish Boring",
                     "machine": {"name": "Boring Machine",  "model": "TPX6117B"},
                     "cutter":  {"name": "High-speed Steel Semi-finish Boring Tool",
                                  "model": "Mitsubishi BORING"}},
                    {"operation": "Finish Boring",
                     "machine": {"name": "Boring Machine",  "model": "TPX6117B"},
                     "cutter":  {"name": "Carbide Finish Boring Tool",
                                  "model": "Sandvik CoroBore 825"}},
                ]
            },
        ]
    },
    "rectangular_through_slot": {
        "process_chains": [
            {
                "chain_id": "PC_SLOT_1",
                "Ra": "3.2–12.5", "Dimensional_accuracy": "IT9-IT11",
                "Materials": "ferrous and non-ferrous metals",
                "Describe": "Rough + semi-finish + finish milling for through slots",
                "steps": [
                    {"operation": "Rough Milling",
                     "machine": {"name": "Gantry Milling Machine", "model": "Doosan BM2740"},
                     "cutter":  {"name": "Carbide End Mill", "model": "Mitsubishi APX4000"}},
                    {"operation": "Semi-Finishing Milling",
                     "machine": {"name": "Horizontal Milling Machine", "model": "OKUMA MB-5000H"},
                     "cutter":  {"name": "High-Speed Steel End Mill", "model": "Kyocera MEGACOAT"}},
                    {"operation": "Finishing Milling",
                     "machine": {"name": "CNC Milling Machine", "model": "DMG MORI NHX 5000"},
                     "cutter":  {"name": "Tungsten Carbide End Mill", "model": "Sandvik CoroMill 316"}},
                    {"operation": "Ultra-Fine Milling",
                     "machine": {"name": "Ultra-Precision CNC Milling Machine", "model": "Makino F5"},
                     "cutter":  {"name": "Ceramic Tool", "model": "Kyocera PV720"}},
                ]
            },
        ]
    },
    "rectangular_pocket": {
        "process_chains": [
            {
                "chain_id": "PC_POCKET_1",
                "Ra": "0.8–3.2", "Dimensional_accuracy": "IT7-IT9",
                "Materials": "ferrous and non-ferrous metals",
                "Describe": "4-step milling for rectangular pockets",
                "steps": [
                    {"operation": "Rough Milling",
                     "machine": {"name": "Gantry Milling Machine", "model": "Doosan BM2740"},
                     "cutter":  {"name": "Carbide End Mill", "model": "Mitsubishi APX4000"}},
                    {"operation": "Semi-Finishing Milling",
                     "machine": {"name": "Horizontal Milling Machine", "model": "OKUMA MB-5000H"},
                     "cutter":  {"name": "High-Speed Steel End Mill", "model": "Kyocera MEGACOAT"}},
                    {"operation": "Finishing Milling",
                     "machine": {"name": "CNC Milling Machine", "model": "DMG MORI NHX 5000"},
                     "cutter":  {"name": "Tungsten Carbide End Mill", "model": "Sandvik CoroMill 316"}},
                    {"operation": "Ultra-Fine Milling",
                     "machine": {"name": "Ultra-Precision CNC Milling Machine", "model": "Makino F5"},
                     "cutter":  {"name": "Ceramic Tool", "model": "Kyocera PV720"}},
                ]
            },
        ]
    },
    "flat_surface": {
        "process_chains": [
            {
                "chain_id": "PC_FLAT_1",
                "Ra": "0.8–3.2", "Dimensional_accuracy": "IT7-IT8",
                "Materials": "ferrous and non-ferrous metals",
                "Describe": "3-step milling for flat reference surfaces",
                "steps": [
                    {"operation": "Rough Milling",
                     "machine": {"name": "Gantry Milling Machine", "model": "Doosan BM2045"},
                     "cutter":  {"name": "Carbide End Mill", "model": "Kyocera PR1535"}},
                    {"operation": "Semi-Finishing Milling",
                     "machine": {"name": "Horizontal Milling Machine", "model": "OKUMA MB-5000H"},
                     "cutter":  {"name": "High-Speed Steel End Mill", "model": "Kennametal KSEM"}},
                    {"operation": "Finishing Milling",
                     "machine": {"name": "CNC Milling Machine", "model": "DMG MORI NHX 5000"},
                     "cutter":  {"name": "Tungsten Carbide End Mill", "model": "Sandvik CoroMill 316"}},
                ]
            },
        ]
    },
    "chamfer": {
        "process_chains": [
            {
                "chain_id": "PC_CHAMFER_1",
                "Ra": "3.2–12.5", "Dimensional_accuracy": "IT11-IT12",
                "Materials": "all metals",
                "Describe": "Manual deburring and chamfering",
                "steps": [
                    {"operation": "Chamfering",
                     "machine": {"name": "Chamfering Machine", "model": "Manual"},
                     "cutter":  {"name": "Chamfering Tool", "model": "Standard"}},
                ]
            },
        ]
    },
}


def _select_chain(chains: List[Dict], ra_target: str,
                  dim_acc_target: str) -> Dict:
    """
    Choose the best-matching process chain for given Ra and IT grade.
    Falls back to the last (most precise) chain if no match found.
    """
    for chain in chains:
        if ra_target in chain["Ra"] or dim_acc_target in chain["Dimensional_accuracy"]:
            return chain
    return chains[-1]  # default: highest precision available


class InMemoryKnowledgeGraph:
    """
    Self-contained knowledge graph using the paper's process chain data.
    No Neo4j installation required.
    """

    def query_process_chain(self, feature_type: str,
                             ra: str = "0.8-1.6",
                             dim_accuracy: str = "IT7-IT8") -> Optional[Dict]:
        """
        Find the best matching process chain for a feature type and precision.

        Args:
            feature_type  : One of through_hole | rectangular_through_slot |
                            rectangular_pocket | flat_surface | chamfer
            ra            : Required surface roughness (Ra) string, e.g. "0.8-1.6"
            dim_accuracy  : Required IT grade string, e.g. "IT7-IT8"

        Returns:
            Selected process chain dict or None if feature not found.
        """
        kb = KNOWLEDGE_BASE.get(feature_type)
        if not kb:
            logger.warning(f"Feature type '{feature_type}' not in knowledge base.")
            return None
        chains = kb["process_chains"]
        chain  = _select_chain(chains, ra, dim_accuracy)
        logger.info(f"Selected chain {chain['chain_id']} for feature '{feature_type}'")
        return chain

    def query_all_features(self, feature_list: List[Dict],
                            ra: str = "0.8-1.6",
                            dim_accuracy: str = "IT7-IT8") -> List[Dict]:
        """
        Batch query: for each feature in feature_list, find the process chain.

        Args:
            feature_list : List of feature dicts with keys 'feature_type' (and optionally
                           'description').
            ra, dim_accuracy: Part-wide precision requirements.

        Returns:
            List of {feature, chain} dicts.
        """
        results = []
        for feat in feature_list:
            ftype = feat.get("feature_type", "flat_surface")
            chain = self.query_process_chain(ftype, ra, dim_accuracy)
            results.append({"feature": feat, "chain": chain})
        return results

    def format_for_agent(self, query_results: List[Dict]) -> str:
        """
        Format query results as structured text for the SPPA agent prompt.
        Mirrors the Cypher-query triplet output style mentioned in the paper.
        """
        lines = []
        for item in query_results:
            feat  = item["feature"]
            chain = item["chain"]
            if chain is None:
                lines.append(f"Feature: {feat.get('feature_type')} — No chain found.")
                continue
            lines.append(f"\n**{feat.get('feature_type', 'unknown')}**"
                         f" (Chain: {chain['chain_id']},"
                         f" Ra: {chain['Ra']},"
                         f" IT: {chain['Dimensional_accuracy']})")
            for step in chain["steps"]:
                lines.append(
                    f"  • {step['operation']}"
                    f" — Machine: {step['machine']['name']} ({step['machine']['model']})"
                    f" | Tool: {step['cutter']['name']} ({step['cutter']['model']})"
                )
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  Neo4j Live Knowledge Graph
# ══════════════════════════════════════════════════════════════════════════════

class Neo4jKnowledgeGraph:
    """
    Live Neo4j backend.  Requires neo4j-driver: `pip install neo4j`.
    """

    def __init__(self, uri: str, user: str, password: str):
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self._test_connection()
            logger.info("Connected to Neo4j.")
        except ImportError:
            raise ImportError("Install neo4j driver: pip install neo4j")
        except Exception as e:
            raise ConnectionError(f"Neo4j connection failed: {e}")

    def _test_connection(self):
        with self.driver.session() as session:
            session.run("RETURN 1")

    def close(self):
        self.driver.close()

    def query_process_chain(self, feature_type: str,
                             ra: str = "0.8-1.6",
                             dim_accuracy: str = "IT7-IT8") -> Optional[Dict]:
        """
        Cypher query matching the paper's PKG ontology.
        Returns the best process chain for the given feature + precision.
        """
        cypher = """
        MATCH (mf:MainProcessFeature {name: $feature_type})-[:have]->(pc:ProcessChain)
        WHERE pc.Ra CONTAINS $ra OR pc.Dimensional_accuracy CONTAINS $dim_acc
        MATCH (pc)-[:have]->(px:ProcessX)
        OPTIONAL MATCH (px)-[:machiner_is]->(m:Machine)-[:have]->(mr:MachineResource)
        OPTIONAL MATCH (px)-[:cutter_is]->(c:Cutter)-[:have]->(cr:CuttingToolResource)
        RETURN pc, collect({
            operation: px.name,
            machine: m.name,
            machine_model: mr.model,
            cutter: c.name,
            cutter_model: cr.model
        }) AS steps
        LIMIT 1
        """
        with self.driver.session() as session:
            result = session.run(cypher,
                                  feature_type=feature_type,
                                  ra=ra,
                                  dim_acc=dim_accuracy)
            record = result.single()
            if not record:
                return None
            pc    = dict(record["pc"])
            steps = record["steps"]
            return {**pc, "steps": steps}

    def query_all_features(self, feature_list: List[Dict],
                            ra: str = "0.8-1.6",
                            dim_accuracy: str = "IT7-IT8") -> List[Dict]:
        results = []
        for feat in feature_list:
            chain = self.query_process_chain(
                feat.get("feature_type", "flat_surface"), ra, dim_accuracy)
            results.append({"feature": feat, "chain": chain})
        return results

    def format_for_agent(self, query_results: List[Dict]) -> str:
        """Delegate to InMemoryKnowledgeGraph formatter (same logic)."""
        return InMemoryKnowledgeGraph().format_for_agent(query_results)

    def populate_from_dict(self):
        """
        Populate the Neo4j database from the built-in KNOWLEDGE_BASE dict.
        Run once to initialise a fresh Neo4j instance.
        """
        cypher_feature = "MERGE (mf:MainProcessFeature {name: $name})"
        cypher_chain   = """
        MATCH (mf:MainProcessFeature {name: $feature_type})
        MERGE (pc:ProcessChain {chain_id: $chain_id})
        SET pc.Ra = $ra, pc.Dimensional_accuracy = $dim_acc,
            pc.Materials = $materials, pc.Describe = $describe
        MERGE (mf)-[:have]->(pc)
        """
        cypher_step = """
        MATCH (pc:ProcessChain {chain_id: $chain_id})
        MERGE (px:ProcessX {name: $op, chain_id: $chain_id})
        MERGE (pc)-[:have]->(px)
        MERGE (m:Machine {name: $machine_name, model: $machine_model})
        MERGE (px)-[:machiner_is]->(m)
        MERGE (c:Cutter {name: $cutter_name, model: $cutter_model})
        MERGE (px)-[:cutter_is]->(c)
        """
        with self.driver.session() as session:
            for feature_type, data in KNOWLEDGE_BASE.items():
                session.run(cypher_feature, name=feature_type)
                for chain in data["process_chains"]:
                    session.run(cypher_chain,
                                 feature_type=feature_type,
                                 chain_id=chain["chain_id"],
                                 ra=chain["Ra"],
                                 dim_acc=chain["Dimensional_accuracy"],
                                 materials=chain["Materials"],
                                 describe=chain["Describe"])
                    for step in chain["steps"]:
                        session.run(cypher_step,
                                     chain_id=chain["chain_id"],
                                     op=step["operation"],
                                     machine_name=step["machine"]["name"],
                                     machine_model=step["machine"]["model"],
                                     cutter_name=step["cutter"]["name"],
                                     cutter_model=step["cutter"]["model"])
        logger.info("Neo4j populated from built-in knowledge base.")


# ── Factory function ───────────────────────────────────────────────────────────

def get_knowledge_graph(use_neo4j: bool = False,
                         uri: str = "", user: str = "", password: str = ""):
    """
    Return the appropriate knowledge graph backend.

    Args:
        use_neo4j : If True, connect to a live Neo4j instance.
        uri, user, password : Neo4j credentials (only needed when use_neo4j=True).
    """
    if use_neo4j:
        return Neo4jKnowledgeGraph(uri, user, password)
    return InMemoryKnowledgeGraph()
