from __future__ import annotations
import os, json
from pathlib import Path
from typing import Optional, Dict, Any, Sequence, List, Set, Tuple
from agentoi.agents.runtime.dynamo_queries import (
    table as ddb_table,
    find_latest_run_id,
    list_latest_run_ids,
)
from agentoi.agents.runtime.persistence import load_run_graph
from agentoi.agents.ontology_of_thought.graph import ThoughtGraph
from agentoi.agents.ontology_of_thought.models import Edge, Thought
from agentoi.agents.ontology_of_thought.serialization import to_jsonable

class OOTReconstructor:
    """
    Rebuilds exact OOT graphs from DynamoDB-stored runs.
    - Does NOT deduplicate or reshape nodes: faithfully restores Thoughts & Edges.
    - Can merge multiple runs into a single ThoughtGraph (IDs are preserved).
    - Keeps run metadata and raw payload snapshots for inspection/export.
    """

    def __init__(
        self,
        *,
        table: str,
        region: Optional[str] = None,
        profile: Optional[str] = None,
        pk: Optional[str] = None,
        sk: Optional[str] = None,
    ):
        self.table_name = table
        self.region = region or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-2"
        self.profile = profile

        # Optional key overrides for restricted IAM environments
        if pk is not None:
            os.environ.setdefault("DYNAMO_PK", pk)
        if sk is not None:
            os.environ.setdefault("DYNAMO_SK", sk)

        self._tbl = ddb_table(self.table_name, region=self.region, profile=self.profile)

        # caches
        self._payloads: Dict[str, Dict[str, Any]] = {}
        self._runs_meta: List[Dict[str, Any]] = []
        self._graph: Optional[ThoughtGraph] = None
        self._loaded_run_ids: List[str] = []

    # ----------- internal helpers -----------

    def _ensure_graph(self) -> ThoughtGraph:
        if self._graph is None:
            self._graph = ThoughtGraph()
        return self._graph

    def _merge_payload(self, run_id: str, payload: Dict[str, Any]) -> None:
        """
        Merge a single Dynamo payload (one run) into the in-memory ThoughtGraph.
        Preserves exact node and edge structure. Skips exact ID dupes safely.
        """
        graph = (payload or {}).get("graph") or {"thoughts": [], "edges": []}
        tg = self._ensure_graph()

        # nodes
        for tdict in graph.get("thoughts", []):
            # Convert dict -> Thought (preserves enums and fields)
            try:
                t_obj = Thought.from_dict(tdict)
            except Exception:
                # fallback if schema drifted slightly
                t_obj = Thought(**tdict)  # type: ignore[arg-type]

            # ensure meta has run_id (older data may miss it at the node level)
            t_obj.meta = dict(t_obj.meta or {})
            t_obj.meta.setdefault("run_id", run_id)

            # Add if not present; preserve original ID, adjacency built later via edges
            if t_obj.id not in tg.thoughts:
                tg.thoughts[t_obj.id] = t_obj

        # edges
        for edict in graph.get("edges", []):
            try:
                e_obj = Edge(**edict)  # Edge dataclass accepts the stored schema
            except TypeError:
                # If you added Edge.from_dict() in your codebase, prefer that
                try:
                    e_obj = Edge.from_dict(edict)  # type: ignore[attr-defined]
                except Exception:
                    raise

            # ensure meta has run_id
            e_obj.meta = dict(e_obj.meta or {})
            e_obj.meta.setdefault("run_id", run_id)

            # sanity: only add if endpoints exist
            if e_obj.src in tg.thoughts and e_obj.dst in tg.thoughts:
                if e_obj.id not in tg.edges:
                    tg.edges[e_obj.id] = e_obj
                    tg.out_adj[e_obj.src].append(e_obj.id)
                    tg.in_adj[e_obj.dst].append(e_obj.id)

    # ----------- public API -----------

    def load_by_run_id(self, run_id: str, *, reset: bool = True) -> "OOTReconstructor":
        """Replace current state with the graph from a single run (or merge if reset=False)."""
        if reset:
            self._graph = ThoughtGraph()
            self._payloads.clear()
            self._runs_meta.clear()
            self._loaded_run_ids.clear()

        payload = load_run_graph(table_name=self.table_name, run_id=run_id)
        self._payloads[run_id] = payload

        run_meta = dict(payload.get("run") or {})
        if run_meta:
            self._runs_meta.append(run_meta)

        self._merge_payload(run_id, payload)
        self._loaded_run_ids.append(run_id)
        return self

    def load_by_run_ids(self, run_ids: Sequence[str], *, reset: bool = True) -> "OOTReconstructor":
        """Load and merge multiple runs into a single ThoughtGraph (preserving IDs)."""
        if reset:
            self._graph = ThoughtGraph()
            self._payloads.clear()
            self._runs_meta.clear()
            self._loaded_run_ids.clear()

        for rid in run_ids:
            payload = load_run_graph(table_name=self.table_name, run_id=rid)
            self._payloads[rid] = payload

            run_meta = dict(payload.get("run") or {})
            if run_meta:
                self._runs_meta.append(run_meta)

            self._merge_payload(rid, payload)
            self._loaded_run_ids.append(rid)

        return self

    def load_latest(self) -> "OOTReconstructor":
        """Load the most recent run (based on CreatedAt/created_at ordering)."""
        rid = find_latest_run_id(self._tbl)
        if not rid:
            raise RuntimeError("No runs found in table.")
        return self.load_by_run_id(rid, reset=True)

    def load_latest_n(self, n: int = 3) -> "OOTReconstructor":
        """Load and merge the N most recent runs."""
        ids = list_latest_run_ids(self._tbl, limit=max(1, n))
        if not ids:
            raise RuntimeError("No runs found in table.")
        return self.load_by_run_ids(ids, reset=True)

    # ----------- accessors / exports -----------

    def run_meta(self) -> Dict[str, Any]:
        """For single-run loads, returns that meta. For multi-run, returns {'runs': [...]}."""
        if len(self._runs_meta) <= 1:
            return (self._runs_meta[0] if self._runs_meta else {})
        return {"runs": self._runs_meta}

    def run_ids(self) -> List[str]:
        """Run IDs included in the current reconstruction (order loaded)."""
        return list(self._loaded_run_ids)

    def graph(self) -> ThoughtGraph:
        if self._graph is None:
            raise RuntimeError("No graph loaded. Call one of the load_* methods first.")
        return self._graph

    def to_dict(self) -> Dict[str, Any]:
        g = self.graph()
        return {
            "run": self.run_meta(),
            "graph": g.to_dict() if hasattr(g, "to_dict") else {},
            "runs_loaded": self.run_ids(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(to_jsonable(self.to_dict()), ensure_ascii=False, indent=indent)

    def save_json(self, path: str, indent: int = 2) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(to_jsonable(self.to_dict()), f, ensure_ascii=False, indent=indent)

    # ----------- convenience: memory traces per run -----------

    def memory_trace_for(self, run_id: str, prefer_edges: bool = True):
        """
        Compute the run-scoped path (memory trace) from the reconstructed graph,
        using the same semantics as ThoughtGraph.get_run_path().
        """
        g = self.graph()
        return g.get_run_path(run_id, prefer_edges=prefer_edges)

    def memory_traces(self, prefer_edges: bool = True) -> Dict[str, Any]:
        """Return {run_id: Path|None} for all runs loaded."""
        g = self.graph()
        out: Dict[str, Any] = {}
        for rid in self._loaded_run_ids:
            mt = g.get_run_path(rid, prefer_edges=prefer_edges)
            out[rid] = (mt.to_trace() if mt else None)
        return out
    
    # ----------- enhanced functionality -----------
    
    def print_summary(self) -> None:
        """Print a comprehensive summary of the loaded thought graph"""
        g = self.graph()
        print("\n" + "="*60)
        print("THOUGHT GRAPH SUMMARY")
        print("="*60)
        print(f"Loaded run IDs: {self._loaded_run_ids}")
        print(f"Number of thoughts (nodes): {len(g.thoughts)}")
        print(f"Number of edges: {len(g.edges)}")
        
        # Count by thought type
        from collections import defaultdict
        type_counts = defaultdict(int)
        for thought in g.thoughts.values():
            type_counts[thought.type.value] += 1
        
        print(f"\nThought types:")
        for thought_type, count in type_counts.items():
            print(f"  - {thought_type}: {count}")
        
        # Show some example thoughts
        print(f"\nExample thoughts (first 5):")
        for i, (thought_id, thought) in enumerate(list(g.thoughts.items())[:5]):
            content = thought.question or thought.answer or thought.content or "No content"
            content_preview = content[:100] + "..." if len(content) > 100 else content
            print(f"  {i+1}. [{thought.type.value}] {thought_id}: {content_preview}")
        
        # Show some example edges
        print(f"\nExample edges (first 5):")
        for i, (edge_id, edge) in enumerate(list(g.edges.items())[:5]):
            kind_str = edge.kind.value if hasattr(edge.kind, 'value') else str(edge.kind)
            print(f"  {i+1}. {edge.src} --[{kind_str}]--> {edge.dst}")
    
    def search_thoughts(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for thoughts containing the query string"""
        g = self.graph()
        results = []
        query_lower = query.lower()
        
        for thought_id, thought in g.thoughts.items():
            # Search in thought ID
            if query_lower in thought_id.lower():
                results.append({
                    "type": "thought_id",
                    "thought_id": thought_id,
                    "thought": thought,
                    "match": thought_id
                })
                continue
            
            # Search in content fields
            content_fields = [
                ("question", thought.question),
                ("answer", thought.answer),
                ("content", thought.content),
                ("name", getattr(thought, 'name', None))
            ]
            
            for field_name, field_value in content_fields:
                if field_value and query_lower in field_value.lower():
                    results.append({
                        "type": f"content_{field_name}",
                        "thought_id": thought_id,
                        "thought": thought,
                        "field": field_name,
                        "match": field_value
                    })
                    break
            
            # Search in tags
            if thought.tags:
                for tag in thought.tags:
                    if query_lower in tag.lower():
                        results.append({
                            "type": "tag",
                            "thought_id": thought_id,
                            "thought": thought,
                            "match": tag
                        })
                        break
        
        return results[:max_results]
    
    def print_search_results(self, results: List[Dict[str, Any]]) -> None:
        """Print search results in a formatted way"""
        if not results:
            print("No results found.")
            return
        
        print(f"\nFound {len(results)} results:")
        print("-" * 50)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. Type: {result['type']}")
            print(f"   Thought ID: {result['thought_id']}")
            thought = result['thought']
            print(f"   Type: {thought.type.value}")
            if 'field' in result:
                print(f"   Field: {result['field']}")
            print(f"   Match: {result['match']}")
            print()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the thought graph"""
        g = self.graph()
        
        # Basic counts
        stats = {
            "total_thoughts": len(g.thoughts),
            "total_edges": len(g.edges),
            "loaded_run_ids": self._loaded_run_ids,
            "run_count": len(self._loaded_run_ids)
        }
        
        # Count by thought type
        from collections import defaultdict
        type_counts = defaultdict(int)
        for thought in g.thoughts.values():
            type_counts[thought.type.value] += 1
        stats["thought_types"] = dict(type_counts)
        
        # Count by edge type
        edge_type_counts = defaultdict(int)
        for edge in g.edges.values():
            kind_str = edge.kind.value if hasattr(edge.kind, 'value') else str(edge.kind)
            edge_type_counts[kind_str] += 1
        stats["edge_types"] = dict(edge_type_counts)
        
        # Content statistics
        total_content_length = 0
        content_count = 0
        for thought in g.thoughts.values():
            content = thought.question or thought.answer or thought.content
            if content:
                total_content_length += len(content)
                content_count += 1
        
        stats["avg_content_length"] = total_content_length / content_count if content_count > 0 else 0
        stats["total_content_length"] = total_content_length
        
        return stats
    
    def print_statistics(self) -> None:
        """Print detailed statistics about the thought graph"""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("THOUGHT GRAPH STATISTICS")
        print("="*60)
        print(f"Total thoughts: {stats['total_thoughts']}")
        print(f"Total edges: {stats['total_edges']}")
        print(f"Loaded runs: {stats['run_count']}")
        print(f"Run IDs: {stats['loaded_run_ids']}")
        
        print(f"\nThought types:")
        for thought_type, count in stats['thought_types'].items():
            print(f"  - {thought_type}: {count}")
        
        print(f"\nEdge types:")
        for edge_type, count in stats['edge_types'].items():
            print(f"  - {edge_type}: {count}")
        
        print(f"\nContent statistics:")
        print(f"  - Average content length: {stats['avg_content_length']:.1f} characters")
        print(f"  - Total content length: {stats['total_content_length']} characters")
    
    def list_available_run_ids(self, limit: int = 10) -> List[str]:
        """List available run IDs in the DynamoDB table"""
        try:
            from agentoi.agents.runtime.dynamo_queries import list_latest_run_ids
            return list_latest_run_ids(self._tbl, limit=limit)
        except Exception as e:
            print(f"Error listing run IDs: {e}")
            return []
    
    def print_available_runs(self, limit: int = 10) -> None:
        """Print available run IDs in the DynamoDB table"""
        run_ids = self.list_available_run_ids(limit)
        if not run_ids:
            print("No run IDs found in DynamoDB table")
            return
        
        print(f"\nAvailable run IDs (showing first {limit}):")
        print("-" * 50)
        for i, run_id in enumerate(run_ids, 1):
            print(f"{i}. {run_id}")
    
    def export_to_networkx(self, max_nodes: int = 100):
        """Export the thought graph to NetworkX format for visualization"""
        try:
            import networkx as nx
        except ImportError:
            print("NetworkX not available. Install with: pip install networkx")
            return None
        
        g = self.graph()
        G = nx.DiGraph()
        
        # Add nodes
        node_count = 0
        for thought_id, thought in g.thoughts.items():
            if node_count >= max_nodes:
                break
            
            # Create label from thought content
            content = thought.question or thought.answer or thought.content or "No content"
            label = f"{thought_id}\n[{thought.type.value}]\n{content[:50]}..."
            
            G.add_node(thought_id,
                      label=label,
                      thought_type=thought.type.value,
                      content=content)
            node_count += 1
        
        # Add edges
        edge_count = 0
        for edge_id, edge in g.edges.items():
            if edge_count >= max_nodes * 2:
                break
            
            if edge.src in G and edge.dst in G:
                kind_str = edge.kind.value if hasattr(edge.kind, 'value') else str(edge.kind)
                G.add_edge(edge.src, edge.dst,
                          kind=kind_str,
                          decision=edge.decision)
                edge_count += 1
        
        return G
    
    def visualize(self, output_file: Optional[str] = None, max_nodes: int = 50, layout: str = "spring") -> str:
        """Create a visualization of the thought graph and save to visualization folder"""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError:
            print("Matplotlib or NetworkX not available. Install with: pip install matplotlib networkx")
            return None
        
        G = self.export_to_networkx(max_nodes)
        if G is None or len(G.nodes) == 0:
            print("No nodes to visualize.")
            return None
        
        # Create visualization directory if it doesn't exist
        viz_dir = Path("visualizations")
        viz_dir.mkdir(exist_ok=True)
        
        # Generate output filename if not provided
        if not output_file:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            table_name = getattr(self, 'table_name', 'unknown_table')
            output_file = viz_dir / f"thought_graph_{table_name}_{timestamp}.png"
        else:
            # Ensure the output file is in the visualization directory
            output_file = viz_dir / Path(output_file).name
        
        plt.figure(figsize=(15, 10))
        
        # Choose layout
        if layout == "spring":
            pos = nx.spring_layout(G, k=3, iterations=50)
        elif layout == "hierarchical":
            try:
                pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
            except:
                pos = nx.spring_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                              node_color='lightblue',
                              node_size=500,
                              alpha=0.7)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos,
                              edge_color='gray',
                              arrows=True,
                              arrowsize=20,
                              alpha=0.6)
        
        # Draw labels
        labels = {node: data['label'] for node, data in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title(f"Thought Graph Visualization ({len(G.nodes)} nodes, {len(G.edges)} edges)")
        plt.axis('off')
        
        # Save the visualization
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        
        print(f"Graph visualization saved to: {output_file}")
        return str(output_file)