from dotenv import load_dotenv
load_dotenv()
import os
import json
import re
import sys
from pathlib import Path
from pprint import pprint
from typing import List, Dict, Union, Any, Optional, Tuple
from langchain_together import ChatTogether
try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None  # type: ignore[misc, assignment]
try:
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
except ImportError:
    ChatHuggingFace = None  # type: ignore[misc, assignment]
    HuggingFacePipeline = None  # type: ignore[misc, assignment]
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None  # type: ignore[misc, assignment]
try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None  # type: ignore[misc, assignment]
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
except ImportError:
    AutoTokenizer = None  # type: ignore[misc, assignment]
    AutoModelForCausalLM = None  # type: ignore[misc, assignment]
    pipeline = None  # type: ignore[misc, assignment]
from agentoi.agents import Agent, ByteTokenizer, create_oot_enhanced_agent
from agentoi.algorithms.graph import ConceptGraph, Ontology, EquivalentClass
from agentoi.algorithms.refinement import OntologyRefiner, RefinementOptions
from agentoi.functions.similarity import (
    SimilarityMethod,
    analyze_alignment_similarity,
    create_embedding_aware_system,
)
from agentoi.retrieval import MediaWikiSearchProvider, WebSearchProvider
from agentoi.parser import Parser
from rdflib import Graph as RDFGraph, Namespace, URIRef, Literal
from collections import defaultdict
import argparse
import torch
import time
import numpy as np
import tiktoken
from agentoi.application import PipelineConfiguration, run_pipeline
from agentoi.model_runtime import ModelRuntime

PACKAGE_ROOT = Path(__file__).resolve().parent

try:
    from agentoi.uq.scorer import UQLMConfidenceScorer
except ImportError:
    UQLMConfidenceScorer = None  # type: ignore[misc, assignment]


def _uqlm_method_name(scorer: Any) -> str:
    """Return a human-readable method name for a UQ scorer instance."""
    cls = type(scorer).__name__
    _map = {
        "UQLMConfidenceScorer": "uqlm_black_box",
        "GraphConfidenceScorer": "uqlm_graph",
        "EnsembleConfidenceScorer": "uqlm_ensemble",
    }
    return _map.get(cls, f"uqlm_{cls}")


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser()
    p.add_argument("--terms", nargs="*", help="List of terms to use with include in user query prompt.")
    p.add_argument(
        "--config_path",
        help="Namespace config",
        type=str,
        default=str(PACKAGE_ROOT / "parser/namespaces/config.json"),
    )
    p.add_argument("--version_num", type=str, default="0.1.0")
    p.add_argument("--ontology", nargs="+", type=str, default=["envo", "sweet"], help="Exactly two ontology keys to union (e.g., envo sweet).")
    p.add_argument("--tool", type=str, default="ontology_term_info")
    p.add_argument("--top_k", type=int, default=12, help="Top k nodes to return when computing nearest nodes")
    p.add_argument("--to_rdf", type=bool, default=True)
    p.add_argument("--format", type=str, default="ttl")
    p.add_argument("--load_pickle", action="store_true", help="If present, load pretrained concept graph pickle file instead of retraining new one.")
    p.add_argument("--pickle_filename", type=str, default=None, help="Name of pickle file to either save or load, depending on load_pickle flag.")
    p.add_argument("--oot_save_local", action="store_true", help="Save oot thought trace locally if enabled, saves to DynamoDB if not.")
    p.add_argument("--refine", action="store_true", help="Refine ontologies using ruleset.")
    p.add_argument("--similarity", action='store_true', help="Compute similarity score of dataset.")
    p.add_argument("--eval_alignment", action='store_true', help="Select each equivalent relation between source and target ontology, then return average similarity score.")
    p.add_argument("--use_groq", action='store_true', help="Toggle to true to use Groq, toggle to false (default) to use TogetherAI backend.")
    p.add_argument("--seed", type=int, default=time.time().__floor__())
    p.add_argument("--temperature", type=float, default=0.2, help="Temperature of model (randomness of output)")
    p.add_argument("--noise", type=int, default=0)
    p.add_argument("--datapaths", type=str, default="data/datapaths.json")
    p.add_argument(
        "--output",
        type=str,
        default="results/runs/agentoi.txt",
        help="File that receives detailed experiment output.",
    )
    p.add_argument("--enable_uqlm", action="store_true",
                   help="Enable formal UQ via UQLM Black-Box scoring (replaces ad-hoc confidence).")
    p.add_argument("--uqlm_num_responses", type=int, default=5,
                   help="Number of sampled LLM responses for UQLM consistency scoring.")
    p.add_argument("--uqlm_confidence_threshold", type=float, default=0.6,
                   help="Confidence threshold on [0,1] scale for UQLM-based decisions.")
    return p.parse_args(argv)

def _byte_len(s: Optional[str]) -> int:
    return len((s or "").encode("utf-8"))

def _token_len(s: Optional[str], tokenizer: Any) -> int:
    if not s:
        return 0
    return len(tokenizer.encode(s))

def human_join(items, *, quote=True) -> str:
    """Join a list as 'a', 'b', and 'c' (Oxford comma), handling 1/2/many."""
    items = [str(x).strip() for x in items if str(x).strip()]
    if quote:
        items = [f"'{x}'" for x in items]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"

def build_user_query(tool: str, terms: List[str], ont_names: List[str]) -> str:
    term_list_text = human_join(terms)
    ont_text = human_join([o.upper() for o in ont_names], quote=False)
    return (
        f"Use {tool} to find relations between {term_list_text} ({ont_text}), including part-of and subclass links. "
        "Place ONLY and ALL relevant concepts from concept cluster and relations in the 'term' argument (as a list). "
        "After using the tool, give a summary of the returned terms."
    )

def index_concepts_to_eqid(cg) -> dict[str, str]:
    """Map raw concept identifiers (names/IRIs) => EquivalentClass.id for gold alignment resolution."""
    idx = {}
    for eq in cg.nodes.values():
        for m in eq.equiv_concepts:
            # store both name and IRI (when present)
            if m.name:
                idx[m.name.strip()] = eq.id
            if m.iri:
                idx[m.iri.strip()] = eq.id
            # also store local parts to be forgiving
            for s in [m.name, m.iri]:
                if not s: continue
                s = s.strip().strip("<>")
                if "://" in s:
                    pos = max(s.rfind("#"), s.rfind("/"))
                    loc = s[pos+1:] if pos != -1 else s
                elif ":" in s and not s.startswith("_:"):
                    loc = s.split(":", 1)[1]
                else:
                    loc = s
                if loc:
                    idx.setdefault(loc, eq.id)
    return idx

def load_labeled_alignment_rdf(
    path: str,
    eq_index: dict[str, str],
    *,
    pos_relations: set[str] | None = None,   # relations considered "equivalent"
    neg_relations: set[str] | None = None,   # relations considered "non-equivalent"
    pos_min_measure: float | None = None,    # e.g., >= 0.5 -> positive
    neg_max_measure: float | None = 0.0,     # e.g., <= 0.0 -> negative
) -> dict[tuple[str, str], int]:
    """
    Return a dict mapping (left_eqid, right_eqid) -> label in {0,1}.
    Supports:
      - Explicit relations, e.g. '=' (positive) and '!=' (negative),
      - Or measure thresholds (>= pos_min_measure => 1; <= neg_max_measure => 0).
    Cells without a resolvable label are ignored.
    """

    pos_relations = pos_relations or {"="}
    neg_relations = neg_relations or {"!=", "≠"}  # include common variants

    g = RDFGraph()
    g.parse(path, format="xml")
    ALIGN = Namespace("http://knowledgeweb.semanticweb.org/heterogeneity/alignment")

    cells = set()
    for s, p, o in g.triples((None, ALIGN.entity1, None)):
        cells.add(s)
    for s, p, o in g.triples((None, ALIGN.entity2, None)):
        cells.add(s)

    labeled: dict[tuple[str, str], int] = {}
    for cell in cells:
        e1 = g.value(cell, ALIGN.entity1)
        e2 = g.value(cell, ALIGN.entity2)
        rel = g.value(cell, ALIGN.relation)
        meas = g.value(cell, ALIGN.measure)

        e1_uri = str(e1) if e1 is not None else ""
        e2_uri = str(e2) if e2 is not None else ""
        rel_str = (str(rel) if rel is not None else "").strip()
        try:
            meas_f = float(str(meas)) if meas is not None else None
        except Exception:
            meas_f = None

        # Map URIs -> EC ids (use local fragments as fallback)
        def to_eqid(uri: str) -> str | None:
            if not uri:
                return None
            if (eq := eq_index.get(uri)) or (eq := eq_index.get(uri.strip("<>"))):
                return eq
            loc = uri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
            return eq_index.get(loc)

        left_eq = to_eqid(e1_uri)
        right_eq = to_eqid(e2_uri)
        if not left_eq or not right_eq:
            continue

        label: int | None = None
        # Priority 1: explicit relations
        if rel_str in pos_relations:
            label = 1
        elif rel_str in neg_relations:
            label = 0

        # Priority 2: measure thresholds (used if still unlabeled or to enforce stricter gates)
        if label is None:
            if pos_min_measure is not None and meas_f is not None and meas_f >= pos_min_measure:
                label = 1
            elif neg_max_measure is not None and meas_f is not None and meas_f <= neg_max_measure:
                label = 0

        if label is not None:
            labeled[(left_eq, right_eq)] = label

    return labeled

def _first_label_or_local(node: "EquivalentClass") -> str:
    c = node.equiv_concepts[0]
    labels = c.ground_set.get("labels") or []
    if labels:
        return labels[0].strip()
    s = (c.iri or c.name or "").strip().strip("<>")
    if "://" in s:
        return s.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
    if ":" in s and not s.startswith("_:"):
        return s.split(":", 1)[1]
    return s or node.id

def _derive_prefix_from_target(node_t: "EquivalentClass") -> str | None:
    c = node_t.equiv_concepts[0]
    s = (c.iri or c.name or "").strip().strip("<>")
    if "://" not in s:
        return None
    # Keep everything up to the last '/' or '#' and add it back so prefix matching works
    if "#" in s:
        base = s.rsplit("#", 1)[0]
        return base + "#" if base else None
   
    base = s.rsplit("/", 1)[0]
    return base + "/" if base else None

def _parse_confidence_response(text: str) -> Tuple[Optional[bool], float]:
    """Parse an LLM response that may contain <Answer> and <Confidence> tags.
    Returns (answer_bool_or_None, confidence_float).
    confidence is -1.0 if not found.
    """
    answer = None
    confidence = -1.0
    ans_m = re.search(r"<Answer>\s*(YES|NO)\s*</Answer>", text, re.IGNORECASE)
    if ans_m:
        answer = ans_m.group(1).upper() == "YES"
    conf_m = re.search(r"<Confidence>\s*(\d+(?:\.\d+)?)\s*</Confidence>", text, re.IGNORECASE)
    if conf_m:
        confidence = min(float(conf_m.group(1)), 10.0)
    return answer, confidence


def _llm_terms_match(
    agent: Agent,
    term_a: str,
    term_a_syns: str,
    term_a_def: str,
    term_b: str,
    term_b_syns: str,
    term_b_def: str,
    tokenizer: Any,
    truncate_fn: Optional[Any] = None,
    max_prompt_tokens: Optional[int] = None,
    enable_confidence: bool = False,
    confidence_threshold: float = 0.0,
    uqlm_scorer: Optional[Any] = None,
) -> Tuple[bool, int, int, float]:
    """
    Ask the LLM to decide if two ontology terms refer to the same concept.
    Returns (matched, prompt_bytes, prompt_tokens, confidence).

    When ``enable_confidence`` is True **and** a ``uqlm_scorer`` is provided,
    confidence is computed via UQLM Black-Box UQ (sampling N responses and
    measuring consistency).  The confidence is on a [0, 1] scale.

    Legacy behaviour (no uqlm_scorer): falls back to self-reported <Confidence>
    tag parsing on a 0-10 scale.
    """
    if agent is None:
        return False, 0, 0, -1.0

    base_prompt = (
        "You are verifying if two ontology terms refer to the same concept. Do not use any tools for this and only use your reasoning capability. "
        "Given are two terms as well as each term's synonyms and definitions. If one or more synonyms/definitions are not provided, go off of intuition of whether the terms are most likely identical based on their grammar and semantic equivalence.\n"
        f"Term A: {term_a.lower()} -- Synonyms: {term_a_syns} -- Definition: {term_a_def}\n"
        f"Term B: {term_b.lower()} -- Synonyms: {term_b_syns} -- Definition: {term_b_def}\n"
    )

    # ── UQLM-based formal UQ ──────────────────────────────────────────
    if enable_confidence and uqlm_scorer is not None:
        prompt = base_prompt + "Answer STRICTLY with 'YES' or 'NO' only."
        if truncate_fn is not None and max_prompt_tokens is not None and max_prompt_tokens > 0:
            prompt = truncate_fn(prompt, max_prompt_tokens, tokenizer)
        prompt_len = _byte_len(prompt)
        prompt_tokens = _token_len(prompt, tokenizer)
        try:
            answer, confidence, _scores = uqlm_scorer.score_prompt(prompt)
            matched = answer and confidence >= confidence_threshold
            return (matched, prompt_len, prompt_tokens, confidence)
        except Exception:
            return (False, prompt_len, prompt_tokens, -1.0)

    # ── Legacy self-reported confidence (fallback) ─────────────────────
    if enable_confidence:
        prompt = base_prompt + (
            "Answer with your decision and confidence level.\n"
            "Format: <Answer>YES or NO</Answer><Confidence>0-10</Confidence>\n"
            "where 0 = no confidence, 10 = absolute certainty."
        )
    else:
        prompt = base_prompt + "Answer STRICTLY with 'YES' or 'NO' only."

    if truncate_fn is not None and max_prompt_tokens is not None and max_prompt_tokens > 0:
        prompt = truncate_fn(prompt, max_prompt_tokens, tokenizer)

    prompt_len = _byte_len(prompt)
    prompt_tokens = _token_len(prompt, tokenizer)
    try:
        resp = agent.invoke(prompt)
        text = (
            getattr(resp, "resp", None)
            or getattr(resp, "content", None)
            or str(resp)
        ).strip()
        text_upper = text.upper()

        if enable_confidence:
            answer, confidence = _parse_confidence_response(text)
            if answer is None:
                first = (text_upper.split() or [""])[0]
                last = (text_upper.split() or [""])[-1]
                answer = (first == "YES" or last == "YES")
            matched = answer and (confidence < 0 or confidence >= confidence_threshold)
            return (matched, prompt_len, prompt_tokens, confidence)
        else:
            first = (text_upper.split() or [""])[0]
            last = (text_upper.split() or [""])[-1]
            return (first == "YES" or last == "YES", prompt_len, prompt_tokens, -1.0)
    except Exception:
        return (False, prompt_len, prompt_tokens, -1.0)

def evaluate_alignment_binary(
    cg: "ConceptGraph",
    gold_alignment_rdf_path: str,
    *,
    top_k: int = 10,
    target_iri_prefix: str | None = None,
    exclusive: bool = True,
    pos_relations: set[str] | None = None,
    neg_relations: set[str] | None = None,
    pos_min_measure: float | None = None,
    neg_max_measure: float | None = 0.0,
    verbose: bool = False,
    agent: Agent | None = None,
    tokenizer: Any | None = None,
    truncate_fn: Any | None = None,
    max_prompt_tokens: int | None = None,
    enable_confidence: bool = False,
    confidence_threshold: float = 0.0,
    uqlm_scorer: Any | None = None,
) -> dict:
    """
    Treats evaluation as binary classification over labeled pairs (pos/neg).
    Prediction rule:
      - Predict 1 iff the gold pair's target is in top-k neighbors of the source.
      - If gold label is 0 and we predicted 1, ask LLM:
          * If LLM answers 'NO' (not equivalent), flip prediction to 0 (veto).
          * If LLM answers 'YES', keep 1 (remain FP).
    """

    eq_index = index_concepts_to_eqid(cg)
    gold_labels = load_labeled_alignment_rdf(
        gold_alignment_rdf_path,
        eq_index,
        pos_relations=pos_relations,
        neg_relations=neg_relations,
        pos_min_measure=pos_min_measure,
        neg_max_measure=neg_max_measure,
    )
    if not gold_labels:
        return {"counts": {"|gold|": 0}, "error": "No labeled gold pairs found."}

    # Group gold pairs by source for efficient single top-k per source
    gold_by_source_all: dict[str, list[tuple[str, int]]] = defaultdict(list)
    gold_by_source_pos: dict[str, set[str]] = defaultdict(set)
    for (s_eq, t_eq), y in gold_labels.items():
        gold_by_source_all[s_eq].append((t_eq, y))
        if y == 1:
            gold_by_source_pos[s_eq].add(t_eq)

    TP = TN = FP = FN = 0
    pos_rr_list = []     # for positives-only MRR
    pos_hit_sources = 0  # for positives-only Hit@k
    pos_sources = 0

    top_ids_by_source: dict[str, list[str]] = {}
    strict_times, llm_times = [], []
    llm_checks = 0
    prompt_bytes_list, prompt_tokens_list = [], []  # Track prompt sizes
    confidence_scores: list[float] = []
    confidence_yes: list[float] = []
    confidence_no: list[float] = []
    decisions_modified_by_confidence = 0

    for s_eq, pairs in gold_by_source_all.items():
        node_s = cg.nodes.get(s_eq)
        if node_s is None:
            for _, y in pairs:
                if y == 1: FN += 1
                else:      TN += 1
            continue

        term = _first_label_or_local(node_s)
        iri_prefix = target_iri_prefix
        if iri_prefix is None:
            any_pos = next((t for (t,y) in pairs if y == 1), None)
            if any_pos:
                node_t_prefix = cg.nodes.get(any_pos)
                if node_t_prefix is not None:
                    iri_prefix = _derive_prefix_from_target(node_t_prefix)

        # One retrieval per source
        t0 = time.time()
        top = cg.nearest_nodes(term, k=top_k, iri_prefix=iri_prefix, exclusive=exclusive)
        top_nodes = [n for n, _ in top]
        top_ids = [n.id for n in top_nodes]
        top_ids_by_source[s_eq] = top_ids
        strict_times.append(max(0.0, time.time() - t0))

        # Positives-only retrieval metrics (compat)
        pos_targets = [t for (t,y) in pairs if y == 1]
        if pos_targets:
            pos_sources += 1
            ranks = [top_ids.index(t)+1 for t in pos_targets if t in top_ids]
            if ranks:
                pos_hit_sources += 1
                pos_rr_list.append(1.0 / min(ranks))
            else:
                pos_rr_list.append(0.0)

        if verbose:
            print(f"SOURCE {s_eq} term='{term}'")
            print(f"  prefix={iri_prefix or '(none)'}  top{top_k}={top_ids}")

        # ---- Per-pair decisions + verbose lines with hit/via ----
        for t_eq, y in pairs:
            strict_hit = t_eq in top_ids
            pred = 1 if strict_hit else 0
            via = 'nearest' if strict_hit else 'none'
            llm_answer = None
            rank_txt = f" rank={top_ids.index(t_eq)+1}" if strict_hit else ""

            src_term = _first_label_or_local(node_s)
            node_t = cg.nodes.get(t_eq)
            tgt_term = _first_label_or_local(node_t) if node_t else ''

            src_synonyms = node_s.equiv_concepts[0].ground_set.get("related_synonyms", [])
            src_synonyms += node_s.equiv_concepts[0].ground_set.get("exact_synonyms", [])
            tgt_synonyms = node_t.equiv_concepts[0].ground_set.get("related_synonyms", [])
            tgt_synonyms += node_t.equiv_concepts[0].ground_set.get("exact_synonyms", [])
            
            src_syns = ", ".join(src_synonyms)
            tgt_syns = ", ".join(tgt_synonyms)

            src_def_list = node_s.equiv_concepts[0].ground_set.get("definitions", [])
            tgt_def_list = node_t.equiv_concepts[0].ground_set.get("definitions", [])

            if len(src_def_list) > 0:
                src_def = src_def_list[0]
            else:
                src_def = ""

            if len(tgt_def_list) > 0:
                tgt_def = tgt_def_list[0]
            else:
                tgt_def = ""
            
            # LLM veto only for suspected false positives and false negatives
            if ((y == 0 and pred == 1) or (y == 1 and pred == 0)) and agent is not None:
                llm_checks += 1
                t1 = time.time()
                matched, prompt_len, prompt_tokens, conf = _llm_terms_match(
                    agent, src_term, src_syns, src_def, tgt_term, tgt_syns, tgt_def, tokenizer,
                    truncate_fn=truncate_fn, max_prompt_tokens=max_prompt_tokens,
                    enable_confidence=enable_confidence,
                    confidence_threshold=confidence_threshold,
                    uqlm_scorer=uqlm_scorer,
                )  # YES => equivalent
                llm_answer = 'YES' if matched else 'NO'
                llm_times.append(max(0.0, time.time() - t1))
                prompt_bytes_list.append(prompt_len)
                prompt_tokens_list.append(prompt_tokens)

                if conf >= 0:
                    confidence_scores.append(conf)
                    if matched:
                        confidence_yes.append(conf)
                    else:
                        confidence_no.append(conf)

                if matched:
                    pred = 1
                else:
                    pred = 0

                via = 'llm'  # final decision came from LLM veto

            # Confusion matrix
            if y == 1 and pred == 1: TP += 1
            elif y == 1 and pred == 0: FN += 1
            elif y == 0 and pred == 1: FP += 1
            elif y == 0 and pred == 0: TN += 1

            # Verbose line: hit + via (nearest/llm/none), just like before
            hit_bool = (pred == y)
            if verbose:
                lab = 'pos' if y == 1 else 'neg'
                print(f"    pair(t={t_eq}, term={tgt_term}, label={lab}) strict_hit={strict_hit}{rank_txt} -> hit={hit_bool} via={via}")
                if via == 'llm' and agent is not None:
                    print(f"      LLM check: src='{src_term}' tgt='{tgt_term}' answer={llm_answer}")
                    if conf >= 0:
                        print(f"      LLM confidence={conf:.1f}/10")
                    print(f"      LLM time={llm_times[-1]}")
                    print(f"      LLM prompt length (in bytes)={prompt_len}")
                    print(f"      LLM prompt token length={prompt_tokens}")
                print(f"      Embedding time={strict_times[-1]}")

    # Metrics
    total = TP + TN + FP + FN
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall    = TP / (TP + FN) if (TP + FN) else 0.0
    specificity = TN / (TN + FP) if (TN + FP) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (TP + TN) / total if total else 0.0
    bal_acc  = 0.5 * (recall + specificity)
    # MCC (guard div-by-zero)
    denom = ((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5
    mcc = ((TP*TN - FP*FN) / denom) if denom else 0.0

    pos_hits_at_k = (pos_hit_sources / pos_sources) if pos_sources else 0.0
    pos_mrr = float(np.mean(pos_rr_list)) if pos_rr_list else 0.0

    strict_time_avg = float(np.mean(strict_times)) if strict_times else 0.0
    llm_time_avg    = float(np.mean(llm_times)) if llm_times else 0.0

    # Prompt size statistics
    prompt_size_stats = {}
    if prompt_bytes_list:
        prompt_size_stats = {
            "total_prompts": len(prompt_bytes_list),
            "total_bytes": sum(prompt_bytes_list),
            "total_tokens": sum(prompt_tokens_list),
            "avg_bytes": float(np.mean(prompt_bytes_list)),
            "avg_tokens": float(np.mean(prompt_tokens_list)),
            "min_bytes": min(prompt_bytes_list),
            "max_bytes": max(prompt_bytes_list),
            "min_tokens": min(prompt_tokens_list),
            "max_tokens": max(prompt_tokens_list),
        }

    confidence_stats: dict[str, Any] = {}
    if confidence_scores:
        confidence_stats = {
            "enabled": True,
            "method": _uqlm_method_name(uqlm_scorer) if uqlm_scorer is not None else "self_reported",
            "scale": "0-1" if uqlm_scorer is not None else "0-10",
            "threshold": confidence_threshold,
            "total_scored": len(confidence_scores),
            "avg": float(np.mean(confidence_scores)),
            "median": float(np.median(confidence_scores)),
            "min": float(min(confidence_scores)),
            "max": float(max(confidence_scores)),
            "std": float(np.std(confidence_scores)),
            "avg_yes": float(np.mean(confidence_yes)) if confidence_yes else None,
            "avg_no": float(np.mean(confidence_no)) if confidence_no else None,
            "count_yes": len(confidence_yes),
            "count_no": len(confidence_no),
            "decisions_modified_by_threshold": decisions_modified_by_confidence,
            "distribution": {
                str(i): sum(1 for c in confidence_scores if int(c) == i)
                for i in range(11)
            },
        }
    elif enable_confidence:
        confidence_stats = {"enabled": True, "threshold": confidence_threshold, "total_scored": 0}
    else:
        confidence_stats = {"enabled": False}

    return {
        "counts": {
            "|gold|": total,
            "TP": TP, "TN": TN, "FP": FP, "FN": FN,
            "llm_checks": llm_checks,
        },
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "balanced_accuracy": bal_acc,
            "mcc": mcc,
        },
        "positives_only_retrieval": {
            "k": top_k,
            "Hits@k": pos_hits_at_k,
            "MRR": pos_mrr,
        },
        "confidence": confidence_stats,
        "top_ids_by_source": top_ids_by_source,
        "gold_by_source": dict(gold_by_source_pos),
        "strict_time_avg": strict_time_avg,
        "llm_time_avg": llm_time_avg,
        "prompt_sizes": prompt_size_stats,
    }

def compute_similarity(
        cg: ConceptGraph,
        similarity_mode: SimilarityMethod = SimilarityMethod.HYBRID,
        similarity_threshold: float | None = None,
        gold_by_source: dict[str, set[str]] | None = None,
        top_ids_by_source: dict[str, list[str]] | None = None,
        gold_alignment_rdf_path: str | None = None,
        pos_relations: set[str] | None = None,
        min_measure: float | None = None,
        search_provider: WebSearchProvider | None = None,
    ):
    """Analyze gold-pair scores and optional per-source threshold predictions."""
    eq_index = index_concepts_to_eqid(cg)
    similarity_system = create_embedding_aware_system(
        concept_graph=cg,
        search_provider=(
            search_provider
            if search_provider is not None
            else MediaWikiSearchProvider(
                lang="en", project="wikipedia", cache_size=5000
            )
        ),
    )

    if not gold_alignment_rdf_path:
        raise ValueError("gold_alignment_rdf_path is required")
    gold = load_labeled_alignment_rdf(
        gold_alignment_rdf_path,
        eq_index,
        pos_relations=pos_relations or {"="},
        pos_min_measure=min_measure,
    )
    positive_gold = {pair for pair, label in gold.items() if label == 1}
    return analyze_alignment_similarity(
        cg,
        similarity_system,
        positive_gold,
        similarity_mode=similarity_mode,
        similarity_threshold=similarity_threshold,
        gold_by_source=gold_by_source,
        top_ids_by_source=top_ids_by_source,
    )

def _legacy_main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    original_stdout = sys.stdout

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        sys.stdout = f

        llm_model_name = os.getenv("LLM_MODEL") or "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
        GROQ_MODEL_MAP = {
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": "deepseek-r1-distill-llama-70b",
            "DeepSeek-R1-Distill-Llama-70B": "deepseek-r1-distill-llama-70b",
            "deepseek-r1-distill-llama-70b": "deepseek-r1-distill-llama-70b",
        }
        
        if os.path.isdir(llm_model_name):
            if ChatHuggingFace is None or HuggingFacePipeline is None or AutoTokenizer is None or AutoModelForCausalLM is None or pipeline is None:
                raise ImportError("Local HuggingFace model requires: pip install langchain-huggingface transformers")
            tok = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                llm_model_name,
                torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            gen_pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tok,
                max_new_tokens=512,
                do_sample=True,                  # set True if you want temperature/top_p to matter
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
                return_full_text=False,
            )
            llm = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=gen_pipe, pipeline_kwargs={"temperature": args.temperature}))
            tokenizer = tok

        elif llm_model_name.startswith("ollama:"):
            # Example: LLM_MODEL="ollama:llama3.1"
            ollama_model = llm_model_name.split(":", 1)[1].strip() or "llama3.1"

            # Default local Ollama server:
            #   http://localhost:11434
            # Can override for remote box on LAN:
            #   export OLLAMA_BASE_URL="http://192.168.1.50:11434"
            ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

            llm = ChatOllama(
                model=ollama_model,
                base_url=ollama_base_url,
                temperature=args.temperature,
                num_predict=512,  # similar role to max_new_tokens
                # validate_model_on_init=True,  # optional; can fail fast if model missing
            )

            tokenizer = ByteTokenizer()

        elif llm_model_name.startswith('gpt'):
            if ChatOpenAI is None:
                raise ImportError("OpenAI model requires: pip install langchain-openai")
            llm = ChatOpenAI(
                model=llm_model_name,
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=args.temperature,
            )
            tokenizer = tiktoken.get_encoding("cl100k_base")

        elif "gemini" in llm_model_name:
            if ChatGoogleGenerativeAI is None:
                raise ImportError("Gemini model requires: pip install langchain-google-genai")
            llm = ChatGoogleGenerativeAI(
                model=llm_model_name,   # e.g. "gemini-3-pro-preview" or "gemini-1.5-pro"
                temperature=args.temperature,
                api_key=os.getenv("GOOGLE_API_KEY")
            )

            tokenizer = ByteTokenizer()

        else:
            if not args.use_groq:
                llm = ChatTogether(
                    model=llm_model_name,
                    api_key=os.getenv("TOGETHER_API_KEY"),
                    temperature=args.temperature,
                )
            else:
                if ChatGroq is None:
                    raise ImportError("Groq model requires: pip install langchain-groq")
                groq_model = GROQ_MODEL_MAP.get(llm_model_name, llm_model_name)
                llm = ChatGroq(
                    model=groq_model,
                    api_key=os.getenv("GROQ_API_KEY"),
                    temperature=args.temperature,
                )
            if AutoTokenizer is not None:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
                except Exception:
                    tokenizer = ByteTokenizer()
            else:
                tokenizer = ByteTokenizer()

        tools_path = PACKAGE_ROOT / "agents/tools/registry/ontology_tools.json"

        agent = create_oot_enhanced_agent(
            llm=llm,
            tools_path=tools_path,
            traces_dir="results/traces/oot",
            verbose=True,
            save_traces=True,
            save_local=args.oot_save_local,
        )

        with open(args.datapaths, "r") as f:
            ontology: Dict[str, Dict[str, Dict[str, str]]] = json.load(f)

        # ENFORCE: exactly two ontologies as requested
        if len(args.ontology) > 2:
            print(f"Please provide at most two ontology keys. Got: {args.ontology}")
            return
        if len(args.ontology) == 2:
            pickle_filename = args.pickle_filename or f"concept_graph_{args.ontology[0]}_{args.ontology[1]}.pkl"
            print(f"{args.ontology[0]}-{args.ontology[1]}")
        else:
            pickle_filename = args.pickle_filename or f"concept_graph_{args.ontology[0]}.pkl"
        
        loaded_onts = []
        for ont_name in args.ontology:
            ont_key = ontology["rdf"].get(ont_name) or ontology["ttl"].get(ont_name)
            if ont_key is None:
                print(f"Invalid ontology given: {ont_name}")
                return
            
            ont = Parser(ont_key["datapath"], config_path=args.config_path, name=ont_name, version=args.version_num).to_ontology()
            loaded_onts.append(ont)

        combined_ont = None
        if len(args.ontology) == 2:
            combined_ont_name = f"{args.ontology[0]}-{args.ontology[1]}"
            if combined_ont := ontology["rdf"].get(combined_ont_name) or ontology["ttl"].get(combined_ont_name):
                ont = Parser(combined_ont["datapath"], config_path=args.config_path, name=combined_ont_name, version=args.version_num).to_ontology()
                loaded_onts.append(ont)

        merged_ont = Ontology.union_ontologies(loaded_onts)

        if args.refine:
            ontology_refiner = OntologyRefiner()
            refined_ont = ontology_refiner.refine(
                merged_ont,
                RefinementOptions(),
                fallback_if_expanded=True,
            )
            if any(
                diagnostic.code == "closure_fallback"
                for diagnostic in ontology_refiner.last_result.diagnostics
            ):
                print("Number of edges in refined ontology greater than before. Re-refining with transitive closure disabled...")

            print(f"Number of edges before refinement: {len(merged_ont.edges)}")
            print(f"Number of edges after refinement: {len(refined_ont.edges)}")
        else:
            refined_ont = None

        if args.noise > 0:
            noise = args.noise / 100
            if refined_ont:
                refined_ont, _ = refined_ont.apply_bart_noise(cfg={"p_record":noise, "p_field":0.8, "fields":["labels","alt_labels","related_synonyms","exact_synonyms"]}, seed=args.seed)
            else:
                merged_ont, _ = merged_ont.apply_bart_noise(cfg={"p_record":noise, "p_field":0.8, "fields":["labels","alt_labels","related_synonyms","exact_synonyms"]}, seed=args.seed)
        
        if args.load_pickle:
            cg = ConceptGraph.load_pickle(filename=pickle_filename)
        else:
            cg = ConceptGraph(nodes=[], edges=[])
            cg.build_from_ontology(refined_ont or merged_ont)              # collapse equivalents + add edges
            cg.compute_all_embeddings(alpha=0.5)     # graph + text + fused embeddings
            cg.to_pickle(filename=pickle_filename)

        if args.eval_alignment and combined_ont:
            # Optionally create UQLM scorer for formal uncertainty quantification
            _uqlm_scorer = None
            if args.enable_uqlm and UQLMConfidenceScorer is not None:
                _uqlm_scorer = UQLMConfidenceScorer(
                    llm=llm,
                    num_responses=args.uqlm_num_responses,
                )
                print(f"[UQLM] Formal UQ enabled: Black-Box exact_match, "
                      f"num_responses={args.uqlm_num_responses}, "
                      f"threshold={args.uqlm_confidence_threshold}")

            res = evaluate_alignment_binary(
                cg,
                gold_alignment_rdf_path=combined_ont.get("fullset_datapath") or combined_ont.get("datapath"),
                top_k=args.top_k,
                target_iri_prefix=None,     # or a concrete prefix like "http://sweetontology.net/"
                exclusive=True,
                # Choose ONE or both of these strategies:
                pos_relations={"="},        # consider '=' as positive
                neg_relations={"!=", "≠"},  # consider '!=' (or ≠) as negative
                # and/or use measure gates:
                pos_min_measure=0.5,        # >= 0.5 => positive
                neg_max_measure=0.0,        # <= 0.0 => negative
                verbose=True,
                agent=agent if args.refine else None,  # keep your earlier conditional if you want,
                tokenizer=tokenizer if args.refine else None,
                enable_confidence=args.enable_uqlm,
                confidence_threshold=args.uqlm_confidence_threshold,
                uqlm_scorer=_uqlm_scorer,
            )

            cm = res["counts"]; m = res["metrics"]; pos = res["positives_only_retrieval"]
            print("=== Binary Alignment Evaluation (with LLM veto for negatives) ===")
            print(f"Gold pairs: {cm['|gold|']}  TP={cm['TP']} TN={cm['TN']} FP={cm['FP']} FN={cm['FN']}  LLM_checks={cm['llm_checks']}")
            print(f"Acc={m['accuracy']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  Spec={m['specificity']:.4f}  F1={m['f1']:.4f}  BalAcc={m['balanced_accuracy']:.4f}  MCC={m['mcc']:.4f}")
            print(f"[Positives-only] HIT@{pos['k']}={pos['Hits@k']:.4f}  MRR={pos['MRR']:.4f}")
            print(f"Embedding avg time: {res['strict_time_avg']:.4f}s")
            print(f"LLM-veto avg time: {res['llm_time_avg']:.4f}s")

        if args.similarity and combined_ont:
            evaluation = res if args.eval_alignment else {}
            sim_res = compute_similarity(
                cg,
                similarity_mode=SimilarityMethod.HYBRID,
                similarity_threshold=None,
                gold_by_source=evaluation.get("gold_by_source"),
                top_ids_by_source=evaluation.get("top_ids_by_source"),
                gold_alignment_rdf_path=combined_ont.get("fullset_datapath")
                or combined_ont["datapath"],
                pos_relations={"="},
                min_measure=None,
            )
            print(f"similarity_avgs_over_gold: {sim_res['similarity_avgs_over_gold']}")

        # Example: Guided prompt lookup (with given tool from arguments using concept graph and text embedding guided prompt)
        if args.terms and args.tool:
            terms = args.terms
            user_query = build_user_query(tool=args.tool, terms=terms, ont_names=args.ontology)
            guided_prompt = cg.make_prompt_for_query(user_query, terms, k=args.top_k, max_edges=60)
            print(guided_prompt)
            
            out = agent.invoke(guided_prompt, return_trace=True)
            pprint(out["trace"])
            pprint(out["result"])
    
    sys.stdout = original_stdout


def _evaluate(
    graph: Any,
    alignment_path: str,
    configuration: PipelineConfiguration,
    runtime: ModelRuntime,
    agent: Agent,
) -> dict[str, Any]:
    scorer = None
    if configuration.enable_uqlm and UQLMConfidenceScorer is not None:
        scorer = UQLMConfidenceScorer(
            llm=runtime.llm,
            num_responses=configuration.uqlm_num_responses,
        )
        print(
            "[UQLM] Formal UQ enabled: Black-Box exact_match, "
            f"num_responses={configuration.uqlm_num_responses}, "
            f"threshold={configuration.uqlm_confidence_threshold}"
        )
    return evaluate_alignment_binary(
        graph,
        gold_alignment_rdf_path=alignment_path,
        top_k=configuration.top_k,
        target_iri_prefix=None,
        exclusive=True,
        pos_relations={"="},
        neg_relations={"!=", "≠"},
        pos_min_measure=0.5,
        neg_max_measure=0.0,
        verbose=True,
        agent=agent if configuration.refine else None,
        tokenizer=runtime.tokenizer if configuration.refine else None,
        enable_confidence=configuration.enable_uqlm,
        confidence_threshold=configuration.uqlm_confidence_threshold,
        uqlm_scorer=scorer,
    )


def _compute_similarity(
    graph: Any, evaluation: dict[str, Any], alignment_path: str
) -> dict[str, Any]:
    return compute_similarity(
        graph,
        similarity_mode=SimilarityMethod.HYBRID,
        similarity_threshold=None,
        gold_by_source=evaluation.get("gold_by_source"),
        top_ids_by_source=evaluation.get("top_ids_by_source"),
        gold_alignment_rdf_path=alignment_path,
        pos_relations={"="},
        min_measure=None,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configuration = PipelineConfiguration(
        package_root=PACKAGE_ROOT,
        graph_kind="graph",
        ontology_names=tuple(args.ontology),
        config_path=args.config_path,
        version=args.version_num,
        datapaths=args.datapaths,
        output=args.output,
        terms=tuple(args.terms or ()),
        tool=args.tool,
        top_k=args.top_k,
        load_pickle=args.load_pickle,
        pickle_filename=args.pickle_filename,
        save_local_traces=args.oot_save_local,
        refine=args.refine,
        similarity=args.similarity,
        evaluate_alignment=args.eval_alignment,
        seed=args.seed,
        noise_percent=args.noise,
        temperature=args.temperature,
        use_groq=args.use_groq,
        enable_uqlm=args.enable_uqlm,
        uqlm_num_responses=args.uqlm_num_responses,
        uqlm_confidence_threshold=args.uqlm_confidence_threshold,
    )
    run_pipeline(
        configuration,
        evaluate=_evaluate,
        compute_similarity=_compute_similarity,
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())