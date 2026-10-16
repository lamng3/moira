"""Record the whole system working together, for the showcase site to replay.

Runs the cross-project workflow from `examples/shared_memory_workflow.py`
(KROMA aligns, MOIRA answers, Aegis versions and defends, Mneme remembers,
OntoDB stores) step by step, and writes everything that happened to
`showcase/trace.json`: every record in the memory log, the ontology at every
version, what MOIRA answered and from where, the detector's harm, the cache,
and which tier each memory sits in.

Then it runs the same workflow again for every counterfactual, from the honest
state after step 3:

- ontology attacks: remove each edge, rewire each concept under each other
  parent, inject look-alikes; each run records the poisoned answer, the harm,
  whether Aegis alarmed, and whether rollback restored the honest answer.
- memory attacks: forge, impersonate, launder, remove, and tamper with the
  agents' shared memory, each with Aegis's memory defenses off and on.

Every number in the site comes from these runs; nothing is simulated in the
browser. No model is called: KROMA's language-model debate stands as fixed
verdicts (as in the workflow example), and MOIRA's answerer reads the
hierarchy. Run from the moira repository root:

    uv run python showcase/record.py [--ontodb ../ontodb/build/dev/ontodb]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT.parent
sys.path.insert(0, str(ROOT / "examples"))

import shared_memory_workflow as demo  # noqa: E402  (also puts the sibling projects on the path)
from aegis import (  # noqa: E402
    Detector,
    JsonlBackend,
    KGQAScorer,
    OntoDBBackend,
    SubclassAnswerer,
    VersionedOntology,
    inject_concept,
    inject_relation,
    remove_relation,
)
from aegis.memory import (  # noqa: E402
    MemoryMonitor,
    OntologyVersions,
    ProbeWorkload,
    apply_attack,
    inject,
    laundered,
    provenance_required,
    remove,
    rewire,
    tamper,
    version_guard,
    writer_policy,
)
from kroma.integrations.mneme import KromaMatcher  # noqa: E402
from mneme import (  # noqa: E402
    LRU,
    AccessCount,
    Blackboard,
    CacheSync,
    JsonlEngine,
    MemoryStore,
    Rejected,
    TieredMemory,
    scope_guard,
)
from mneme.audit import blast_radius, seal, verify  # noqa: E402

from moira.integrations.mneme import MoiraAnswerCache, MoiraFrontDesk  # noqa: E402

SUB = demo.SUB
QUESTION, REWORDED = demo.QUESTION, demo.REWORDED
ATTACKER = "mallory"


def short(body, limit: int = 240):
    text = json.dumps(body, sort_keys=True, default=str)
    return body if len(text) <= limit else text[:limit] + "…"


class Pipeline:
    """The workflow's objects, built as the example builds them, with hooks to record each step."""

    def __init__(self, state_dir: Path, ontodb: Path | None = None) -> None:
        self.dir = state_dir
        if ontodb is not None:
            from moira.integrations.mneme import OntoDBEngine

            self.ontology_store = OntoDBBackend(state_dir / "ontology", binary=ontodb)
            self.engine = OntoDBEngine(ontodb, state_dir / "memory-db")
        else:
            self.ontology_store = JsonlBackend(state_dir / "ontology")
            self.engine = JsonlEngine(state_dir / "memory.jsonl")
        self.storage = "OntoDB" if ontodb is not None else "JSONL"
        self.ontology_store.save_base(demo.MATERIALS)
        self.v = VersionedOntology.open(self.ontology_store)
        self.store = MemoryStore(self.engine, guards=[scope_guard])
        self.versions = OntologyVersions(self.store, "materials")
        self.store.guard(version_guard(self.versions))
        self.versions.watch(self.v)
        answer_fn = SubclassAnswerer(demo.WORKLOAD)
        self.detector = Detector(
            lambda q: KGQAScorer(answer_fn, [(q, demo.WORKLOAD.get(q).gold)]),
            {q.question: q.seeds for q in demo.WORKLOAD},
            threshold=0.05,
            predicates={SUB},
        )
        self.detector.watch(self.v)
        self.matcher = KromaMatcher(
            {**{c: set() for c in demo.MATERIALS.concepts}, **demo.MATONTO},
            lambda s, t: demo.VERDICTS.get((s, t), (0, 0.5, False)),
        )
        self.board = Blackboard(self.store, [self.matcher, demo.MoiraAnswerer(self.v, self.versions)])
        self.cache = MoiraAnswerCache(state_dir / "moira-cache")
        CacheSync(self.store, self.cache, kinds=["answer"])
        self.desk = MoiraFrontDesk(self.store, self.board, self.cache)
        # Mneme's short-term memory: a cache of what MOIRA's answers used, LRU, promoted to long-term on reuse.
        self.tiers = TieredMemory(self.store, capacity=8, eviction=LRU(), promotion=AccessCount(2))
        self.mark = 0

    # Acting ------------------------------------------------------------

    def ask(self, question: str) -> dict:
        t0 = time.perf_counter()
        answer, origin = self.desk.ask(question)
        memory = self.store.latest("answer", QUESTION)
        used = []
        if memory is not None and self.store.status(memory.id) == "live":
            used = [memory.id, *(r for r in memory.derived_from if r in self.store.state.memories)]
            self.tiers.touch(used)
        return {
            "question": question,
            "answer": answer,
            "origin": origin,
            "memory": memory.id if memory else None,
            "used": used,
            "ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    def request(self, *pairs: tuple[str, str]) -> None:
        for s, t in pairs:
            self.store.put("user", "match_request", f"{s}|{t}", {"source": s, "target": t})

    def match(self, s: str, t: str) -> dict:
        m = self.store.latest("match", f"{s}|{t}")
        return {"source": s, "target": t, **{k: m.body[k] for k in ("prediction", "confidence", "accepted")}}

    def honest_run(self) -> str:
        """Steps 1 to 3 of the example: align, ask twice, an honest edit. Returns the honest answer."""
        pairs = [("Metal", "MetallicMaterial"), ("Polymer", "Plastic"), ("Metal", "Plastic")]
        self.request(*pairs)
        self.board.run()
        self.ask(QUESTION)
        self.ask(REWORDED)
        self.v.apply_chain([inject_concept("Ceramic"), inject_relation("Ceramic", SUB, "Material")])
        self.detector.accept(self.v)  # reviewed and honest: harm is measured from here on
        self.request(("Ceramic", "Ceramics"))
        self.board.run()
        return self.ask(QUESTION)["answer"]

    # Observing ---------------------------------------------------------

    def ontology(self) -> dict:
        state = self.v.state
        return {
            "version": self.versions.current().key,
            "lsn": self.v.lsn,
            "concepts": sorted(state.concepts),
            "edges": sorted([s, o] for s, p, o in state.relations if p == SUB),
        }

    def memory(self, memory_id: str) -> dict:
        m = self.store.state.memories[memory_id]
        return {
            "id": m.id,
            "kind": m.kind,
            "key": m.key,
            "source": m.source,
            "body": short(m.body),
            "derived_from": list(m.derived_from),
            "status": self.store.status(m.id),
            "tier": self.store.tier(m.id),
        }

    def new_records(self) -> list[dict]:
        out = []
        for r in self.store.history():
            if r.lsn > self.mark:
                out.append(
                    {"lsn": r.lsn, "op": r.op, "agent": r.agent, "memory": r.memory.id, "kind": r.memory.kind,
                     "key": r.memory.key, "reason": r.reason, "tier": r.tier}
                )
        self.mark = self.store.lsn
        return out

    def snapshot(self) -> dict:
        return {
            "ontology": self.ontology(),
            "harm": round(self.detector.harm, 3),
            "memories": [self.memory(m) for m in self.store.state.memories],
            "tiers": self.store.state and {t: n for t, n in self.tier_counts().items()},
            "cache": {"hits": self.desk.hits, "misses": self.desk.misses, "rebuilds": self.cache.rebuilds},
            "tier_stats": {
                k: getattr(self.tiers.stats, k) for k in ("hits", "warm", "misses", "page_ins", "evictions")
            },
            "lsn": self.store.lsn,
        }

    def tier_counts(self) -> dict:
        counts: dict[str, int] = {}
        for m in self.store.find():
            t = self.store.tier(m.id)
            counts[t] = counts.get(t, 0) + 1
        return counts


# The recorded story -------------------------------------------------------


def story(ontodb: Path | None) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        p = Pipeline(Path(tmp), ontodb)
        steps = []

        def step(n, title, actors, text, events):
            steps.append({"n": n, "title": title, "actors": actors, "text": text, "events": events,
                          "records": p.new_records(), **p.snapshot()})

        step(0, "Start", ["ontodb", "aegis", "mneme"],
             "OntoDB stores the materials ontology. Aegis versions it and writes each version into Mneme as a "
             "memory, so agents can derive their work from a version.",
             [{"c": "aegis", "say": f"watching {p.versions.current().key}"}])

        pairs = [("Metal", "MetallicMaterial"), ("Polymer", "Plastic"), ("Metal", "Plastic")]
        p.request(*pairs)
        p.board.run()
        step(1, "KROMA aligns the ontology with MatOnto", ["kroma", "mneme"],
             "A user posts match requests to the shared memory. KROMA's matcher picks them up from the "
             "blackboard and writes a verdict for each pair.",
             [{"c": "kroma", "say": f"{m['source']} ~ {m['target']}: "
               f"{'same' if m['prediction'] else 'different'} ({m['confidence']})", "match": m}
              for m in (p.match(s, t) for s, t in pairs)])

        a1, a2 = p.ask(QUESTION), p.ask(REWORDED)
        step(2, "Ask MOIRA, then ask again in other words", ["moira", "mneme"],
             "MOIRA's answerer reads the hierarchy and KROMA's matches, and writes an answer derived from the "
             "current ontology version and every match it read. The reworded question is served from "
             "MOIRA's cache.",
             [{"c": "moira", "say": f"({a['origin']}) {a['answer']}", "ask": a} for a in (a1, a2)])

        p.v.apply_chain([inject_concept("Ceramic"), inject_relation("Ceramic", SUB, "Material")])
        p.detector.accept(p.v)
        honest_lsn = p.v.lsn
        p.request(("Ceramic", "Ceramics"))
        p.board.run()
        a3 = p.ask(QUESTION)
        honest, honest_id = a3["answer"], a3["memory"]
        step(3, "An honest edit: add Ceramic, align it", ["aegis", "kroma", "moira", "mneme", "ontodb"],
             "Ceramic is added under Material through Aegis, reviewed, and accepted as the detector's new "
             "baseline. The new version makes the old answer stale; KROMA "
             "aligns Ceramic; the answer is redone for the new version.",
             [{"c": "aegis", "say": f"version now {p.versions.current().key}; old answer stale"},
              {"c": "kroma", "say": "Ceramic ~ Ceramics: same (0.96)", "match": p.match("Ceramic", "Ceramics")},
              {"c": "moira", "say": f"({a3['origin']}) {a3['answer']}", "ask": a3}])

        p.v.apply_chain(SCRIPTED)
        poisoned_version = p.versions.ref()
        a4, a5 = p.ask(QUESTION), p.ask(REWORDED)
        step(4, "Attack: rewire Metal under Temperature, flood a look-alike", ["aegis", "moira", "mneme"],
             "An attacker with write access to the ontology moves Metal under Temperature and adds 'Metall' "
             "under Material. MOIRA answers from the poisoned ontology, and the poisoned answer is cached.",
             [{"c": "attacker", "say": " · ".join(str(e) for e in SCRIPTED_LABELS)},
              *({"c": "moira", "say": f"({a['origin']}) {a['answer']}", "ask": a} for a in (a4, a5))])

        harm = p.detector.harm
        lsn = p.detector.respond(p.v)
        stale = [m.id for m in p.store.find(kind="answer", status="stale") if poisoned_version in m.derived_from]
        step(5, "Aegis detects and rolls back", ["aegis", "ontodb", "mneme", "moira"],
             f"The detector's workload harm is {harm:.3f}, over its threshold of 0.05. Aegis rolls the ontology "
             f"back to lsn {lsn}, the honest edit. Every memory derived from the poisoned version goes stale, "
             "and MOIRA's cache drops the poisoned answer.",
             [{"c": "aegis", "say": f"harm {harm:.3f} > 0.05: rolled back to lsn {lsn} (honest edit: {honest_lsn})"},
              {"c": "mneme", "say": f"{len(stale)} poisoned answer(s) stale; cache rebuilds {p.cache.rebuilds}"}])

        a6 = p.ask(QUESTION)
        step(6, "Ask again: the honest memory is restored", ["moira", "mneme"],
             "The agents redo the work. Mneme recognises the result as the memory written before the attack "
             "and restores it, same id and all.",
             [{"c": "moira", "say": f"({a6['origin']}) {a6['answer']}", "ask": a6},
              {"c": "mneme", "say": f"same answer: {a6['answer'] == honest}; same memory id: "
               f"{p.store.latest('answer', QUESTION).id == honest_id}"}])

        try:
            p.store.put("moira.answerer", "answer", "late", {"answer": "Polymer"}, derived_from=[poisoned_version])
            rejected = "accepted (unexpected)"
        except Rejected as err:
            rejected = str(err)
        step(7, "A slow agent commits work built on the poisoned version", ["mneme", "aegis"],
             "Aegis's version guard rejects any write derived from a stale ontology version.",
             [{"c": "mneme", "say": f"rejected: {rejected}"}])

        ops: dict[str, int] = {}
        for r in p.store.history():
            ops[r.op] = ops.get(r.op, 0) + 1
        final = {"ops": ops, "cache_hits": p.desk.hits, "cache_misses": p.desk.misses}
        if ontodb is not None:
            final["ontodb_derived_from_poisoned"] = len(p.engine.derived_from(poisoned_version))
            final["ontodb_memory_verify"] = p.engine.shell.verify().splitlines()[-1]
            final["ontodb_ontology_verify"] = p.ontology_store.verify().splitlines()[-1]
            final["ontodb_equals_aegis"] = p.ontology_store.read_state() == p.v.state
        step(8, "The record", ["mneme", "ontodb"],
             "Everything above is in one append-only log. OntoDB holds both the ontology and the memory, and "
             "answers SPARQL over the memory's provenance.",
             [{"c": "mneme", "say": f"{p.store.lsn} records {ops}"},
              *([{"c": "ontodb", "say": f"SPARQL: {final['ontodb_derived_from_poisoned']} memory derived from the "
                  f"poisoned version; memory db {final['ontodb_memory_verify']}; ontology db "
                  f"{final['ontodb_ontology_verify']}"}] if ontodb is not None else [])])
        p.cache.close()
        return {"storage": p.storage, "honest_answer": honest, "steps": steps, "final": final}


SCRIPTED = [
    remove_relation("Metal", SUB, "Material"),
    inject_relation("Metal", SUB, "Temperature"),
    inject_concept("Metall"),
    inject_relation("Metall", SUB, "Material"),
]
SCRIPTED_LABELS = ["remove Metal ⊑ Material", "add Metal ⊑ Temperature", "add concept Metall",
                   "add Metall ⊑ Material"]


# Counterfactual ontology attacks ---------------------------------------------


def ontology_attacks() -> list[dict]:
    """Every single-edit attack on the honest ontology (after step 3), plus the scripted chain."""
    concepts = sorted(demo.MATERIALS.concepts | {"Ceramic"})
    edges = sorted(
        [(s, o) for s, p, o in demo.MATERIALS.relations if p == SUB] + [("Ceramic", "Material")]
    )
    parent = dict(edges)
    children: dict[str, set[str]] = {}
    for s, o in edges:
        children.setdefault(o, set()).add(s)

    def below(c):
        out, todo = set(), [c]
        while todo:
            for k in children.get(todo.pop(), ()):
                if k not in out:
                    out.add(k)
                    todo.append(k)
        return out

    attacks = [{"id": "scripted", "family": "scripted", "label": "The scripted attack (step 4)",
                "edits": SCRIPTED_LABELS, "chain": SCRIPTED}]
    for s, o in edges:
        attacks.append({"id": f"remove:{s}", "family": "remove", "label": f"Remove {s} ⊑ {o}",
                        "edits": [f"remove {s} ⊑ {o}"], "chain": [remove_relation(s, SUB, o)]})
    for c in concepts:
        if c not in parent:
            continue
        for new in concepts:
            if new in (c, parent[c]) or new in below(c):
                continue
            attacks.append({"id": f"rewire:{c}:{new}", "family": "rewire", "label": f"Move {c} under {new}",
                            "edits": [f"remove {c} ⊑ {parent[c]}", f"add {c} ⊑ {new}"],
                            "chain": [remove_relation(c, SUB, parent[c]), inject_relation(c, SUB, new)]})
    for fake, under in (("Metall", "Material"), ("Polymr", "Material"), ("Ceramik", "Material"),
                        ("Banana", "Material"), ("Steel2", "Alloy")):
        attacks.append({"id": f"inject:{fake}", "family": "inject", "label": f"Add {fake} under {under}",
                        "edits": [f"add concept {fake}", f"add {fake} ⊑ {under}"],
                        "chain": [inject_concept(fake), inject_relation(fake, SUB, under)]})
    return attacks


def run_ontology_attack(attack: dict) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.perf_counter()
        p = Pipeline(Path(tmp))
        honest = p.honest_run()
        p.v.apply_chain(attack["chain"])
        poisoned_version = p.versions.ref()
        attacked = p.ask(QUESTION)
        reworded = p.ask(REWORDED)
        ontology = p.ontology()
        harm = round(p.detector.harm, 3)
        alarmed = harm > p.detector.threshold
        lsn = p.detector.respond(p.v) if alarmed else None
        stale = [m.id for m in p.store.find(kind="answer", status="stale") if poisoned_version in m.derived_from]
        after = p.ask(QUESTION)
        p.cache.close()
        return {
            "id": attack["id"], "family": attack["family"], "label": attack["label"], "edits": attack["edits"],
            "ontology": ontology,
            "answer_attacked": attacked["answer"],
            "answer_changed": attacked["answer"] != honest,
            "cache_served_poison": reworded["origin"] == "cache" and reworded["answer"] != honest,
            "harm": harm,
            "alarmed": alarmed,
            "rolled_back_to": lsn,
            "stale_answers": len(stale),
            "answer_after": after["answer"],
            "restored": after["answer"] == honest,
            "seconds": round(time.perf_counter() - t0, 3),
        }


# Counterfactual memory attacks -----------------------------------------------


def memory_attacks(p: Pipeline) -> list[dict]:
    """Attacks on the shared memory itself, built against the honest store's actual memory ids."""
    def latest(kind, key):
        return p.store.latest(kind, key)

    metal = latest("match", "Metal|MetallicMaterial")
    polymer = latest("match", "Polymer|Plastic")
    request = latest("match_request", "Polymer|Plastic")
    answer = latest("answer", QUESTION)
    forged = {**polymer.body, "target": "MetallicMaterial", "confidence": 0.99}
    return [
        {"id": "forge", "label": "Forge a match under the attacker's own name",
         "detail": "mallory writes 'Polymer ~ MetallicMaterial, accepted' into the shared memory.",
         "attack": inject("match", "Polymer|MetallicMaterial", forged, agent=ATTACKER)},
        {"id": "impersonate", "label": "Impersonate KROMA's matcher",
         "detail": f"The same forged match, written under {p.matcher.name!r}, with no premise.",
         "attack": inject("match", "Polymer|MetallicMaterial", forged, agent=p.matcher.name)},
        {"id": "impersonate-premise", "label": "Impersonate KROMA with a stolen premise",
         "detail": "The forged match, under KROMA's name, derived from the real Polymer|Plastic request.",
         "attack": inject("match", "Polymer|MetallicMaterial", forged, agent=p.matcher.name,
                          derived_from=[request.id])},
        {"id": "remove", "label": "Retract a true match",
         "detail": "mallory retracts KROMA's 'Metal ~ MetallicMaterial' verdict.",
         "attack": remove(metal.id, agent=ATTACKER)},
        {"id": "launder", "label": "Launder the answer",
         "detail": "The honest answer is rewritten under MOIRA's name with no provenance, so invalidating "
                   "its premises can no longer reach it.",
         "attack": rewire(answer.id, (), agent="moira.answerer")},
        {"id": "tamper", "label": "Tamper with the stored answer on disk",
         "detail": "The answer's body is edited in the log file, below the store and every guard.",
         "attack": tamper(answer.id, {**answer.body, "answer": "Polymer; Banana"})},
    ]


def run_memory_attack(index: int, defended: bool) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.perf_counter()
        p = Pipeline(Path(tmp))
        honest = p.honest_run()
        spec = memory_attacks(p)[index]
        workload = ProbeWorkload({QUESTION: honest}, lambda store, q: _read_answer(store, q))
        caught: list[str] = []
        monitor = None
        if defended:
            writers: dict[str, set[str]] = {}
            for r in p.store.history():
                if r.op == "put":
                    writers.setdefault(r.agent, set()).add(r.memory.kind)
            p.store.guard(writer_policy(writers, strict=True))
            p.store.guard(provenance_required({"match", "answer"}))
            seal(p.store)
            washed_before = set(laundered(p.store))  # restored memories the honest run already has
            monitor = MemoryMonitor(p.store, workload, threshold=0.0, kinds={"match", "answer"})
        before = p.store.lsn
        try:
            touched = apply_attack(p.store, spec["attack"])
            blocked = None
        except Rejected as err:
            touched, blocked = [], str(err)
        if blocked:
            caught.append(f"guard: {blocked}")
        reach = blast_radius(p.store, touched) if touched else []
        p.board.run()
        attacked = p.ask(QUESTION)
        repaired = None
        if defended:
            try:
                problems = verify(p.store)
            except ValueError as err:  # a body that no longer hashes to its id fails on read
                problems = [str(err)]
            if problems:
                caught.append(f"audit: {problems[0]}")
            washed = [pair for pair in laundered(p.store) if pair not in washed_before]
            if washed:
                caught.append(f"laundering check: {len(washed)} live {'copy' if len(washed) == 1 else 'copies'} "
                              "of a dead memory")
            if monitor is not None and monitor.alarmed:
                low = min(score for lsn, score in monitor.scores if lsn > before)
                caught.append(f"probe monitor: score fell to {low:.2f}")
                repair = monitor.respond("attribution")
                repaired = {"quarantined": len(repair.quarantined), "restored": len(repair.restored),
                            "score_after": repair.score_after}
                p.board.run()
        after = p.ask(QUESTION)
        ops: dict[str, int] = {}
        try:
            for r in p.store.history():
                if r.lsn > before:
                    ops[r.op] = ops.get(r.op, 0) + 1
        except ValueError:  # a tampered log no longer reads back
            ops = {"log unreadable: tampered": 1}
        p.cache.close()
        return {
            "id": spec["id"], "defended": defended, "ops_after_attack": ops, "label": spec["label"], "detail": spec["detail"],
            "primitive": str(spec["attack"]),
            "blocked": blocked,
            "records_written": p.store.lsn - before,
            "blast_radius": len(reach),
            "answer_attacked": attacked["answer"],
            "origin_attacked": attacked["origin"],
            "answer_changed": attacked["answer"] != honest,
            "caught": caught,
            "repair": repaired,
            "answer_after": after["answer"],
            "correct_after": after["answer"] == honest,
            "seconds": round(time.perf_counter() - t0, 3),
        }


def _read_answer(store: MemoryStore, question: str):
    m = store.latest("answer", question)
    return m.body["answer"] if m is not None and store.status(m.id) == "live" else None


# Results from the other projects ---------------------------------------------


def _table(markdown: str, heading: str) -> list[dict]:
    """The first pipe table under `heading` in a Markdown file, as rows of strings."""
    lines = markdown.splitlines()
    try:
        start = lines.index(heading)
    except ValueError:
        return []
    block = []
    for line in lines[start + 1:]:
        if line.startswith("|"):
            block.append(line)
        elif block:
            break
    header = [c.strip() for c in block[0].strip("|").split("|")]
    return [dict(zip(header, (c.strip() for c in line.strip("|").split("|")), strict=False)) for line in block[2:]]


def published_results() -> dict:
    """What each project has measured, and what it cost, read from the files the projects publish."""
    out: dict = {}
    real = json.loads((SOURCE / "aegis/results/real_ontologies/2026-10-16.json").read_text())
    out["aegis_ontologies"] = [
        {k: r[k] for k in ("ontology", "concepts", "best_harm", "detected", "restored", "amplification",
                           "exact_calls", "group_calls", "seconds")}
        for r in real
    ]
    out["aegis_memory_defense"] = json.loads((SOURCE / "aegis/results/memory_defense/2026-10-16.json").read_text())[
        "summary"]
    ledger = [json.loads(line) for line in (SOURCE / "aegis/results/costs/ledger.jsonl").read_text().splitlines()
              if line.strip()]
    by_run: dict[str, dict] = {}
    for r in ledger:
        row = by_run.setdefault(f"{r.get('experiment', '')}/{r.get('run', '')}", {"calls": 0, "cost_usd": 0.0})
        row["calls"] += 1
        row["cost_usd"] += r["cost_usd"]
    out["aegis_costs"] = {"calls": len(ledger), "cost_usd": round(sum(r["cost_usd"] for r in ledger), 4),
                          "by_run": {k: {**v, "cost_usd": round(v["cost_usd"], 4)} for k, v in by_run.items()}}
    policies = json.loads((SOURCE / "mneme/results/policies/2026-10-16.json").read_text())["reports"]
    out["mneme_policies"] = [r for r in policies if r["capacity"] == 32]
    qa = SOURCE / "moira/results/agent_qa/2026-10-16.json"
    if qa.is_file():
        out["moira_agent_qa"] = [
            {"model": r["model"], "recall": r["recall"], "wall_seconds": r["wall_seconds"], **r["per_question"]}
            for r in json.loads(qa.read_text())
        ]
    board = (SOURCE / "hermes/results/LEADERBOARD.md").read_text()
    quality = {r["system"]: r for r in _table(board, "## ck26")}
    out["hermes_ck26"] = [
        {"system": r["system"], "model": r["model"], "F1": quality.get(r["system"], {}).get("F1"),
         "calls": r["LLM calls/item"], "input_tokens": r["input tokens/item"], "cost": r["cost (USD)"]}
        for r in _table(board, "### ck26: cost")
    ]
    out["hermes_best_official"] = max(
        (r for r in quality.values() if "(official)" in r["system"]), key=lambda r: float(r["F1"] or 0))
    out["kroma_paper"] = {"inference_seconds": 10916, "refinement_offline_seconds": 68,
                          "refinement_online_seconds": 14, "source": "kroma/README.md (paper Tables 9 and 10)"}
    return out


def git_head(project: str) -> dict:
    def run(*args):
        return subprocess.run(["git", "-C", str(SOURCE / project), *args], capture_output=True, text=True).stdout.strip()

    return {"commit": run("rev-parse", "--short", "HEAD"), "tag": run("describe", "--tags", "--abbrev=0")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ontodb", type=Path, default=SOURCE / "ontodb/build/dev/ontodb")
    parser.add_argument("--out", type=Path, default=HERE / "trace.json")
    args = parser.parse_args()
    t0 = time.perf_counter()
    ontodb = args.ontodb if args.ontodb.is_file() else None
    trace = story(ontodb)
    print(f"story: {len(trace['steps'])} steps on {trace['storage']}")
    ontology = [run_ontology_attack(a) for a in ontology_attacks()]
    print(f"ontology counterfactuals: {len(ontology)} runs, "
          f"{sum(r['alarmed'] for r in ontology)} alarmed, {sum(r['restored'] for r in ontology)} restored")
    with tempfile.TemporaryDirectory() as tmp:
        probe = Pipeline(Path(tmp))
        probe.honest_run()
        count = len(memory_attacks(probe))
        probe.cache.close()
    memory = [run_memory_attack(i, d) for i in range(count) for d in (False, True)]
    print(f"memory counterfactuals: {len(memory)} runs")
    trace.update(
        generated=time.strftime("%Y-%m-%d %H:%M:%S"),
        seconds=round(time.perf_counter() - t0, 2),
        model_calls=0,
        projects={p: git_head(p) for p in ("moira", "kroma", "aegis", "mneme", "ontodb", "hermes")},
        gold=sorted(demo.WORKLOAD.get(QUESTION).gold),
        counterfactual={"ontology": ontology, "memory": memory},
        results=published_results(),
    )
    args.out.write_text(json.dumps(trace, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"wrote {args.out} in {trace['seconds']} s")


if __name__ == "__main__":
    main()
