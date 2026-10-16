# Shared Memory Lab

An interactive site showing KROMA, MOIRA, Aegis, Mneme and OntoDB working together through one shared memory. Open `index.html` in a browser; it needs no server.

- **Playground.** Load a sample (Materials, EBSD, MatProc, XRD, Mouse anatomy) or upload your own ontology (`.owl`/`.rdf`, `.ttl`/`.nt`, `.obo`). Click a concept to ask MOIRA about it, then edit the ontology as a curator (edits become the truth) or as an attacker (drag a concept onto another, detach it, add a look-alike). Aegis measures the harm to your questions and rolls back, or searches for the worst single edit as a red team; KROMA flags look-alike names; Mneme shows every memory, tier, and log record; OntoDB keeps every version. It runs a compact JavaScript version of the projects' rules, so it needs no server, and an uploaded file stays in the browser.
- **Recorded run.** Steps through one recorded run of `examples/shared_memory_workflow.py`, stored on OntoDB: KROMA aligns, MOIRA answers, an honest edit, an attack, Aegis detects it and rolls back, and Mneme restores the honest memory. Each step shows which components act, the ontology at that version, what MOIRA answers and whether its cache served it, every memory by tier (click one to see its body and lineage), and the log records written.
- **Recorded attacks.** 45 counterfactual runs: every edge removal, every move of a concept to another parent, look-alike concepts, and the scripted attack. Each shows MOIRA's answer with and without Aegis, the harm, the rollback, and whether the honest answer came back.
  Below them, attacks on the memory: forgery, impersonation, a stolen premise, retraction, laundering and tampering against the shared memory, each with Aegis's memory defences off and on.
- **Results and cost.** What each project has measured and what it cost, read from the files the projects publish.

Every number comes from real runs. Nothing is simulated in the browser. The live run makes no model calls: KROMA's language-model verdicts are fixed, as in the workflow example, and MOIRA's answerer reads the hierarchy.

```bash
uv run python showcase/record.py   # runs everything, writes trace.json (about 6 s)
uv run python showcase/samples.py  # exports the sample ontologies through Aegis's loader
python showcase/build.py           # inlines both into index.html
```

Run both from the moira repository root, with `mneme`, `aegis`, `kroma`, `hermes` and `ontodb` cloned beside it. Without an OntoDB binary, the recorded run uses JSONL.
