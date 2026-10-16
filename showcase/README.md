# Shared Memory Lab

An interactive site showing KROMA, MOIRA, Aegis, Mneme and OntoDB working together through one shared memory. Open `index.html` in a browser; it needs no server.

- **Playground.** An ontology viewer you can edit. Load a sample (Materials, EBSD, MatProc, XRD, Mouse anatomy) or upload `.owl`/`.rdf`, `.ttl`/`.nt`, `.obo`. Pan and zoom the graph; drag a concept onto another to move it, shift-drag to add a subclass link, click a link to remove it, add or delete concepts. Ask MOIRA about a concept, then edit as a curator (edits become the truth) or an attacker: Aegis measures the harm and rolls back, or, as a red team, finds the worst single edit. KROMA flags look-alike names, Mneme shows the shared memory, OntoDB the versions. Hover anything for details. It runs a compact JavaScript version of the projects' rules, needs no server, and keeps uploads in the browser.
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
