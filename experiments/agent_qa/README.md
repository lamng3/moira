# Agent QA: answer quality and cost by model

**Question.** How well does MOIRA's agent answer hierarchy questions with each
local model, and what does each answer cost in calls, tokens, and time?

**Protocol.** Mouse anatomy (OAEI). Eight concepts with three to eight direct
subclasses, chosen in a fixed order, become "What kinds of LABEL are there?".
MOIRA answers through its full pipeline (retrieval over the cached concept
embeddings, top 10, then the model, temperature 0). An answer's **recall** is
the fraction of the concept's direct subclasses whose label it names. Every
model call is metered by `moira.costs`.

**Run.** `uv run python experiments/agent_qa/run.py --models ollama:llama3.1 ollama:mistral ollama:phi3 --json results/agent_qa/2026-10-16.json`
