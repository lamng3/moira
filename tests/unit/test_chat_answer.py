from moira.agents.core.tool_executor import preview
from moira.agents.observers.general import LogCapture
from moira.progress import WorkingStatus
from moira.workspace import _is_tool_monologue, _presentable, _spoken_answer, _usable_prose


def test_spoken_answer_uses_the_model_summary() -> None:
    answer = _spoken_answer(
        {
            "query": {"term": "organ system"},
            "results": [{"label": "Organ System"}],
            "llm_summary": "The ontology includes the digestive system and other organ systems.",
        }
    )

    assert answer == "The ontology includes the digestive system and other organ systems."
    assert "NCIT" not in answer


def test_prompt_headings_are_left_out_of_the_answer() -> None:
    stored = (
        "The organ systems in the mouse include:\n\n"
        "* Visceral organ system\n"
        "* Digestive system (which is a part of the visceral organ system)\n\n"
        "These concepts answer the question by referencing specific terms from the "
        "RELEVANT CONCEPT CLUSTERS and KEY RELATIONS."
    )
    narration = (
        "Based on the provided concept clusters and relations, I can identify the following organ systems in a mouse:\n"
        "1. Head organ (from cluster 4)\n"
        "2. Thorax organ (from cluster 10)\n"
        "These organ systems are relevant to mouse anatomy (cluster 7)."
    )

    shown = _presentable(stored)

    assert "Visceral organ system" in shown
    assert "RELEVANT CONCEPT CLUSTERS" not in shown
    assert "KEY RELATIONS" not in shown
    assert _presentable(narration) == ""
    assert _usable_prose(narration) == ""


def test_a_concept_list_footer_is_left_out() -> None:
    stored = (
        "The organ systems in the mouse include:\n\n"
        "* Visceral organ system\n"
        "* Digestive system (which is a part of the visceral organ system)\n\n"
        'The relevant concepts used to answer this question include "organ system", '
        '"visceral organ system", and "digestive system".'
    )

    shown = _presentable(stored)

    assert "Visceral organ system" in shown
    assert "Digestive system" in shown
    assert "relevant concepts" not in shown.lower()


def test_a_tool_plan_is_not_usable_prose() -> None:
    plan = 'I will use the tools provided. {"tool_name": "search_term_context"}'

    assert _is_tool_monologue(plan)
    assert _usable_prose(plan) == ""
    assert _usable_prose("The visceral organ system sits in the body cavity.")


def test_log_capture_keeps_the_run_for_later() -> None:
    capture = LogCapture()
    capture({"event": "trace.start", "run_id": "abc"})
    capture({"event": "tool.exec", "tool_name": "ontology_term_info"})

    assert "Run abc started" in capture.text()
    assert "ontology_term_info" in capture.text()
    capture.clear()
    assert capture.text() == ""


def test_working_status_stops_without_printing_when_idle() -> None:
    status = WorkingStatus(enabled=False)
    status.start()
    status.stop()


def test_log_keeps_the_full_tool_result() -> None:
    payload = {"description": "A specimen that is derived from a longer ontology record." * 8}
    shown = preview(payload, limit=None)
    capture = LogCapture()
    capture({"event": "tool.result", "preview": shown})
    capture({"event": "trace.end", "result_preview": shown})

    text = capture.text()
    assert "…" not in text
    assert shown in text
    assert len(shown) > 240


def test_working_status_keeps_the_phrase_close() -> None:
    line = WorkingStatus._render("beep boop")

    assert line == "\rWorking · beep boop\033[K"
    assert "  " not in line
