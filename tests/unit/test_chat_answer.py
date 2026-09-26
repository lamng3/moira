from agentoi.agents.core.tool_executor import preview
from agentoi.agents.observers.general import LogCapture
from agentoi.progress import WorkingStatus
from agentoi.workspace import _spoken_answer


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
