"""MOIRA's usage meter: every metered model call is counted and priced, local calls are timed."""

from types import SimpleNamespace

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from moira.costs import UsageMeter, attach, price


def result(input_tokens, output_tokens):
    message = SimpleNamespace(usage_metadata={"input_tokens": input_tokens, "output_tokens": output_tokens})
    return SimpleNamespace(generations=[[SimpleNamespace(message=message, generation_info=None)]], llm_output=None)


def test_the_meter_counts_and_prices_calls():
    meter = UsageMeter()
    meter.on_chat_model_start({}, [], run_id=1, invocation_params={"model": "claude-opus-5-5"})
    meter.on_llm_end(result(1000, 100), run_id=1)
    meter.on_chat_model_start({}, [], run_id=2, invocation_params={"model": "ollama:llama3.1"})
    meter.on_llm_end(result(500, 50), run_id=2)
    summary = meter.summary()
    assert summary["calls"] == 2 and summary["input_tokens"] == 1500 and summary["unpriced_calls"] == 1
    assert summary["cost_usd"] == round((1000 * 4 + 100 * 20) / 1e6, 6)
    assert price("gpt-4o-mini", 1_000_000, 0) == 0.15 and price("llama3.1", 10, 10) is None


def test_attach_meters_a_real_langchain_model():
    meter = UsageMeter()
    llm = attach(FakeListChatModel(responses=["Metal is a Material."]), meter)
    llm.invoke("What is a Metal?")
    llm.invoke("What is a Metal?")
    assert meter.summary()["calls"] == 2
    assert attach(llm, meter).callbacks.count(meter) == 1
