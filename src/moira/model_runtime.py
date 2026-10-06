from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import tiktoken
from langchain_together import ChatTogether

from moira.agents import Agent, ByteTokenizer, create_oot_enhanced_agent


DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
GROQ_MODEL_MAP = {
    DEFAULT_MODEL: "deepseek-r1-distill-llama-70b",
    "DeepSeek-R1-Distill-Llama-70B": "deepseek-r1-distill-llama-70b",
    "deepseek-r1-distill-llama-70b": "deepseek-r1-distill-llama-70b",
}


@dataclass(frozen=True, slots=True)
class ModelRuntime:
    llm: Any
    tokenizer: Any


def _optional_import(module: str, name: str) -> Any:
    try:
        imported = __import__(module, fromlist=[name])
    except ImportError as exc:
        package = module.replace("_", "-")
        hint = f"{name} requires the optional package {package}"
        if package == "langchain-ollama":
            hint += ". Install it with: uv sync --extra ollama"
        raise ImportError(hint) from exc
    return getattr(imported, name)


def create_model_runtime(
    *,
    model_name: str | None = None,
    temperature: float = 0.2,
    use_groq: bool = False,
    local_do_sample: bool = True,
) -> ModelRuntime:
    """Create an LLM and its matching prompt tokenizer."""
    model_name = model_name or os.getenv("LLM_MODEL") or DEFAULT_MODEL

    if os.path.isdir(model_name):
        chat_hugging_face = _optional_import(
            "langchain_huggingface", "ChatHuggingFace"
        )
        hugging_face_pipeline = _optional_import(
            "langchain_huggingface", "HuggingFacePipeline"
        )
        auto_tokenizer = _optional_import("transformers", "AutoTokenizer")
        auto_model = _optional_import("transformers", "AutoModelForCausalLM")
        transformers_pipeline = _optional_import("transformers", "pipeline")
        tokenizer = auto_tokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        model = auto_model.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        generation_pipeline = transformers_pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=local_do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_full_text=False,
        )
        llm = chat_hugging_face(
            llm=hugging_face_pipeline(
                pipeline=generation_pipeline,
                pipeline_kwargs={"temperature": temperature},
            )
        )
        return ModelRuntime(llm, tokenizer)

    if model_name.startswith("ollama:"):
        chat_ollama = _optional_import("langchain_ollama", "ChatOllama")
        ollama_model = model_name.partition(":")[2].strip() or "llama3.1"
        llm = chat_ollama(
            model=ollama_model,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=temperature,
            num_predict=512,
        )
        return ModelRuntime(llm, ByteTokenizer())

    if model_name.startswith("gpt"):
        chat_open_ai = _optional_import("langchain_openai", "ChatOpenAI")
        llm = chat_open_ai(
            model=model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=temperature,
        )
        return ModelRuntime(llm, tiktoken.get_encoding("cl100k_base"))

    if "gemini" in model_name:
        chat_google = _optional_import(
            "langchain_google_genai", "ChatGoogleGenerativeAI"
        )
        llm = chat_google(
            model=model_name,
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=temperature,
        )
        return ModelRuntime(llm, ByteTokenizer())

    if use_groq:
        chat_groq = _optional_import("langchain_groq", "ChatGroq")
        llm = chat_groq(
            model=GROQ_MODEL_MAP.get(model_name, model_name),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=temperature,
        )
    else:
        llm = ChatTogether(
            model=model_name,
            api_key=os.getenv("TOGETHER_API_KEY"),
            temperature=temperature,
        )

    try:
        auto_tokenizer = _optional_import("transformers", "AutoTokenizer")
        tokenizer = auto_tokenizer.from_pretrained(model_name)
    except (ImportError, OSError, ValueError):
        tokenizer = ByteTokenizer()
    return ModelRuntime(llm, tokenizer)


def create_application_agent(
    runtime: ModelRuntime,
    *,
    package_root: Path,
    save_local: bool,
    show_log: bool = False,
) -> Agent:
    from moira.agents.observers.general import LogCapture, print_observer

    capture = LogCapture()
    observers = [capture]
    if show_log:
        observers.append(print_observer)
    agent = create_oot_enhanced_agent(
        llm=runtime.llm,
        tools_path=package_root / "agents/tools/registry/ontology_tools.json",
        traces_dir="results/traces/oot",
        observers=observers,
        verbose=show_log,
        save_traces=True,
        save_local=save_local,
    )
    agent.run_log = capture
    return agent
