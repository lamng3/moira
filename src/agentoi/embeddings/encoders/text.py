from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from transformers import AutoModel, AutoTokenizer

if TYPE_CHECKING:
    from agentoi.algorithms.graph import EquivalentClass


class TextEmbedding:
    """Transformer-backed text encoder for concepts and free text."""

    DEFAULT_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)
        self.embed_dim = self.model.config.hidden_size

    def compute_embedding(self, node: EquivalentClass) -> torch.Tensor:
        """Embed an equivalence class and cache the result on the node."""
        parts: list[str] = []
        keys = ["labels", "alt_labels", "related_synonyms", "exact_synonyms"]
        for concept in node.equiv_concepts:
            if labels := [
                value for key in keys for value in concept.ground_set.get(key, [])
            ]:
                examples = ", ".join(labels)
            else:
                examples = concept.name.split("/")[-1]
            parts.append(examples)

        embedding = self.to_embedding(" | ".join(parts))
        setattr(node, "text_embedding", embedding)
        return embedding

    def to_embedding(self, text: str, max_length: int = 512) -> torch.Tensor:
        """Return the mean-pooled token embedding for ``text``."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding="longest",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = self.model(**inputs).last_hidden_state
        return output.mean(dim=1).squeeze(0).cpu()
