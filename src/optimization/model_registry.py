# src/optimization/model_registry.py
"""Centralised model lifecycle management.

This module exposes a :class:`ModelRegistry` utility that lazily loads
embedding models and large language models once and reuses them across the
project.  Many parts of the system previously instantiated their own
``SentenceTransformer`` or ``AutoModelForCausalLM`` which resulted in
duplicated GPU memory usage and expensive initialisation costs.  The registry
keeps a shared cache of model instances guarded by thread locks so that
components can obtain handles without worrying about race conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Dict, Tuple, Optional

from loguru import logger
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


@dataclass
class LLMResource:
    """Container for a loaded tokenizer/model pair."""

    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    device: str


class ModelRegistry:
    """Registry that owns heavyweight model instances.

    The registry keeps two caches: one for embedding models (Sentence
    Transformers) and one for causal language models.  Keys are derived from the
    requested model name and the resolved execution device so the same resource
    can be safely reused by multiple callers.
    """

    _embedding_models: Dict[Tuple[str, str], SentenceTransformer] = {}
    _llm_models: Dict[Tuple[str, str], LLMResource] = {}

    _embedding_lock: Lock = Lock()
    _llm_lock: Lock = Lock()

    @staticmethod
    def _resolve_device(device: Optional[str]) -> str:
        if device in (None, "auto"):
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    @classmethod
    def get_sentence_transformer(
        cls, model_name: str, device: Optional[str] = "auto"
    ) -> SentenceTransformer:
        """Return a shared :class:`SentenceTransformer` instance."""

        resolved_device = cls._resolve_device(device)
        cache_key = (model_name, resolved_device)

        if cache_key not in cls._embedding_models:
            with cls._embedding_lock:
                if cache_key not in cls._embedding_models:
                    logger.info(
                        f"Loading embedding model '{model_name}' on {resolved_device}"
                    )
                    cls._embedding_models[cache_key] = SentenceTransformer(
                        model_name, device=resolved_device
                    )
                else:
                    logger.debug(
                        f"Reusing cached embedding model '{model_name}' on"
                        f" {resolved_device}"
                    )
        return cls._embedding_models[cache_key]

    @classmethod
    def get_llm(
        cls,
        model_name: str,
        device: Optional[str] = "auto",
        token: Optional[str] = None,
    ) -> LLMResource:
        """Return a shared tokenizer/model bundle for a causal LLM."""

        resolved_device = cls._resolve_device(device)
        cache_key = (model_name, token or "")

        if cache_key not in cls._llm_models:
            with cls._llm_lock:
                if cache_key not in cls._llm_models:
                    logger.info(
                        f"Loading LLM '{model_name}' with shared registry access"
                    )
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name, trust_remote_code=True, token=token
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype="auto",
                        device_map="auto",
                        trust_remote_code=True,
                        token=token,
                    )
                    cls._llm_models[cache_key] = LLMResource(
                        tokenizer=tokenizer,
                        model=model,
                        device=resolved_device,
                    )
                else:
                    logger.debug(
                        f"Reusing cached LLM '{model_name}' with shared registry"
                    )

        return cls._llm_models[cache_key]


__all__ = ["ModelRegistry", "LLMResource"]

