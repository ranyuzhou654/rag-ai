"""Convert a subset of HotpotQA into TestCase format used by evaluation pipeline."""

from pathlib import Path
from typing import List, Dict
import json
from datasets import load_dataset

from configs.config import config as default_config
from src.evaluation.evaluation_pipeline import TestCase


def extract_reference_docs(sample: Dict) -> List[str]:
    """Extract supporting sentences with robust handling of context format."""
    supporting_facts = sample.get("supporting_facts", []) or []
    context_pairs = sample.get("context", []) or []

    context_dict: Dict[str, List[str]] = {}
    for entry in context_pairs:
        title = None
        sentences = None
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            title, sentences = entry[0], entry[1]
        elif isinstance(entry, dict):
            title = entry.get("title") or entry.get("heading")
            sentences = entry.get("sentences") or entry.get("context")
        if title is None or sentences is None:
            continue
        if isinstance(sentences, str):
            sentences = [sentences]
        context_dict[title] = list(sentences)

    refs: List[str] = []
    for fact in supporting_facts:
        if isinstance(fact, (list, tuple)) and len(fact) >= 2:
            title, sent_idx = fact[0], fact[1]
            sentences = context_dict.get(title)
            if sentences and 0 <= sent_idx < len(sentences):
                refs.append(sentences[sent_idx])

    if not refs:
        # fallback: take first sentence of each context block
        for sentences in context_dict.values():
            if sentences:
                refs.append(sentences[0])
                if len(refs) >= 5:
                    break

    seen = set()
    unique_refs = []
    for r in refs:
        if r not in seen:
            seen.add(r)
            unique_refs.append(r)
    return unique_refs


def convert_hotpotqa(split: str = "validation", limit: int = 1000) -> List[TestCase]:
    dataset = load_dataset("hotpot_qa", "distractor", split=split)
    cases: List[TestCase] = []
    for sample in dataset.select(range(min(limit, len(dataset)))):
        refs = extract_reference_docs(sample)
        difficulty = sample.get("level", "hard")
        cases.append(TestCase(
            query=sample["question"],
            expected_answer=sample.get("answer", ""),
            reference_documents=refs,
            difficulty_level=difficulty,
            query_type="analytical" if difficulty == "hard" else "factual"
        ))
    return cases


def main():
    output_dir = default_config.EVALUATION_OUTPUT_DIR
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / "hotpotqa_testcases.json"
    cases = convert_hotpotqa(split="validation", limit=1000)
    payload = {
        "test_cases": [
            {
                "query": case.query,
                "expected_answer": case.expected_answer,
                "reference_documents": case.reference_documents,
                "difficulty_level": case.difficulty_level,
                "query_type": case.query_type,
                "ground_truth_chunks": case.ground_truth_chunks
            }
            for case in cases
        ],
        "source": "HotpotQA",
        "split": "validation",
        "count": len(cases)
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(cases)} HotpotQA cases to {output_path}")


if __name__ == "__main__":
    main()
