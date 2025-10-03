"""Convert a subset of HotpotQA into TestCase format used by evaluation pipeline."""

from pathlib import Path
from typing import List, Dict
import json
from datasets import load_dataset

from configs.config import config as default_config
from src.evaluation.evaluation_pipeline import TestCase


def extract_reference_docs(sample: Dict) -> List[str]:
    refs: List[str] = []
    supporting_facts = sample.get("supporting_facts", [])
    context_pairs = sample.get("context", [])
    context_dict = {title: sentences for title, sentences in context_pairs}
    for title, sentence_idx in supporting_facts:
        if title in context_dict and sentence_idx < len(context_dict[title]):
            refs.append(context_dict[title][sentence_idx])
    return refs


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
