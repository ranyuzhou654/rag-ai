"""Convert DuReader-robust dataset into TestCase format."""

from pathlib import Path
from typing import List, Dict
import json

from datasets import load_dataset

from configs.config import config as default_config
from src.evaluation.evaluation_pipeline import TestCase


def extract_reference_docs(sample: Dict) -> List[str]:
    passages = sample.get("documents", []) or []
    refs: List[str] = []
    for doc in passages:
        paragraphs = doc.get("paragraphs") or []
        for paragraph in paragraphs:
            if paragraph:
                refs.append(paragraph)
        if refs:
            break
    if not refs and passages:
        for doc in passages:
            title = doc.get("title")
            if title:
                refs.append(title)
    return refs[:5]


def convert_dureader(split: str = "validation", limit: int = 1000) -> List[TestCase]:
    dataset = load_dataset("baidu_dureader", "robust", split=split)
    cases: List[TestCase] = []
    for sample in dataset.select(range(min(limit, len(dataset)))):
        question = sample.get("question", "").strip()
        answers = sample.get("answers", {}) or {}
        expected_answer = ""
        if isinstance(answers, dict):
            best = answers.get("span_answers") or answers.get("text") or []
            if best:
                expected_answer = best[0]
        refs = extract_reference_docs(sample)
        if not question or not refs:
            continue
        cases.append(TestCase(
            query=question,
            expected_answer=expected_answer,
            reference_documents=refs,
            difficulty_level=sample.get("question_type", "medium"),
            query_type="factual"
        ))
    return cases


def main():
    output_dir = default_config.EVALUATION_OUTPUT_DIR
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / "dureader_testcases.json"
    cases = convert_dureader(split="validation", limit=1000)
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
        "source": "DuReader-robust",
        "split": "validation",
        "count": len(cases)
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(cases)} DuReader cases to {output_path}")


if __name__ == "__main__":
    main()
