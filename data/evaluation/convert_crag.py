"""Convert CRAG benchmark subset into TestCase format."""

from pathlib import Path
from typing import List, Dict
import json
import tarfile
import tempfile
import shutil
import requests

from configs.config import config as default_config
from src.evaluation.evaluation_pipeline import TestCase


CRAG_URL = "https://github.com/Alibaba-NLP/CRAG/archive/refs/heads/main.tar.gz"


def download_crag(dest_dir: Path) -> Path:
    archive_path = dest_dir / "crag.tar.gz"
    if archive_path.exists():
        return archive_path

    dest_dir.mkdir(exist_ok=True, parents=True)
    print("Downloading CRAG benchmark...")
    with requests.get(CRAG_URL, stream=True) as resp:
        resp.raise_for_status()
        with open(archive_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
    return archive_path


def extract_crag_cases(archive_path: Path, limit: int = 1000) -> List[TestCase]:
    work_dir = Path(tempfile.mkdtemp())
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(work_dir)
        base_dir = next(work_dir.glob("CRAG-main*/datasets/crag_data"), None)
        if not base_dir:
            raise FileNotFoundError("CRAG dataset structure not found after extraction")

        qa_files = list(base_dir.glob("*.json"))
        cases: List[TestCase] = []
        for qa_file in qa_files:
            with open(qa_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for sample in data:
                query = sample.get("question", "")
                answer = sample.get("answer", "")
                passages = sample.get("passages", [])
                refs = []
                for passage in passages:
                    chunk = passage.get("text") or passage.get("passage")
                    if chunk:
                        refs.append(chunk)
                if not query or not answer:
                    continue
                cases.append(TestCase(
                    query=query,
                    expected_answer=answer,
                    reference_documents=refs,
                    difficulty_level=sample.get("difficulty", "medium"),
                    query_type=sample.get("type", "factual")
                ))
                if len(cases) >= limit:
                    return cases
        return cases
    finally:
        shutil.rmtree(work_dir)


def main():
    output_dir = default_config.EVALUATION_OUTPUT_DIR
    output_dir.mkdir(exist_ok=True, parents=True)
    dest_path = output_dir / "crag_testcases.json"
    cache_dir = output_dir / "cache"
    archive = download_crag(cache_dir)
    cases = extract_crag_cases(archive, limit=1000)
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
        "source": "CRAG",
        "count": len(cases)
    }
    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(cases)} CRAG cases to {dest_path}")


if __name__ == "__main__":
    main()
