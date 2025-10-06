"""Quick smoke-test for external data sources.

Run this script after configuring network access and environment variables to
verify that OpenAlex and Semantic Scholar integrations are working as expected.
"""

import asyncio
from collections import Counter
from pathlib import Path
from typing import List

from src.data_ingestion.multi_source_collector import Document, MultiSourceCollector


async def main() -> None:
    data_dir = Path("storage/smoke_test")
    collector = MultiSourceCollector(data_dir=data_dir, metadata_only=True)

    docs: List[Document] = await collector.collect_metadata_only(days_back=1, max_papers=10)
    source_counter = Counter(doc.source for doc in docs)

    print("Collected documents by source:")
    for source, count in source_counter.items():
        print(f"  - {source}: {count}")

    enriched = [doc for doc in docs if doc.tldr or doc.concepts]
    if enriched:
        sample = enriched[0]
        print("\nSample enriched metadata:")
        print(f"ID: {sample.id}")
        print(f"Title: {sample.title}")
        if sample.tldr:
            print(f"TLDR: {sample.tldr}")
        if sample.concepts:
            print(f"Concepts: {', '.join(sample.concepts)}")
    else:
        print("\nNo Semantic Scholar enrichment detected. Check API access and rate limits.")


if __name__ == "__main__":
    asyncio.run(main())
