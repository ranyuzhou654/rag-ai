import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '/Users/Zhuanz/Documents/GitHub/rag-ai/src')))
import asyncio
import numpy as np
from src.processing.multi_representation_indexer import MultiRepresentationIndexer

config = {
    "embedding_model": "BAAI/bge-m3",  # or any local model you have cached
    "llm_model": None,                 # skip LLM to avoid API usage
    "device": "cpu",
    "enable_multi_representation": False   # focus on progress for embeddings
}

indexer = MultiRepresentationIndexer(config)

chunks = [
    {
        "content": f"Section {i} explores async progress tracking.",
        "chunk_id": f"sample_{i}",
        "source_id": "doc_1",
        "metadata": {"section": "test"},
        "embedding": np.zeros(1024, dtype=np.float32)
    }
    for i in range(40)
]

async def run():
    await indexer.create_multi_representations(chunks, show_progress=True)

asyncio.run(run())
