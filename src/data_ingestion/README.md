# 数据采集改进

- [`multi_source_collector.py`](./multi_source_collector.py) 在 `collect_all` 与 `fetch_arxiv_papers` 中复用单一 `aiohttp.ClientSession`，并在 `_process_single_pdf` 里共享连接池下载 PDF，提高高并发场景下的抓取效率。
