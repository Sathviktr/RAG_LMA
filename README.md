# RAG over the FMLA Employer's Guide: Baseline vs. Cross-Encoder Re-Ranking

Course assignment (Language Models and Agents). Two retrieval pipelines answer questions about one complex PDF, the U.S. Department of Labor's *Employer's Guide to the Family and Medical Leave Act* (76 pages):

- **Pipeline 1 (baseline):** a bi-encoder (`sentence-transformers/all-MiniLM-L6-v2`) and a FAISS inner-product index retrieve the top 10 chunks. The top 3 go to the LLM, giving **Answer 1**.
- **Pipeline 2 (re-ranking):** the same top 10 are re-scored by a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`). The new top 3 go to the LLM with the same prompt, giving **Answer 2**.

The LLM is Gemini (`gemini-3.5-flash-lite`, temperature 0, low thinking level), the same for both pipelines.

Everything is in one notebook: **`rag_baseline_vs_rerank.ipynb`**.

## Notebook steps

| Step | What it does |
|---|---|
| 1 | Setup: packages, API key, model choice |
| 2 | Load and clean the PDF with PyMuPDF (drop headers, footers, contents, index, diagram text; keep page/chapter/section) |
| 3 | Token-aware chunking: whole sentences, max 200 tokens, 40-token overlap, section heading in every chunk (207 chunks) |
| 4 | Bi-encoder embeddings + FAISS `IndexFlatIP` (cosine), `retrieve(q, k=10)` |
| 5 | Grounded prompt + cached Gemini call, Pipeline 1 |
| 6 | Cross-encoder re-ranking, Pipeline 2, rank-shift tables |
| 7 | Stress test on 21 pre-registered questions (keyword traps, exceptions, negations, controls) with evidence-phrase gold labels |
| 8 | Latency profiling per stage |
| 9 | Scaling: exhaustive cross-encoder, larger k, FAISS search time for large collections |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # installs CPU-only PyTorch (no GPU needed)
echo "GEMINI_API_KEY=your-key-here" > .env
jupyter notebook rag_baseline_vs_rerank.ipynb
```

Get a free key at https://aistudio.google.com/apikey. `.env` is git-ignored.

## Reproducibility and API quota

- All Gemini answers are cached in `results/llm_cache.json`. The smoke test and the LLM latency timings are saved in `results/smoke_test.json` and `results/latency_llm.json`.
- A full re-run therefore makes **no** Gemini requests and shows exactly the answers analysed. To regenerate answers, delete these files; calls are throttled to stay under the free tier's 15 requests per minute.
- Local timings (Steps 8 and 9) are re-measured on every run and vary by a few percent.
- The chunks (`results/chunks.csv`) are deterministic.

## Results files (`results/`)

| File | Content |
|---|---|
| `chunks.csv` | the 207 chunks with chunk id, chapter, section, pages, token count |
| `rank_shifts.csv` | per test question: gold chunk(s), bi-encoder rank, cross-encoder rank, outcome |
| `retrieval_metrics.csv` | Hit@3, Recall@10, MRR@10 for both pipelines |
| `answers.csv` | Answer 1 vs Answer 2 for all 21 questions, with context chunks and automatic signals (abstained, gold in context, gold cited) |
| `latency.csv`, `latency_llm.json` | latency per stage (mean, median, p95) |
| `ce_scaling.csv`, `k_sweep.csv`, `scaling.csv` | cross-encoder cost vs number of pairs, re-ranking top 10/20/30, time per question vs collection size |
| `figures/` | chunk lengths, embedding map, rank-shift slope chart, latency, cross-encoder scaling, scaling |

## Data

`data/GOVPUB-L-PURL-gpo84393.pdf`, *The Employer's Guide to the Family and Medical Leave Act*, U.S. Department of Labor, Wage and Hour Division. This is a U.S. Government work in the public domain.
