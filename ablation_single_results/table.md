| Variant | Only change from previous row | Recall@5 | Kept | Rejected | Accuracy |
|---|---|---:|---:|---:|---:|
| B0 | Naive get_text + recursive splitting + whitespace BM25 top-5 | 39.1% | 100.0% | 0.0% | 46.7% |
| V1 | Add table-to-text conversion | 32.6% | 100.0% | 0.0% | 48.3% |
| V2 | Replace recursive splitting with section/table-aware splitting | 21.7% | 100.0% | 0.0% | 40.0% |
| V3 | Add contextual chunk headers, including research broker and report title | 26.1% | 100.0% | 0.0% | 41.7% |
| V4 | Add company/type/year metadata filtering | 84.8% | 97.8% | 71.4% | 88.3% |
| V5 | Replace whitespace tokenisation with normalised tokenisation | 80.4% | 97.8% | 71.4% | 86.7% |
| V6 | Add query rewriting: remove entity/year terms and expand synonyms | 100.0% | 100.0% | 71.4% | 98.3% |
| V7 | Add table score boost | 100.0% | 100.0% | 71.4% | 98.3% |
| V8 | Reserve an evidence slot for each comparison company | 100.0% | 100.0% | 71.4% | 100.0% |
| V9 | Add LLM reranking | 97.8% | 97.8% | 71.4% | 96.7% |
| V10 | Apply the reranker answerability decision | 100.0% | 100.0% | 100.0% | 98.3% |
