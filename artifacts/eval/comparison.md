# Evaluation comparison

Judge: `offline-deterministic`.
Embeddings: `hashing` / `signed-hash-tf`.
Reader: `mock` / `mock-extractive`.
Questions: 28 (27 answerable, 1 meant to abstain).

Offline run. No API key. Faithfulness and answer relevance are deterministic lexical proxies, not RAGAS model scores.

| Experiment | Recall@5 | Recall@10 | MRR | Faithfulness | Answer relevance | Citation support | Abstention accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25-recursive-900 | 0.981 | 1.000 | 0.901 | 1.000 | 0.633 | 1.000 | 1.000 |
| dense-hash-recursive-900 | 0.926 | 1.000 | 0.819 | 1.000 | 0.613 | 1.000 | 1.000 |
| hybrid-rerank-recursive-900 | 1.000 | 1.000 | 0.926 | 1.000 | 0.653 | 1.000 | 1.000 |
| hybrid-rerank-fixed-500 | 1.000 | 1.000 | 0.944 | 0.992 | 0.649 | 1.000 | 1.000 |

Recall and MRR are averaged over answerable questions only. An empty gold set would make those metrics zero for a correct abstention.
Faithfulness, answer relevance, and citation support are averaged over every question.
Abstention accuracy is the fraction of questions marked `expect_abstention` where the reader abstained. It is not a retrieval metric.
