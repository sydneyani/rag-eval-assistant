# Walkthrough

This is the version of CiteGround you can talk through in an interview. It names the modules and the numbers in `artifacts/eval/results.json`. Those numbers are from the offline stack: hashed TF embeddings, the extractive reader, and the deterministic judge. No API key was used.

## Thirty seconds

CiteGround answers questions about a filing-style corpus and cites the passage. Ingestion chunks Markdown without crossing a section. Retrieval is BM25 plus a dense index, fused by reciprocal rank fusion, then a small reranker. The reader may only use the retrieved passages. Evaluation checks whether the right section came back, and whether the answer is actually supported by it.

The corpus is synthetic on purpose. Helios Systems is not a company. The facts are stable, so a labeled set can be checked into git and CI can score it without EDGAR, without a model download, and without a key.

## The corpus

`data/corpus/` has two fake Forms 10-K and one shareholder letter. `data/corpus/README.md` says so, and the web UI says so. A hiring manager who thinks you scraped Apple's 10-K and called it a benchmark will trust you less, not more.

The loader in `citeground.ingest.loaders` reads a short `key: value` front matter and splits on `##` headings. A heading can end with `{customers}` to pin the section id. The id stored on every chunk is `doc_id/slug`, for example `helios-10k-fy2024/risk-power`.

Why label sections instead of chunks: chunk boundaries move when you change the chunk size. The gold set would have to move with them, and then you cannot tell whether Recall@5 changed because retrieval got better or because you relabeled. Section ids do not move. `hybrid-rerank-fixed-500` and `hybrid-rerank-recursive-900` share `data/eval/qa.jsonl`.

If they ask "why not the real EDGAR corpus": the loader will read any directory of the same shape. The limit is the index, not the file format. A real corpus is the right next experiment. It is a bad first experiment if you cannot reproduce the number.

## Ingestion

`citeground.ingest.pipeline.ingest_corpus` loads, chunks, embeds, and writes three files: `manifest.json`, `chunks.jsonl`, and `vectors.npy`. BM25 postings are not stored. They are rebuilt from chunk text when the retriever loads. On this corpus that is milliseconds. Say that out loud. Rebuilding is the right call until ingestion is incremental. Pretending you need OpenSearch on day one is how small projects grow a second system.

Three chunkers live in `citeground.ingest.chunking`:

- `recursive` packs paragraphs up to a character budget, then sentences, then fixed windows. Default, 900 characters, overlap 150.
- `fixed` is a character window. It will cut a sentence in half. The eval shows the scar: faithfulness 0.992 instead of 1.000, because the reader quoted a fragment that no longer matches a context sentence exactly.
- `sentence` is a sliding window of N sentences. It is there because some answers are one statement and the window itself is the citation.

None of them cross a section.

## Embeddings

`citeground.llm.factory.build_embedder` picks the provider. `auto` means OpenAI when `OPENAI_API_KEY` is set, and the hashing embedder otherwise. An explicit `openai` with no key is an error. Silent fallback hides the fact that CI and your laptop ran different systems.

`HashingEmbedder` is the hashing trick. Tokens and bigrams are hashed into a fixed vector with a random sign, weighted by log term frequency, and L2-normalized. Cosine similarity is a dot product. It does not know that "revenue" and "sales" are related. The eval is the evidence: dense-only Recall@5 is 0.926, BM25 is 0.981, and the two misses in the top 5 are `q-capex` and `q-customer-count`.

Say this if they ask why you did not ship a local transformer embedding model: a download makes CI slower and the number harder to reproduce. The interface is the same object either way. `OpenAIEmbedder` posts to `/embeddings` with httpx, normalizes the vectors, and keeps the token count for the cost estimate. Tests inject a mock transport. The key goes in a header and is not logged.

Dense search is exact. `dense_search_numpy` is a matrix multiply. `dense_search_faiss` builds an `IndexFlatIP` over the same normalized vectors, and the test checks the top hits match. Flat, not IVF. Approximate search is what you add when the matrix multiply shows up in the latency trace. It is not what you add to look serious on 30 documents.

## BM25

`citeground.retrieval.bm25` is Okapi BM25. `k1 = 1.5`, `b = 0.75`. IDF is `log(1 + (N - df + 0.5) / (df + 0.5))` so a common term cannot go negative. The tokenizer keeps `$4.82`, `61.4%`, `HIR-3`, and `10-K` as single tokens and then strips the currency mark. If you split `HIR-3` into `hir` and `3`, the product questions fall apart. There is a unit test for that.

BM25 on the default chunks has Recall@5 of 0.981 and MRR of 0.901. The Recall@5 hole is `q-customer-a`. That question has two gold sections, because both the customers section and the risk factor state that Customer A was 11 percent of fiscal 2024 revenue. BM25 returns the risk factor and not the customers section in the top 5. Recall@10 is 1.000. The section is in the index. The rank is the problem.

## Hybrid fusion

`citeground.retrieval.fusion.reciprocal_rank_fusion` implements

```
score(d) = sum over lists of 1 / (k + rank(d))
```

with `k = 60`, the constant from Cormack, Clarke, and Buettcher. BM25 is unbounded. Cosine on normalized vectors is in `[-1, 1]`. A weighted sum needs a normalization that changes with the query. Rank fusion does not.

You can say: "I refused to add 0.5 * BM25 + 0.5 * cosine, because those numbers do not live on one scale, and the 0.5 would be a hyperparameter I had not measured."

## The reranker

`citeground.retrieval.rerank` is a linear combination of query-term coverage, BM25 normalized by the max score in the pool, and cosine mapped into `[0, 1]`. Coverage gets the most weight. The hashing embedder is lexical, so giving it equal weight would count term overlap twice and call it semantic.

On this set the reranked hybrid stack is the one that closes the Customer A hole: Recall@5 goes from 0.981 to 1.000, and MRR from 0.901 to 0.926. That is a real gain. It is also a small gain, because BM25 was already the right tool for a corpus whose questions share the filing's words. If someone asks "was hybrid worth it", the honest answer is: on this set, for the last section and for rank, yes; for a paraphrase-heavy query set with a real embedding model, that is a different measurement and this table does not contain it.

It is not a cross-encoder. A cross-encoder reads the query and the passage together. This one never does. Replacing `rerank()` with a hosted reranker does not change `HybridRetriever`.

`hybrid-rerank-fixed-500` has the best MRR in the table, 0.944. Smaller windows put the fact in a chunk that overlaps the question more tightly, so it ranks higher. The cost shows up on `q-northline-revenue`. A 500-character window sliced `$71 million` so the reader quoted a fragment that ends in `$7`, and that question's faithfulness is 0.789. Averaged across the set, faithfulness is 0.992 instead of 1.000. That is the trade you are making when you pick a chunk size, and it is why the table exists.

## Answering, citations, abstention

`citeground.generation.answer.AnswerService` retrieves, builds the prompt in `generation.prompts`, calls the chat model, and parses `[1]` markers back onto chunks. The marker numbers are the order of the passages in the prompt. The UI renders those markers as buttons that scroll to the passage. It inserts them as text nodes, not as HTML, so a model cannot inject markup into the page.

`MockChatModel` is the offline reader. It scores context sentences by overlap with the question and quotes the best ones with their markers. If the best overlap is weak, or if two content words from the question never appear in the chosen passage, it says the passages do not state the answer. The dividend question in the labeled set is there to lock that behavior. Abstention accuracy on this run is 1.000, on one question. Do not inflate that into a calibration study.

The same system prompt is what `OpenAIChatModel` sees. Temperature is 0. If you turn the key on, you are changing the reader, not the retriever, unless you also switch the embedder.

Answer relevance on the default stack is 0.653. That is token-F1 against the gold answer. The reader often quotes a true sentence that is longer than the gold sentence, or the wrong true sentence. `q-litigation` scores 0.240 because the quote says no loss was accrued and then points at where the allegation is discussed, without stating the allegation. `q-fy2023-filing` scores 0.350 because it quotes a sentence that tells the reader where the number is, plus the hardware breakout, instead of the sentence that states total revenue. Both passages were retrieved. The reader picked poorly inside them. That is the failure mode you talk about: retrieval succeeded, attribution is narrow, the answer is still not the one you wanted.

## What the metrics are

All of this is in `citeground.eval.metrics` and `citeground.eval.judge`.

- Recall@k: fraction of gold sections that show up in the top k chunks. Macro-averaged over answerable questions.
- MRR: reciprocal rank of the first hit. A correct section at rank 2 scores 0.5.
- Faithfulness: for each answer sentence, the best token-F1 against any context sentence, then the mean. An abstention scores 1, because it made no claim.
- Answer relevance: token-F1 against the gold answer. A correct abstention scores 1. An abstention on an answerable question scores 0.
- Citation support: token-recall of each citing sentence against the passage it cites.

The deterministic judge is the offline mode. `LLMJudge` asks the chat model for a JSON object with the same two answer scores. It refuses to run on the mock reader. The committed table does not contain LLM-judge numbers. If you run `--judge llm` later, say that those numbers are a different column, and keep the offline column next to them.

Faithfulness of 1.000 will draw a question. Answer it before they ask. The offline reader copies sentences out of the passage. A metric that checks whether those sentences appear in the passage will return 1. Citation support is the same fact, measured at the marker. The fixed-window run is the existence proof that the metric can move: when the chunker slices the sentence, faithfulness is 0.992. A generative model judged by another model will not score 1.000 here, and the README says that.

## Observability and cost

`citeground.observability` does three small things.

- JSON logs, one object per line, with `trace_id`, path, status, and latency. The formatter does not log headers, so the bearer token is not one `extra=` away from the log file.
- A JSONL trace per answer: stage timings for retrieve and generate, token counts, provider, and the section ids that came back.
- `estimate_cost`. Offline providers report `cost_usd` of 0 and `cost_basis` of `offline-zero`. OpenAI multiplies the API's token counts by the prices in settings and labels the result `list-price-estimate`. Prices are configuration. They are not an invoice, and they will be wrong when the price changes if nobody updates `.env.example`.

`/v1/metrics` is the in-process aggregate: request count, error count, mean latency, tokens, cost, abstentions. It resets when the process restarts. That is fine for a single-process demo. It is not a dashboard.

The trace id is on the response as `x-trace-id` and in the JSONL line. That is the join key if you later ship the lines to a collector.

## The API and the UI

`citeground.api.app.create_app` is the test seam. The module-level `app` is what uvicorn loads. A missing index does not crash the process. Health reports `index_loaded: false`, and `/v1/answer` returns 503 with the ingest command in the detail. The UI shows that string.

Tests use `TestClient` and a temporary index. They check the fiscal 2024 revenue answer cites `helios-10k-fy2024/results`, that the dividend-style color question abstains, that cost is zero, and that a trace line was written.

Streamlit is a client. It does not retrieve. If you demo it, the API has to be up. The page served by FastAPI is the one that runs in the container.

## What you would change at a larger scale

Say this as a sequence, not a shopping list.

1. Swap the hashing embedder for a real one and rerun the same labeled set before you touch the reranker weights. The table is the baseline you are trying to beat. Dense-only at 0.926 is the number a real embedder has to clear, and then you care about paraphrase questions this set barely contains.
2. Put the vectors in FAISS or pgvector once exact search is in the latency budget. The protocol is already there. Do not start with an approximate index.
3. Replace the linear reranker with a cross-encoder when the error analysis says rank, not recall, is the problem. On this set, recall is mostly solved and rank is not: MRR 0.926 means the right section is often not first.
4. Replace the extractive reader with the OpenAI model on the same prompt, and add the LLM judge as a second column. Keep the deterministic column. Do not delete the only number CI can reproduce.
5. Grow the corpus. The loader does not care. The labeled set will, because questions that are easy on 30 sections get ambiguous when ten filings use the phrase "Customer A".
6. Persist the inverted index and make ingestion incremental when you are no longer rebuilding the world on startup.
7. Export the JSONL traces instead of inventing a new logging stack.

## Questions they tend to ask

**Why not LangChain or LlamaIndex?** The retrieval path is a few hundred lines and it is the thing the eval measures. A framework would hide the fusion and the chunk boundary, which are the decisions you want to be able to defend. The HTTP client for OpenAI is httpx for the same reason: the tests can see the request.

**Is Recall@5 of 1.000 good?** It is good on this set. The set was written in the vocabulary of the filings, and there are 27 answerable questions. It is not a leaderboard result. The interesting cell is the gap between BM25 and dense-only, and the fact that hybrid closed one two-section miss.

**Why is answer relevance only 0.653 if faithfulness is 1.000?** Because those metrics measure different things. Faithfulness asks "is this sentence in the passage?" Relevance asks "is this the gold answer?" A true sentence that is not the requested fact scores high on the first and low on the second. Litigation is the example.

**What happens with no key in CI?** `auto` selects the hashing embedder and the mock reader. GitHub Actions runs ruff and pytest on Python 3.11. The workflow does not install FAISS. The FAISS test skips when the package is absent. The NumPy path is the one CI guarantees.

**Where is the key stored?** It is not. `.env.example` has an empty `OPENAI_API_KEY`. `.env` is gitignored. The Docker image does not copy an env file.
