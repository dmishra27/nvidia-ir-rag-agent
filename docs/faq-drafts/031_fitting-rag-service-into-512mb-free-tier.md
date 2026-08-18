---
id: 7b6b6a9842
question: 'Project: how do I fit a RAG service with embeddings/rerankers onto a 512 MB free-tier host?'
sort_order: 31
---

Skip loading the heavy models entirely in whatever mode you deploy, rather than trying to shrink them to fit.

A vector encoder and a cross-encoder reranker (sentence-transformers/torch) can each pull well past a few hundred MB into memory before a single request is served — on a 512 MB plan, loading both at once OOMs on the very first request, and the host's own event log usually says so directly ("ran out of memory") rather than throwing a normal application error, so check there first instead of guessing at the cause. The fix isn't a smaller model; it's not loading the model at all in the deployed configuration, and falling back to a lighter-weight retrieval path (e.g. lexical/keyword search only) that needs no model load.

- Make the expensive components (dense encoder, reranker) optional behind a mode flag, not always-on.
- Have the fallback mode return degraded-but-correct results (e.g. lexical-only ranking) rather than failing outright.
- Make sure any code path that builds a fallback object can't silently re-trigger the expensive load anyway — a pattern like `heavy_thing or load_heavy_thing()` can't tell "explicitly disabled" from "not provided," and will reconnect regardless.
- Wire the mode flag into both your app code and your deploy config — a code fix that defaults to the heavy mode in production config is still going to OOM.
