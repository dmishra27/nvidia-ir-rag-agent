# nvidia-ir-rag-agent — SKILLS.md

## Skill: retrieve-hybrid
Build a hybrid retrieval function combining BM25 + dense + RRF.
Pattern:
  BM25Index(corpus).search(q, top_k=100)
  DenseIndex(qdrant_client).search(q, top_k=100)
  rrf_fusion(bm25_results, dense_results, k=60)
All functions return List[Candidate(chunk_id, text, score, rank)]

## Skill: build-langgraph-node
LangGraph nodes take AgentState and return AgentState.
Pattern:
  def node_name(state: AgentState) -> AgentState:
      logger.info('node_name', query_id=state.query_id, stage='node_name')
      return state.model_copy(update={'field': new_value})
Always use structlog. Always preserve all existing state fields.

## Skill: write-ragas-eval
RAGAS evaluation pattern:
  from ragas import evaluate
  from ragas.metrics import faithfulness, answer_relevancy, context_precision
  dataset = Dataset.from_dict({'question': [...], 'answer': [...],
    'contexts': [...], 'ground_truth': [...]})
  result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
  mlflow.log_metrics(result)

## Skill: build-mcp-tool
Custom MCP servers expose tools via FastMCP.
Pattern:
  from mcp.server.fastmcp import FastMCP
  mcp = FastMCP('server-name')
  @mcp.tool()
  def tool_name(param: str) -> dict: ...
Register in Claude Code MCP config JSON file.

## Skill: add-otel-span
Wrap any function with an OpenTelemetry span:
  from opentelemetry import trace
  tracer = trace.get_tracer('nvidia-ir-rag-agent')
  with tracer.start_as_current_span('span-name') as span:
      span.set_attribute('query_id', query_id)
      span.set_attribute('reranker_config', config)

## Skill: run-reranker-benchmark
Three-way benchmark pattern.
Always use same 50-query set (evaluation/benchmark_queries.jsonl)
and same RRF top-100 candidates as input to all three configs.
Log per-config per-query to MLflow experiment 'reranker_benchmark'.
Configs: config_A_ms_marco / config_B_bge_reranker / config_C_cohere_rerank
Metrics: ndcg_at_10, mrr, prec_at_k, faithfulness, citation_accuracy, latency_ms, cost_usd
