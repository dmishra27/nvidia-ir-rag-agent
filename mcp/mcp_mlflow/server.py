"""MCP server exposing MLflow experiment and benchmark run data.

Talks to a local MLflow tracking server via MlflowClient — no model loading,
no artifact downloads. Mirrors mcp/mcp_airflow/server.py's pattern of a thin
client over an already-running service, kept memory-safe per AGENTS.md.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import structlog
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mlflow.tracking import MlflowClient

load_dotenv()

# MCP stdio transport reserves stdout for JSON-RPC framing; structlog's
# default PrintLoggerFactory writes to stdout, which corrupts every tool
# call's response. Logs must go to stderr instead.
structlog.configure(logger_factory=structlog.PrintLoggerFactory(file=sys.stderr))
log = structlog.get_logger()

mcp = FastMCP("nvidia-ir-mlflow")

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")

_client: MlflowClient | None = None


def _get_client() -> MlflowClient:
    global _client
    if _client is None:
        _client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    return _client


@mcp.tool()
def list_experiments() -> dict[str, Any]:
    """Return all MLflow experiments (id, name, lifecycle_stage, artifact_location)."""
    log.info("list_experiments", stage="mcp_mlflow", query_id="tool_call")
    client = _get_client()
    experiments = client.search_experiments()
    return {
        "experiments": [
            {
                "experiment_id": e.experiment_id,
                "name": e.name,
                "lifecycle_stage": e.lifecycle_stage,
                "artifact_location": e.artifact_location,
            }
            for e in experiments
        ]
    }


@mcp.tool()
def get_benchmark_experiment(experiment_name: str = "reranker_benchmark", limit: int = 20) -> dict[str, Any]:
    """Return the most recent runs and their metrics/params for a named MLflow experiment."""
    log.info(
        "get_benchmark_experiment", stage="mcp_mlflow", query_id="tool_call",
        experiment_name=experiment_name, limit=limit,
    )
    client = _get_client()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return {"experiment_name": experiment_name, "error": "experiment not found", "runs": []}

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attribute.start_time DESC"],
        max_results=limit,
    )
    return {
        "experiment_name": experiment_name,
        "experiment_id": experiment.experiment_id,
        "runs": [
            {
                "run_id": r.info.run_id,
                "run_name": r.info.run_name,
                "status": r.info.status,
                "start_time": r.info.start_time,
                "metrics": dict(r.data.metrics),
                "params": dict(r.data.params),
            }
            for r in runs
        ],
    }


if __name__ == "__main__":
    mcp.run()
