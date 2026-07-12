"""MCP server exposing Airflow DAG status and task logs.

Talks to the Airflow REST API (v2, Airflow 3) over basic auth. Credentials
and base URL are read from env vars so the same server code works against
the local docker-compose standalone instance or a remote deployment.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import requests
import structlog
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

# MCP stdio transport reserves stdout for JSON-RPC framing; structlog's
# default PrintLoggerFactory writes to stdout, which corrupts every tool
# call's response. Logs must go to stderr instead.
structlog.configure(logger_factory=structlog.PrintLoggerFactory(file=sys.stderr))
log = structlog.get_logger()

mcp = FastMCP("nvidia-ir-airflow")

AIRFLOW_API_URL = os.environ.get("AIRFLOW_API_URL", "http://localhost:8080")
AIRFLOW_API_USERNAME = os.environ.get("AIRFLOW_API_USERNAME", "airflow")
AIRFLOW_API_PASSWORD = os.environ.get("AIRFLOW_API_PASSWORD", "airflow")


def _get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.get(
        f"{AIRFLOW_API_URL}{path}",
        auth=(AIRFLOW_API_USERNAME, AIRFLOW_API_PASSWORD),
        params=params,
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


@mcp.tool()
def dag_status(dag_id: str, limit: int = 5) -> dict[str, Any]:
    """Return the most recent DAG runs for dag_id, most recent first."""
    log.info("dag_status", stage="mcp_airflow", query_id="tool_call", dag_id=dag_id)
    data = _get(
        f"/api/v2/dags/{dag_id}/dagRuns",
        params={"limit": limit, "order_by": "-logical_date"},
    )
    runs = [
        {
            "dag_run_id": r.get("dag_run_id"),
            "state": r.get("state"),
            "logical_date": r.get("logical_date"),
            "start_date": r.get("start_date"),
            "end_date": r.get("end_date"),
        }
        for r in data.get("dag_runs", [])
    ]
    return {"dag_id": dag_id, "runs": runs}


@mcp.tool()
def task_logs(dag_id: str, dag_run_id: str, task_id: str, try_number: int = 1) -> dict[str, Any]:
    """Return the log text for one task instance's try_number within a DAG run."""
    log.info(
        "task_logs", stage="mcp_airflow", query_id="tool_call",
        dag_id=dag_id, dag_run_id=dag_run_id, task_id=task_id,
    )
    response = requests.get(
        f"{AIRFLOW_API_URL}/api/v2/dags/{dag_id}/dagRuns/{dag_run_id}"
        f"/taskInstances/{task_id}/logs/{try_number}",
        auth=(AIRFLOW_API_USERNAME, AIRFLOW_API_PASSWORD),
        params={"accept": "application/json"},
        timeout=15,
    )
    response.raise_for_status()
    return {
        "dag_id": dag_id,
        "dag_run_id": dag_run_id,
        "task_id": task_id,
        "try_number": try_number,
        "content": response.json() if "application/json" in response.headers.get("content-type", "") else response.text,
    }


if __name__ == "__main__":
    mcp.run()
