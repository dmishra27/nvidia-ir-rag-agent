import torch, transformers, sentence_transformers
import langchain, langsmith, langchain_anthropic
from langgraph.graph import StateGraph
import pandas, numpy, scipy, sklearn
import fastapi, uvicorn, pydantic, httpx
import psycopg2, alembic, sqlalchemy
import ragas, deepeval, mlflow, wandb
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from opentelemetry.sdk.trace import TracerProvider
import structlog, phoenix
import pytest
import pymupdf, pdfplumber, datasketch, spacy
from slack_bolt import App
import cohere, anthropic

print("=== PHASE C COMPLETE ===")
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("sentence-transformers:", sentence_transformers.__version__)
print("langchain:", langchain.__version__)
print("langsmith:", langsmith.__version__)
print("langchain-anthropic:", langchain_anthropic.__version__)
print("langgraph: StateGraph OK")
print("pandas:", pandas.__version__)
print("numpy:", numpy.__version__)
print("fastapi:", fastapi.__version__)
print("pydantic:", pydantic.__version__)
print("sqlalchemy:", sqlalchemy.__version__)
print("ragas:", ragas.__version__)
print("mlflow:", mlflow.__version__)
print("wandb:", wandb.__version__)
print("rank-bm25: BM25Okapi OK")
print("qdrant-client: QdrantClient OK")
print("opentelemetry: TracerProvider OK")
print("structlog:", structlog.__version__)
print("phoenix:", phoenix.__version__)
print("pytest:", pytest.__version__)
print("pymupdf:", pymupdf.__version__)
print("spacy:", spacy.__version__)
print("slack-bolt: App OK")
print("cohere:", cohere.__version__)
print("anthropic:", anthropic.__version__)
print("locust: verified separately - run 'locust --version'")
print("=== ALL IMPORTS PASSED ===")
