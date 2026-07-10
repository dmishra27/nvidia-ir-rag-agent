"""Run the 4 benchmark questions through text_to_sql_agent.run() against the live DB."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents.text_to_sql_agent import run

QUESTIONS = [
    ("Q1", "How many documents are indexed per GPU family?"),
    ("Q2", "Which chunks scored below the quality threshold of 0.60?"),
    ("Q3", "What is the average chunk quality score per document?"),
    ("Q4", "How many total chunks were ingested across all documents?"),
]

SEP = "=" * 72

def main() -> None:
    for label, question in QUESTIONS:
        print(f"\n{SEP}")
        print(f"{label}: {question}")
        print(SEP)

        result = run(question, query_id=label.lower())

        print(f"Tool selected : {result.tool_name}")
        print(f"Tool params   : {json.dumps(result.tool_params)}")
        print(f"Error         : {result.error}")
        print(f"\n--- DB result ({len(result.sql_result or [])} rows) ---")
        for row in (result.sql_result or []):
            print(f"  {row}")
        print(f"\n--- Agent answer ---")
        print(result.answer)

    print(f"\n{SEP}")
    print("All 4 queries complete.")


if __name__ == "__main__":
    main()
