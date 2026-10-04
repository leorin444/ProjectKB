"""Retrieval and question-answering evaluation script.

Evaluates:
- Hit Rate @ 1, 3, 5 (is an expected chunk ID / expected keyword found in top-K?)
- Mean Reciprocal Rank (MRR)
- Distractor Rejection Rate (% of unanswerable questions correctly returning found: False)

Usage:
    python scripts/evaluate.py --host http://127.0.0.1:8000 --api-key <key>
"""

from __future__ import annotations

import argparse
from typing import Any

import httpx

EVAL_BENCHMARK: list[dict[str, Any]] = [
    {
        "question": "What is the primary storage driver for Docker containers?",
        "expected_keywords": ["overlay2", "storage driver"],
        "is_answerable": True,
    },
    {
        "question": "How does Docker provide container process isolation?",
        "expected_keywords": ["namespaces", "cgroups"],
        "is_answerable": True,
    },
    {
        "question": "What is the largest moon in the Solar System?",
        "expected_keywords": ["Ganymede", "Jupiter"],
        "is_answerable": True,
    },
    {
        "question": "What error code occurred when npm peer dependencies failed?",
        "expected_keywords": ["ERESOLVE", "peer dependency"],
        "is_answerable": True,
    },
    # Distractors / Unanswerable queries:
    {
        "question": "How do you bake a classic French sourdough baguette?",
        "expected_keywords": [],
        "is_answerable": False,
    },
    {
        "question": "What are the rules and player positions in Australian rules football?",
        "expected_keywords": [],
        "is_answerable": False,
    },
    {
        "question": "What is the capital city of ancient Atlantis according to Plato?",
        "expected_keywords": [],
        "is_answerable": False,
    },
]


def run_evaluation(base_url: str, api_key: str) -> dict[str, float]:
    headers = {"X-API-Key": api_key}
    client = httpx.Client(base_url=base_url, timeout=30.0)

    total_answerable = 0
    hit1 = 0
    hit3 = 0
    hit5 = 0
    reciprocal_ranks: list[float] = []

    total_distractors = 0
    distractors_rejected = 0

    print("Running evaluation suite against:", base_url)
    print("-" * 60)

    for item in EVAL_BENCHMARK:
        q = item["question"]
        is_ans = item["is_answerable"]

        if is_ans:
            total_answerable += 1
            # Check /search retrieval ranking
            try:
                res = client.post("/search", headers=headers, json={"query": q, "top_k": 5})
                res.raise_for_status()
                hits = res.json()
            except Exception as e:
                print(f"Error querying search for '{q}': {e}")
                hits = []

            # Check if expected keywords appear in hits
            rank_found = 0
            for rank, hit in enumerate(hits, start=1):
                text = hit.get("text", "").lower()
                if any(kw.lower() in text for kw in item["expected_keywords"]):
                    rank_found = rank
                    break

            if rank_found == 1:
                hit1 += 1
                hit3 += 1
                hit5 += 1
                reciprocal_ranks.append(1.0)
            elif 1 < rank_found <= 3:
                hit3 += 1
                hit5 += 1
                reciprocal_ranks.append(1.0 / rank_found)
            elif 3 < rank_found <= 5:
                hit5 += 1
                reciprocal_ranks.append(1.0 / rank_found)
            else:
                reciprocal_ranks.append(0.0)

            status_str = f"Rank {rank_found}" if rank_found else "Miss"
            print(f"[Answerable] '{q[:40]}...' -> {status_str}")

        else:
            total_distractors += 1
            # Check /query rejection
            try:
                res = client.post("/query", headers=headers, json={"question": q})
                res.raise_for_status()
                ans = res.json()
                found = ans.get("found", True)
            except Exception as e:
                print(f"Error querying Q&A for '{q}': {e}")
                found = True

            if not found:
                distractors_rejected += 1
                print(f"[Distractor] '{q[:40]}...' -> Correctly Rejected")
            else:
                print(f"[Distractor] '{q[:40]}...' -> False Positive!")

    print("-" * 60)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    h1_rate = (hit1 / total_answerable) if total_answerable else 0.0
    h3_rate = (hit3 / total_answerable) if total_answerable else 0.0
    h5_rate = (hit5 / total_answerable) if total_answerable else 0.0
    distractor_rejection_rate = (
        (distractors_rejected / total_distractors) if total_distractors else 0.0
    )

    metrics = {
        "Hit@1": round(h1_rate, 4),
        "Hit@3": round(h3_rate, 4),
        "Hit@5": round(h5_rate, 4),
        "MRR": round(mrr, 4),
        "Distractor_Rejection_Rate": round(distractor_rejection_rate, 4),
    }

    print("Results:")
    for k, v in metrics.items():
        print(f"  {k:26}: {v}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate KB retrieval and rejection accuracy")
    parser.add_argument("--host", default="http://127.0.0.1:8000", help="Base URL of API")
    parser.add_argument("--api-key", default="change-me", help="API key")
    args = parser.parse_args()

    run_evaluation(args.host, args.api_key)
