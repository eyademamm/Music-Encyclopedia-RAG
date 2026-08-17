"""
Evaluates text search, vector search, and hybrid (RRF) search using
hit rate and MRR against the generated ground truth. Same metrics as
HW3.

Usage:
    python src/evaluate.py
"""

import json
from pathlib import Path

from search import (build_text_index, build_vector_index, hybrid_search,
                     load_docs, text_search, vector_search)

GT_PATH = Path(__file__).resolve().parent.parent / "data" / "ground_truth.json"


def hit_rate(relevance_list):
    return sum(any(r) for r in relevance_list) / len(relevance_list)


def mrr(relevance_list):
    total = 0.0
    for r in relevance_list:
        for rank, is_relevant in enumerate(r):
            if is_relevant:
                total += 1 / (rank + 1)
                break
    return total / len(relevance_list)


def evaluate(search_fn, ground_truth):
    relevance_list = []
    for gt in ground_truth:
        results = search_fn(gt["question"])
        relevance = [d["id"] == gt["doc_id"] for d in results]
        relevance_list.append(relevance)
    return {"hit_rate": hit_rate(relevance_list), "mrr": mrr(relevance_list)}


def main():
    docs = load_docs()
    ground_truth = json.loads(GT_PATH.read_text(encoding="utf-8"))

    index = build_text_index(docs)
    vindex, embedder = build_vector_index(docs)

    methods = {
        "text": lambda q: text_search(index, q, num_results=5),
        "vector": lambda q: vector_search(vindex, embedder, q, num_results=5),
        "hybrid": lambda q: hybrid_search(index, vindex, embedder, q, num_results=5),
    }

    print(f"Evaluating on {len(ground_truth)} questions...\n")
    results = {}
    for name, fn in methods.items():
        metrics = evaluate(fn, ground_truth)
        results[name] = metrics
        print(f"{name:8s} -> hit_rate={metrics['hit_rate']:.3f}  mrr={metrics['mrr']:.3f}")

    best = max(results, key=lambda k: results[k]["mrr"])
    print(f"\nBest method: {best}")

    out_path = Path(__file__).resolve().parent.parent / "data" / "retrieval_eval.json"
    out_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
