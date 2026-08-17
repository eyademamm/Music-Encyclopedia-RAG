"""
Generates ground-truth (question -> doc_id) pairs for retrieval evaluation,
same approach as LLM Zoomcamp HW3: ask the LLM to write a question a user
might ask that this specific chunk answers.

Requires OPENAI_API_KEY in the environment.

Usage:
    python src/generate_ground_truth.py
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from search import load_docs

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = "gpt-5.4-mini"

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "ground_truth.json"

PROMPT = """\
You are helping build a test set for a music-knowledge Q&A system.
Given the excerpt below, write ONE natural question that a curious
music fan might ask, whose answer is contained in this excerpt.
Return ONLY the question text, nothing else.

Topic: {topic}
Excerpt: {chunk}
"""


def generate_question(doc):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT.format(**doc)}],
        temperature=0.7,
    )
    question = resp.choices[0].message.content.strip()
    return {"question": question, "doc_id": doc["id"]}


def main():
    docs = load_docs()
    results = []

    # Concurrent generation with backoff-friendly low parallelism,
    # same fix you used in HW3 after hitting rate limits.
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(generate_question, d): d for d in docs}
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Failed on {futures[future]['id']}: {e}")

    OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Saved {len(results)} ground-truth pairs -> {OUT_PATH}")


if __name__ == "__main__":
    main()
