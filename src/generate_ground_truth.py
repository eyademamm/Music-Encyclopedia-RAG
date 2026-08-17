import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

from search import load_docs

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = "gpt-5.4-mini"

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "ground_truth.json"

class GeneratedQuestions(BaseModel):
    questions: list[str]

PROMPT = """\
You are helping build a test set for a music-knowledge Q&A system.
Given the excerpt below, write 3 to 5 natural questions that a curious
music fan might ask, whose answers are contained in this excerpt.
Return varied question types (factual, comparative, explanatory).

Topic: {topic}
Excerpt: {chunk}
"""

def generate_questions(doc):
    resp = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT.format(**doc)}],
        response_format=GeneratedQuestions,
        temperature=0.7,
    )
    parsed = resp.choices[0].message.parsed
    return [
        {"question": q, "doc_id": doc["id"]}
        for q in parsed.questions
    ]

def main():
    docs = load_docs()
    results = []

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(generate_questions, d): d for d in docs}
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                results.extend(future.result())
            except Exception as e:
                print(f"Failed on {futures[future]['id']}: {e}")

    OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Saved {len(results)} ground-truth pairs -> {OUT_PATH}")


if __name__ == "__main__":
    main()
