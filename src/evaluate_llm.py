"""
Evaluates two prompt variants for the final answer-generation step using
an LLM-as-a-judge (relevance: RELEVANT / PARTLY_RELEVANT / NON_RELEVANT),
same style as the course's RAG evaluation approach.

Requires OPENAI_API_KEY. Requires data/ground_truth.json (run
generate_ground_truth.py first).

Usage:
    python src/evaluate_llm.py
"""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from openai import OpenAI

from search import build_text_index, build_vector_index, hybrid_search, load_docs

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = "gpt-5.4-mini"

GT_PATH = Path(__file__).resolve().parent.parent / "data" / "ground_truth.json"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "llm_eval.json"

PROMPT_A = """\
You are a knowledgeable music encyclopedia assistant. Answer the QUESTION
using only the CONTEXT below. If the context doesn't contain the answer,
say you don't have enough information.

CONTEXT:
{context}

QUESTION: {question}
"""

PROMPT_B = """\
Answer the user's question about music as a friendly, concise expert.
Base your answer strictly on the CONTEXT provided — do not use outside
knowledge. Keep the answer to 2-3 sentences. If the context is
insufficient, say so explicitly.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:
"""

JUDGE_PROMPT = """\
You are evaluating the quality of an answer to a question, given the
context it was generated from.

QUESTION: {question}
CONTEXT: {context}
ANSWER: {answer}

Classify the answer as one of: "RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT".
Respond with only the label.
"""


def build_context(chunks):
    return "\n\n".join(f"[{c['topic']}] {c['chunk']}" for c in chunks)


def generate_answer(prompt_template, context, question):
    prompt = prompt_template.format(context=context, question=question)
    resp = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}])
    return resp.choices[0].message.content


def judge(question, context, answer):
    prompt = JUDGE_PROMPT.format(question=question, context=context, answer=answer)
    resp = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}])
    return resp.choices[0].message.content.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=150)
    args = parser.parse_args()

    docs = load_docs()
    text_index = build_text_index(docs)
    vindex, embedder = build_vector_index(docs)
    ground_truth = json.loads(GT_PATH.read_text(encoding="utf-8"))
    
    if len(ground_truth) > args.max_samples:
        by_topic = defaultdict(list)
        for gt in ground_truth:
            topic = gt["doc_id"].rsplit("-", 1)[0]
            by_topic[topic].append(gt)
            
        sampled_gt = []
        total_gt = len(ground_truth)
        for topic, items in by_topic.items():
            # max(1, ...) ensures small topics get at least 1 sample
            target_count = max(1, int(round(len(items) / total_gt * args.max_samples)))
            sampled_gt.extend(random.sample(items, min(target_count, len(items))))
            
        random.shuffle(sampled_gt)
        ground_truth = sampled_gt[:args.max_samples]

    results = {"prompt_a": [], "prompt_b": []}
    for gt in ground_truth:
        chunks = hybrid_search(text_index, vindex, embedder, gt["question"], num_results=5)
        context = build_context(chunks)

        for name, template in [("prompt_a", PROMPT_A), ("prompt_b", PROMPT_B)]:
            answer = generate_answer(template, context, gt["question"])
            verdict = judge(gt["question"], context, answer)
            results[name].append(verdict)

    summary = {}
    for name, verdicts in results.items():
        summary[name] = {v: verdicts.count(v) / len(verdicts) for v in set(verdicts)}

    print(json.dumps(summary, indent=2))
    OUT_PATH.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
