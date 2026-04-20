#!/usr/bin/env python3
"""
GenAI Sandbox — RAG Evaluation Script
=======================================
Evaluates your RAG pipeline using Ragas metrics:

  • Faithfulness       — is the answer grounded in the retrieved context?
  • Answer Relevance   — does the answer actually address the question?
  • Context Recall     — did retrieval surface the right chunks?
  • Context Precision  — are retrieved chunks relevant (not noisy)?

Usage:
    python3 scripts/eval.py                         # uses built-in sample questions
    python3 scripts/eval.py --questions my_qs.json  # custom question set
    make eval                                        # shorthand

Output:
    A table printed to stdout + scores saved to eval_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# ── Bootstrap path so we can import src.llm_factory ──────────
HERE = Path(__file__).resolve().parent.parent  # genai-sandbox/
sys.path.insert(0, str(HERE))

load_dotenv(HERE / ".env")

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── Built-in sample questions (used when no --questions file given) ──
SAMPLE_QUESTIONS = [
    {
        "question": "What is the main topic of the ingested documents?",
        "ground_truth": "",   # leave empty → Ragas uses LLM-as-judge mode
    },
    {
        "question": "Summarise the key points in three sentences.",
        "ground_truth": "",
    },
    {
        "question": "What specific techniques or methods are described?",
        "ground_truth": "",
    },
    {
        "question": "Are there any limitations or caveats mentioned?",
        "ground_truth": "",
    },
    {
        "question": "What conclusions or recommendations are given?",
        "ground_truth": "",
    },
]

# ── ANSI colours ──────────────────────────────────────────────
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
NC     = "\033[0m"


def c(color: str, text: str) -> str:
    return f"{color}{text}{NC}"


def _check_api_alive() -> bool:
    try:
        r = requests.get(f"{API_URL}/", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _query_pipeline(question: str) -> dict:
    """Hit POST /query and return the response dict."""
    r = requests.post(
        f"{API_URL}/query",
        json={"question": question, "session_id": "eval"},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def _run_ragas_eval(samples: list[dict]) -> dict:
    """
    Build a Ragas Dataset from query results and evaluate it.
    Returns a dict of metric_name -> score (0.0 – 1.0).
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    print()
    print(c(CYAN, BOLD + f"  Querying pipeline with {len(samples)} questions…"))

    for i, sample in enumerate(samples, 1):
        q = sample["question"]
        print(f"  [{i}/{len(samples)}] {q[:70]}…" if len(q) > 70 else f"  [{i}/{len(samples)}] {q}")
        try:
            result = _query_pipeline(q)
            answer   = result.get("answer", "")
            contexts = [s["content"] for s in result.get("sources", [])] or [""]
        except Exception as exc:
            print(c(YELLOW, f"      ⚠ Query failed: {exc}"))
            answer   = ""
            contexts = [""]

        data["question"].append(q)
        data["answer"].append(answer)
        data["contexts"].append(contexts)
        data["ground_truth"].append(sample.get("ground_truth", ""))

    dataset = Dataset.from_dict(data)

    # Choose metrics based on whether ground_truth is provided
    has_ground_truth = any(gt for gt in data["ground_truth"])
    metrics = [faithfulness, answer_relevancy]
    if has_ground_truth:
        metrics += [context_recall, context_precision]
    else:
        metrics += [context_precision]

    print()
    print(c(CYAN, BOLD + "  Running Ragas evaluation…"))
    result = evaluate(dataset, metrics=metrics)
    return result


def _print_table(scores: dict) -> None:
    """Pretty-print a score table to stdout."""
    metric_labels = {
        "faithfulness":       "Faithfulness      (answer grounded in context?)",
        "answer_relevancy":   "Answer Relevance  (does it answer the question?)",
        "context_recall":     "Context Recall    (right chunks retrieved?)",
        "context_precision":  "Context Precision (no noisy chunks?)",
    }

    print()
    print(c(CYAN, BOLD + "╔══════════════════════════════════════════════════════════════╗"))
    print(c(CYAN, BOLD + "║          RAG Evaluation Results  (Ragas)                     ║"))
    print(c(CYAN, BOLD + "╚══════════════════════════════════════════════════════════════╝"))
    print()

    for key, label in metric_labels.items():
        if key not in scores:
            continue
        val = scores[key]
        if isinstance(val, float):
            bar_len = int(val * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            colour = GREEN if val >= 0.7 else YELLOW if val >= 0.4 else RED
            print(f"  {label}")
            print(f"  {c(colour, bar)}  {c(BOLD, f'{val:.3f}')}")
            print()

    overall = sum(v for v in scores.values() if isinstance(v, float))
    count   = sum(1 for v in scores.values() if isinstance(v, float))
    avg = overall / count if count else 0.0
    overall_colour = GREEN if avg >= 0.7 else YELLOW if avg >= 0.4 else RED
    print(f"  {c(BOLD, 'Overall Average')}:  {c(overall_colour, f'{avg:.3f}')}")
    print()

    if avg < 0.4:
        print(c(RED, "  Diagnosis: Low scores suggest poor retrieval quality."))
        print(c(DIM, "  Try: smaller chunk_size, larger top_k, or better embedding model."))
    elif avg < 0.7:
        print(c(YELLOW, "  Diagnosis: Moderate quality — room for improvement."))
        print(c(DIM, "  Try: agentic_rag template, MMR search, or reranking."))
    else:
        print(c(GREEN, "  Diagnosis: Strong RAG quality. "))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline quality with Ragas.")
    parser.add_argument(
        "--questions", "-q",
        type=Path,
        help="Path to a JSON file: [{\"question\": \"...\", \"ground_truth\": \"...\"}]",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=HERE / "eval_results.json",
        help="Where to save scores (default: eval_results.json)",
    )
    args = parser.parse_args()

    # ── Check API is reachable ────────────────────────────────
    print()
    print(c(CYAN, BOLD + "╔══════════════════════════════════════════════╗"))
    print(c(CYAN, BOLD + "║     GenAI Sandbox — RAG Evaluator            ║"))
    print(c(CYAN, BOLD + "╚══════════════════════════════════════════════╝"))

    if not _check_api_alive():
        print(c(RED, f"\n  ✗ API not reachable at {API_URL}"))
        print(c(DIM, "  Start it first:  make serve"))
        sys.exit(1)
    print(c(GREEN, f"\n  ✓ API is online at {API_URL}"))

    # ── Load questions ────────────────────────────────────────
    if args.questions and args.questions.exists():
        samples = json.loads(args.questions.read_text())
        print(c(GREEN, f"  ✓ Loaded {len(samples)} questions from {args.questions}"))
    else:
        samples = SAMPLE_QUESTIONS
        print(c(DIM, "  Using built-in sample questions (pass --questions to customise)"))

    # ── Check Ragas is installed ──────────────────────────────
    try:
        import ragas  # noqa: F401
        import datasets  # noqa: F401
    except ImportError:
        print(c(RED, "\n  ✗ Ragas not installed."))
        print(c(DIM, "  Run:  pip install ragas datasets"))
        sys.exit(1)

    # ── Run evaluation ────────────────────────────────────────
    t0 = time.perf_counter()
    scores_obj = _run_ragas_eval(samples)
    elapsed = time.perf_counter() - t0

    # Ragas returns a dict-like EvaluationResult
    scores = dict(scores_obj)

    _print_table(scores)

    # ── Save results ──────────────────────────────────────────
    output = {
        "scores": {k: float(v) for k, v in scores.items() if isinstance(v, float)},
        "elapsed_seconds": round(elapsed, 2),
        "questions_evaluated": len(samples),
        "api_url": API_URL,
    }
    args.output.write_text(json.dumps(output, indent=2))
    print(c(DIM, f"  Results saved to {args.output}"))
    print()


if __name__ == "__main__":
    main()
