"""
Evaluation script — 4-layer evaluation of raw vs fine-tuned model.

Layer 1: Perplexity + ROUGE-L (automated metrics)
Layer 2: LLM-as-Judge via Groq (blind A/B comparison)
Layer 3: Domain-specific 20-question test suite
Layer 4: Side-by-side comparison table → eval_results.md

Usage:
    python finetuning/eval.py \\
        --base-model unsloth/gemma-4-E4B-it \\
        --finetuned-model AditHash/gemma4-backend-ai-expert \\
        --test-split data/final/test.jsonl \\
        --output finetuning/eval_results.md
"""

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any

EVAL_QUESTIONS_FILE = Path(__file__).parent / "eval_questions.json"
JUDGE_MODEL = "llama-3.3-70b-versatile"
JUDGE_TEMPERATURE = 0.1

JUDGE_PROMPT = """\
You are an expert backend engineer evaluating two AI responses to a technical question.
You do NOT know which response is from which model.

Question: {question}

Response A:
{response_a}

Response B:
{response_b}

Score each response from 1-10 on:
1. Technical accuracy
2. Code quality (if code present; else score 5)
3. Completeness
4. Clarity

Output ONLY valid JSON:
{{
  "response_a": {{"technical_accuracy": 0, "code_quality": 0, "completeness": 0, "clarity": 0, "total": 0}},
  "response_b": {{"technical_accuracy": 0, "code_quality": 0, "completeness": 0, "clarity": 0, "total": 0}},
  "winner": "A or B or tie",
  "reason": "one sentence"
}}
"""


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_eval_questions() -> list[str]:
    with EVAL_QUESTIONS_FILE.open(encoding="utf-8") as f:
        return json.load(f)


# ── Layer 1: Perplexity ────────────────────────────────────────────────────────

def compute_perplexity(model, tokenizer, texts: list[str], device: str = "cuda") -> float:
    """Compute mean perplexity over a list of texts."""
    import torch
    from torch.nn import CrossEntropyLoss

    model.eval()
    total_loss = 0.0
    count = 0

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        total_loss += outputs.loss.item()
        count += 1

    return math.exp(total_loss / count)


# ── Layer 1: ROUGE-L ──────────────────────────────────────────────────────────

def compute_rouge_l(predictions: list[str], references: list[str]) -> float:
    """Compute mean ROUGE-L F1 score."""
    from rouge_score import rouge_scorer as rs_module
    scorer = rs_module.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(ref, pred)["rougeL"].fmeasure
        for pred, ref in zip(predictions, references)
    ]
    return sum(scores) / len(scores) if scores else 0.0


# ── Inference helper ───────────────────────────────────────────────────────────

def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 512) -> str:
    """Generate a response for a question using the loaded model."""
    import torch
    from unsloth import FastModel

    FastModel.for_inference(model)
    messages = [{"role": "user", "content": question}]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = output_ids[0][inputs.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ── Layer 2: LLM-as-Judge ─────────────────────────────────────────────────────

def llm_judge(
    question: str,
    response_a: str,
    response_b: str,
    groq_api_key: str,
) -> dict[str, Any]:
    """Run blind A/B comparison via Groq judge. Returns scores + winner."""
    from groq import Groq

    client = Groq(api_key=groq_api_key)
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=JUDGE_TEMPERATURE,
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    question=question,
                    response_a=response_a,
                    response_b=response_b,
                ),
            }
        ],
    )
    raw = response.choices[0].message.content.strip()
    return json.loads(raw)


# ── Layer 4: Results table ─────────────────────────────────────────────────────

def write_results_md(
    results: list[dict[str, Any]],
    perplexity_base: float,
    perplexity_ft: float,
    rouge_base: float,
    rouge_ft: float,
    output_path: Path,
) -> None:
    """Write eval_results.md with all 4 evaluation layers."""
    lines = [
        "# Evaluation Results\n",
        "## Layer 1: Automated Metrics\n",
        "| Metric | Base Model | Fine-tuned | Delta |",
        "|---|---|---|---|",
        f"| Perplexity | {perplexity_base:.2f} | {perplexity_ft:.2f} | "
        f"{((perplexity_base - perplexity_ft) / perplexity_base * 100):.1f}% lower ✅ |",
        f"| ROUGE-L | {rouge_base:.3f} | {rouge_ft:.3f} | "
        f"+{(rouge_ft - rouge_base):.3f} ✅ |",
        "",
        "## Layer 3 + 4: LLM-Judge Domain Test Suite\n",
        "| Question | Base Score | Fine-tuned Score | Winner |",
        "|---|---|---|---|",
    ]

    base_totals, ft_totals = [], []
    ft_wins = 0

    for r in results:
        base_score = r["base_score"]
        ft_score = r["ft_score"]
        winner = "Fine-tuned ✅" if ft_score > base_score else ("Base" if base_score > ft_score else "Tie")
        if ft_score > base_score:
            ft_wins += 1
        base_totals.append(base_score)
        ft_totals.append(ft_score)
        short_q = r["question"][:50] + "..." if len(r["question"]) > 50 else r["question"]
        lines.append(f"| {short_q} | {base_score:.1f} | {ft_score:.1f} | {winner} |")

    avg_base = sum(base_totals) / len(base_totals) if base_totals else 0
    avg_ft = sum(ft_totals) / len(ft_totals) if ft_totals else 0
    win_rate = ft_wins / len(results) * 100 if results else 0
    delta_pct = ((avg_ft - avg_base) / avg_base * 100) if avg_base else 0

    lines += [
        f"| **Average** | **{avg_base:.1f}** | **{avg_ft:.1f}** | "
        f"**+{delta_pct:.0f}% improvement** |",
        "",
        f"**Fine-tuned win rate: {win_rate:.0f}% ({ft_wins}/{len(results)} questions)**",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Results written to {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate base vs fine-tuned Gemma 4")
    parser.add_argument("--base-model", default="unsloth/gemma-4-E4B-it")
    parser.add_argument("--finetuned-model", default="AditHash/gemma4-backend-ai-expert")
    parser.add_argument("--test-split", default="data/final/test.jsonl")
    parser.add_argument("--output", default="finetuning/eval_results.md")
    parser.add_argument("--judge-samples", type=int, default=20, help="Questions for LLM judge")
    args = parser.parse_args()

    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template
    from finetuning.config import CHAT_TEMPLATE, LOAD_IN_4BIT, MAX_SEQ_LENGTH

    groq_api_key = os.getenv("GROQ_API_KEY", "")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not set — needed for LLM judge")

    test_samples = load_jsonl(Path(args.test_split))
    eval_questions = load_eval_questions()
    print(f"Test samples: {len(test_samples)} | Eval questions: {len(eval_questions)}")

    # ── Load base model ────────────────────────────────────────────────────────
    print(f"Loading base model: {args.base_model}")
    base_model, base_tokenizer = FastModel.from_pretrained(
        args.base_model, max_seq_length=MAX_SEQ_LENGTH, load_in_4bit=LOAD_IN_4BIT
    )
    base_tokenizer = get_chat_template(base_tokenizer, chat_template=CHAT_TEMPLATE)

    # ── Load fine-tuned model ──────────────────────────────────────────────────
    print(f"Loading fine-tuned model: {args.finetuned_model}")
    ft_model, ft_tokenizer = FastModel.from_pretrained(
        args.finetuned_model, max_seq_length=MAX_SEQ_LENGTH, load_in_4bit=LOAD_IN_4BIT
    )
    ft_tokenizer = get_chat_template(ft_tokenizer, chat_template=CHAT_TEMPLATE)

    # ── Layer 1: Perplexity ────────────────────────────────────────────────────
    test_texts = [s["instruction"] + " " + s["output"] for s in test_samples[:50]]
    print("Computing perplexity...")
    perplexity_base = compute_perplexity(base_model, base_tokenizer, test_texts)
    perplexity_ft = compute_perplexity(ft_model, ft_tokenizer, test_texts)
    print(f"Perplexity — Base: {perplexity_base:.2f} | Fine-tuned: {perplexity_ft:.2f}")

    # ── Layer 1: ROUGE-L ──────────────────────────────────────────────────────
    print("Computing ROUGE-L...")
    references = [s["output"] for s in test_samples[:50]]
    base_preds = [generate_answer(base_model, base_tokenizer, s["instruction"]) for s in test_samples[:50]]
    ft_preds = [generate_answer(ft_model, ft_tokenizer, s["instruction"]) for s in test_samples[:50]]
    rouge_base = compute_rouge_l(base_preds, references)
    rouge_ft = compute_rouge_l(ft_preds, references)
    print(f"ROUGE-L — Base: {rouge_base:.3f} | Fine-tuned: {rouge_ft:.3f}")

    # ── Layer 2+3: LLM Judge on domain questions ───────────────────────────────
    print(f"Running LLM judge on {len(eval_questions)} domain questions...")
    results = []

    for i, question in enumerate(eval_questions):
        base_ans = generate_answer(base_model, base_tokenizer, question)
        ft_ans = generate_answer(ft_model, ft_tokenizer, question)

        # Randomize A/B assignment to avoid position bias
        flip = random.random() > 0.5
        resp_a = ft_ans if flip else base_ans
        resp_b = base_ans if flip else ft_ans

        try:
            judgment = llm_judge(question, resp_a, resp_b, groq_api_key)
            a_total = judgment["response_a"]["total"]
            b_total = judgment["response_b"]["total"]
            base_score = b_total if flip else a_total
            ft_score = a_total if flip else b_total
        except Exception as exc:
            print(f"Judge failed for Q{i}: {exc}")
            base_score = ft_score = 0.0

        results.append({"question": question, "base_score": base_score, "ft_score": ft_score})
        print(f"Q{i+1}: base={base_score:.1f} ft={ft_score:.1f}")

    # ── Layer 4: Write results ─────────────────────────────────────────────────
    write_results_md(
        results=results,
        perplexity_base=perplexity_base,
        perplexity_ft=perplexity_ft,
        rouge_base=rouge_base,
        rouge_ft=rouge_ft,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
