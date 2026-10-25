"""Member 3 experiments: prompt engineering, decoding, and evaluation.

This module deliberately depends on the dataset and model loader supplied by
Member 1. It owns only the assignment's E2 prompt comparison, E3 decoding
study, and the comparative evaluation matrix used in report sections 7, 8,
and 11.

The module is intended to run on a Google Colab GPU. Example:

    python -m src.member3_experiments --sample-limit 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


DEFAULT_DATA_PATH = "data/experimental_subset.csv"
DEFAULT_OUTPUT_DIR = "data/member3_results"
DEFAULT_SEED = 42


PROMPT_VARIANTS: Mapping[str, str] = {
    "zero_shot": (
        "Summarize the following conversation.\n\n"
        "Conversation:\n{dialogue}\n\nSummary:"
    ),
    "few_shot": (
        "Summarize conversations by preserving the people, decisions, and "
        "important next steps.\n\n"
        "Example conversation:\n"
        "Maya: I sent the revised budget.\n"
        "Noah: Thanks, I will review it by Friday.\n"
        "Example summary:\n"
        "Maya sent the revised budget, and Noah will review it by Friday.\n\n"
        "Conversation:\n{dialogue}\n\nSummary:"
    ),
    "role_context": (
        "You are a professional meeting-notes assistant. A reader who did not "
        "see the conversation must understand its main topic, decisions, and "
        "commitments. Write an accurate summary of the conversation below.\n\n"
        "Conversation:\n{dialogue}\n\nSummary:"
    ),
    "constraint_based": (
        "Summarize the conversation below in no more than two sentences. "
        "Use only information stated in the conversation. Preserve names, "
        "decisions, deadlines, and action items when present. Do not add a "
        "heading, explanation, or unsupported detail.\n\n"
        "Conversation:\n{dialogue}\n\nSummary:"
    ),
}


DECODING_CONFIGS: Mapping[str, Mapping[str, Any]] = {
    "greedy": {
        "do_sample": False,
    },
    "temperature": {
        "do_sample": True,
        "temperature": 0.7,
    },
    "top_k": {
        "do_sample": True,
        "temperature": 0.7,
        "top_k": 50,
    },
    "top_p": {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
    },
}


@dataclass(frozen=True)
class GenerationRecord:
    """One traceable output from E2 or E3."""

    experiment: str
    variant: str
    trial: int
    sample_id: str
    dialogue: str
    reference_summary: str
    prompt: str
    generated_summary: str
    decoding_parameters: str
    latency_seconds: float


def set_reproducible_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed Python, NumPy, and Torch for a repeatable comparison."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_experiment_samples(
    data_path: str = DEFAULT_DATA_PATH, sample_limit: int = 3
) -> pd.DataFrame:
    """Load a deterministic slice of the shared experimental subset."""

    if sample_limit < 1:
        raise ValueError("sample_limit must be at least 1")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"{data_path} was not found. Run Member 1's data preparation first."
        )

    samples = pd.read_csv(data_path, dtype={"id": str})
    required_columns = {"id", "dialogue", "summary"}
    missing = required_columns.difference(samples.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if samples.empty:
        raise ValueError("The experimental dataset is empty")

    return samples.head(sample_limit).copy()


def build_prompt(variant: str, dialogue: str) -> str:
    """Render one of the four E2 prompt strategies."""

    if variant not in PROMPT_VARIANTS:
        raise KeyError(
            f"Unknown prompt variant '{variant}'. "
            f"Choose one of: {', '.join(PROMPT_VARIANTS)}"
        )
    return PROMPT_VARIANTS[variant].format(dialogue=dialogue.strip())


def _model_input_device(model: Any) -> torch.device:
    """Find a usable input device for regular and device-mapped models."""

    device = getattr(model, "device", None)
    if device is not None and str(device) != "meta":
        return torch.device(device)

    device_map = getattr(model, "hf_device_map", {})
    for mapped_device in device_map.values():
        if mapped_device not in {"cpu", "disk"}:
            return torch.device(
                f"cuda:{mapped_device}" if isinstance(mapped_device, int) else mapped_device
            )
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_summary(
    model: Any,
    tokenizer: Any,
    prompt: str,
    decoding_parameters: Mapping[str, Any],
    max_new_tokens: int = 150,
) -> tuple[str, float]:
    """Generate only the continuation, excluding the original prompt tokens."""

    inputs = tokenizer(prompt, return_tensors="pt")
    device = _model_input_device(model)
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    input_length = inputs["input_ids"].shape[-1]

    generation_args = dict(decoding_parameters)
    generation_args.update(
        {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
        }
    )

    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_args)
    latency = time.perf_counter() - started

    continuation = output_ids[0, input_length:]
    summary = tokenizer.decode(continuation, skip_special_tokens=True).strip()
    return summary, latency


def run_prompt_comparison(
    model: Any,
    tokenizer: Any,
    samples: pd.DataFrame,
    max_new_tokens: int = 150,
) -> List[GenerationRecord]:
    """Run E2: four required prompt variants under fixed greedy decoding."""

    records: List[GenerationRecord] = []
    decoding = DECODING_CONFIGS["greedy"]
    for variant in PROMPT_VARIANTS:
        for row in samples.itertuples(index=False):
            prompt = build_prompt(variant, row.dialogue)
            summary, latency = generate_summary(
                model, tokenizer, prompt, decoding, max_new_tokens
            )
            records.append(
                GenerationRecord(
                    experiment="E2_prompt_comparison",
                    variant=variant,
                    trial=1,
                    sample_id=str(row.id),
                    dialogue=row.dialogue,
                    reference_summary=row.summary,
                    prompt=prompt,
                    generated_summary=summary,
                    decoding_parameters=json.dumps(decoding, sort_keys=True),
                    latency_seconds=round(latency, 4),
                )
            )
    return records


def run_decoding_study(
    model: Any,
    tokenizer: Any,
    samples: pd.DataFrame,
    prompt_variant: str = "constraint_based",
    max_new_tokens: int = 150,
    sampling_repeats: int = 3,
) -> List[GenerationRecord]:
    """Run E3 while holding the prompt constant and changing decoding only.

    Sampling configurations are repeated so their variability can be measured.
    Greedy decoding is deterministic and therefore needs only one trial.
    """

    if sampling_repeats < 2:
        raise ValueError("sampling_repeats must be at least 2")
    records: List[GenerationRecord] = []
    for config_name, decoding in DECODING_CONFIGS.items():
        for row in samples.itertuples(index=False):
            prompt = build_prompt(prompt_variant, row.dialogue)
            trial_count = sampling_repeats if decoding.get("do_sample") else 1
            for trial in range(1, trial_count + 1):
                summary, latency = generate_summary(
                    model, tokenizer, prompt, decoding, max_new_tokens
                )
                records.append(
                    GenerationRecord(
                        experiment="E3_decoding_study",
                        variant=config_name,
                        trial=trial,
                        sample_id=str(row.id),
                        dialogue=row.dialogue,
                        reference_summary=row.summary,
                        prompt=prompt,
                        generated_summary=summary,
                        decoding_parameters=json.dumps(decoding, sort_keys=True),
                        latency_seconds=round(latency, 4),
                    )
                )
    return records


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def rouge_1_scores(candidate: str, reference: str) -> Dict[str, float]:
    """Compute unigram overlap precision, recall, and F1 without extra packages."""

    candidate_counts = Counter(_tokens(candidate))
    reference_counts = Counter(_tokens(reference))
    overlap = sum((candidate_counts & reference_counts).values())
    precision = _safe_divide(overlap, sum(candidate_counts.values()))
    recall = _safe_divide(overlap, sum(reference_counts.values()))
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return {
        "rouge_1_precision": precision,
        "rouge_1_recall": recall,
        "rouge_1_f1": f1,
    }


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(candidate: str, reference: str) -> float:
    """Compute longest-common-subsequence F1."""

    candidate_tokens = _tokens(candidate)
    reference_tokens = _tokens(reference)
    lcs = _lcs_length(candidate_tokens, reference_tokens)
    precision = _safe_divide(lcs, len(candidate_tokens))
    recall = _safe_divide(lcs, len(reference_tokens))
    return _safe_divide(2 * precision * recall, precision + recall)


def evaluate_record(record: GenerationRecord) -> Dict[str, Any]:
    """Score an output on quality, coverage, conciseness, and efficiency."""

    candidate_tokens = _tokens(record.generated_summary)
    dialogue_tokens = _tokens(record.dialogue)
    overlap = rouge_1_scores(record.generated_summary, record.reference_summary)

    return {
        **asdict(record),
        **{name: round(value, 4) for name, value in overlap.items()},
        "rouge_l_f1": round(
            rouge_l_f1(record.generated_summary, record.reference_summary), 4
        ),
        "source_token_support": round(
            _safe_divide(
                sum((Counter(candidate_tokens) & Counter(dialogue_tokens)).values()),
                len(candidate_tokens),
            ),
            4,
        ),
        "compression_ratio": round(
            _safe_divide(len(candidate_tokens), len(dialogue_tokens)), 4
        ),
        "generated_word_count": len(candidate_tokens),
        "is_empty": int(not candidate_tokens),
    }


def build_evaluation_matrix(
    records: Iterable[GenerationRecord],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return detailed scores and an aggregate comparison table."""

    detail = pd.DataFrame(evaluate_record(record) for record in records)
    if detail.empty:
        raise ValueError("No generation records were supplied for evaluation")

    numeric_metrics = [
        "rouge_1_precision",
        "rouge_1_recall",
        "rouge_1_f1",
        "rouge_l_f1",
        "source_token_support",
        "compression_ratio",
        "generated_word_count",
        "latency_seconds",
        "is_empty",
    ]
    matrix = (
        detail.groupby(["experiment", "variant"], sort=False)[numeric_metrics]
        .mean()
        .reset_index()
    )
    per_sample_diversity = (
        detail.groupby(
            ["experiment", "variant", "sample_id"], sort=False
        )["generated_summary"]
        .agg(lambda outputs: outputs.nunique() / outputs.size)
        .rename("output_diversity")
        .reset_index()
    )
    diversity_by_variant = (
        per_sample_diversity.groupby(["experiment", "variant"], sort=False)[
            "output_diversity"
        ]
        .mean()
        .reset_index()
    )
    matrix = matrix.merge(
        diversity_by_variant, on=["experiment", "variant"], how="left"
    )
    for column in numeric_metrics + ["output_diversity"]:
        matrix[column] = matrix[column].map(
            lambda value: round(float(value), 4) if not math.isnan(value) else value
        )
    return detail, matrix


def save_results(
    records: Iterable[GenerationRecord], output_dir: str = DEFAULT_OUTPUT_DIR
) -> Dict[str, str]:
    """Save raw outputs, per-output metrics, and the report-ready matrix."""

    records = list(records)
    os.makedirs(output_dir, exist_ok=True)
    raw_path = os.path.join(output_dir, "generation_outputs.csv")
    detail_path = os.path.join(output_dir, "evaluation_details.csv")
    matrix_path = os.path.join(output_dir, "comparison_matrix.csv")

    pd.DataFrame(asdict(record) for record in records).to_csv(raw_path, index=False)
    detail, matrix = build_evaluation_matrix(records)
    detail.to_csv(detail_path, index=False)
    matrix.to_csv(matrix_path, index=False)
    return {"outputs": raw_path, "details": detail_path, "matrix": matrix_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-limit", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--sampling-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Member 3 generation experiments require a CUDA GPU. "
            "Run this module in the project's Google Colab T4 environment."
        )

    # Imported here so metric functions remain usable without loading the model.
    from src.model_pipeline import load_quantized_model

    set_reproducible_seed(args.seed)
    samples = load_experiment_samples(args.data_path, args.sample_limit)
    model, tokenizer = load_quantized_model()

    records = run_prompt_comparison(
        model, tokenizer, samples, args.max_new_tokens
    )
    records.extend(
        run_decoding_study(
            model,
            tokenizer,
            samples,
            max_new_tokens=args.max_new_tokens,
            sampling_repeats=args.sampling_repeats,
        )
    )
    paths = save_results(records, args.output_dir)

    print("Member 3 experiments completed.")
    for result_type, path in paths.items():
        print(f"{result_type}: {path}")


if __name__ == "__main__":
    main()
