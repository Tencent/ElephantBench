from __future__ import annotations

import uuid
from typing import Any


def _is_canonical_uuid(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return str(uuid.UUID(value)) == value
    except ValueError:
        return False


def validate_item(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    benchmark_id = record.get("benchmark_id")
    if not isinstance(benchmark_id, str) or not benchmark_id.strip():
        errors.append("benchmark_id must be a non-empty string")
    elif not _is_canonical_uuid(benchmark_id):
        errors.append("benchmark_id must be a canonical UUID string")
    item_group_id = record.get("item_group_id")
    if not isinstance(item_group_id, str) or not item_group_id.strip():
        errors.append("item_group_id must be a non-empty string")
    elif not _is_canonical_uuid(item_group_id):
        errors.append("item_group_id must be a canonical UUID string")

    evaluation = record.get("eval")
    if not isinstance(evaluation, dict):
        return errors + ["eval must be an object"]
    if not isinstance(evaluation.get("question"), str) or not evaluation["question"].strip():
        errors.append("eval.question must be a non-empty string")
    for removed_field in ("difficulty", "answer_type", "slot"):
        if removed_field in evaluation:
            errors.append(f"eval.{removed_field} is not part of the release schema")
    preferred_answer = evaluation.get("preferred_answer")
    if not isinstance(preferred_answer, str) or not preferred_answer.strip():
        errors.append("eval.preferred_answer must be a non-empty string")
    golds = evaluation.get("gold_answers")
    if not isinstance(golds, list):
        errors.append("eval.gold_answers must be a list")
    else:
        if len(golds) < 2:
            errors.append("eval.gold_answers must contain at least two items")
        normalized_values: list[str] = []
        for index, gold in enumerate(golds):
            if (
                not isinstance(gold, dict)
                or not isinstance(gold.get("value"), str)
                or not gold["value"].strip()
            ):
                errors.append(f"eval.gold_answers[{index}].value must be a non-empty string")
            else:
                normalized_values.append(gold["value"].strip().casefold())
        if len(normalized_values) != len(set(normalized_values)):
            errors.append("eval.gold_answers must not contain duplicate values")
    return errors
