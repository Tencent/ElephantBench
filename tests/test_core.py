from __future__ import annotations

import json
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

from elephantbench.analysis import analyze
from elephantbench.client import OpenAIChatClient, parse_headers
from elephantbench.inference import build_parser as build_inference_parser
from elephantbench.judge import build_judge_input, parse_judgment
from elephantbench.judge import build_parser as build_judge_parser
from elephantbench.metrics import summarize
from elephantbench.schema import validate_item
from elephantbench.validate import validate_dataset


def item(benchmark_id: str) -> dict:
    return {
        "benchmark_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"test-benchmark:{benchmark_id}")),
        "item_group_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"test-group:{benchmark_id}")),
        "eval": {
            "question": "What values are reported?",
            "gold_answers": [{"value": "1"}, {"value": "2"}],
            "preferred_answer": "Reports give 1 and 2.",
        },
    }


class MetricsTests(unittest.TestCase):
    def test_fixed_denominator_and_conditional_completeness(self) -> None:
        benchmark = {"a": item("a"), "b": item("b"), "c": item("c")}
        results = [
            {"benchmark_id": "a", "status": "success", "judgment": {"label": "complete"}},
            {"benchmark_id": "b", "status": "success", "judgment": {"label": "partial"}},
        ]
        report = summarize(benchmark, results)
        self.assertEqual(report["counts"], {"complete": 1, "partial": 1, "failed": 1})
        self.assertAlmostEqual(report["rates"]["K"], 0.5)
        self.assertAlmostEqual(sum(report["rates"][key] for key in ("C", "P", "F")), 1.0)

    def test_oracle_analysis_adds_complementary_models(self) -> None:
        benchmark = {"a": item("a"), "b": item("b")}
        report = analyze(
            benchmark,
            {
                "left": {"a": "complete", "b": "failed"},
                "right": {"a": "failed", "b": "complete"},
            },
        )
        curve = report["greedy_oracle"]
        self.assertEqual(curve[0]["C"], 0.5)
        self.assertEqual(curve[1]["C"], 1.0)


class JudgeTests(unittest.TestCase):
    def test_judge_input_uses_golds_without_an_item_rubric(self) -> None:
        payload = json.loads(build_judge_input(item("a"), "One value is 1."))
        self.assertEqual(set(payload), {"question", "verified_answers", "model_answer"})
        self.assertEqual(payload["verified_answers"], ["1", "2"])

    def test_json_and_fenced_json_are_supported(self) -> None:
        expected = {"label": "partial", "rationale": "one value"}
        self.assertEqual(parse_judgment(json.dumps(expected)), expected)
        self.assertEqual(parse_judgment("```json\n" + json.dumps(expected) + "\n```"), expected)

    def test_detailed_credit_is_normalized_and_checked(self) -> None:
        detailed = {
            "gold_assessments": [
                {"covered": True, "evidence": "first"},
                {"covered": False, "evidence": ""},
            ],
            "material_contradictions": [],
            "credit": "partial_credit",
            "reasoning": "Only one account is covered.",
        }
        parsed = parse_judgment(json.dumps(detailed), expected_gold_count=2)
        self.assertEqual(parsed["label"], "partial")
        self.assertEqual(parsed["credit"], "partial_credit")
        detailed["credit"] = "full_credit"
        with self.assertRaisesRegex(ValueError, "inconsistent with coverage"):
            parse_judgment(json.dumps(detailed), expected_gold_count=2)


class ClientTests(unittest.TestCase):
    def test_openai_compatible_request(self) -> None:
        response = MagicMock()
        response.read.return_value = json.dumps(
            {
                "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 1},
            }
        ).encode()
        context = MagicMock()
        context.__enter__.return_value = response
        with patch("urllib.request.urlopen", return_value=context) as urlopen:
            result = OpenAIChatClient("https://example.test/v1", "test-key", "example-model").chat(
                "system", "question", max_tokens=123
            )

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, "https://example.test/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(payload["model"], "example-model")
        self.assertEqual(payload["max_tokens"], 123)
        self.assertFalse(payload["stream"])
        self.assertEqual(result["content"], "answer")

    def test_additional_headers_must_use_name_value_syntax(self) -> None:
        self.assertEqual(parse_headers(["X-Route=public"]), {"X-Route": "public"})
        with self.assertRaises(ValueError):
            parse_headers(["invalid"])

    def test_openai_request_may_omit_endpoint_selected_model(self) -> None:
        response = MagicMock()
        response.read.return_value = json.dumps(
            {"choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}]}
        ).encode()
        context = MagicMock()
        context.__enter__.return_value = response
        with patch("urllib.request.urlopen", return_value=context) as urlopen:
            OpenAIChatClient("https://example.test/v1", "", "").chat("system", "question")
        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertNotIn("model", payload)


class DatasetTests(unittest.TestCase):
    def test_packaged_run_and_judge_prompts_are_defaults(self) -> None:
        run_args = build_inference_parser().parse_args(
            ["--input", "data.jsonl", "--output", "responses.jsonl", "--model", "model"]
        )
        judge_args = build_judge_parser().parse_args(
            [
                "--benchmark",
                "data.jsonl",
                "--responses",
                "responses.jsonl",
                "--output",
                "judged.jsonl",
                "--judge-model",
                "judge",
            ]
        )
        self.assertTrue(run_args.system_prompt.is_file())
        self.assertTrue(judge_args.judge_prompt.is_file())

    def test_dataset_is_valid_and_complete(self) -> None:
        dataset = Path(__file__).resolve().parents[1] / "data" / "elephantbench.jsonl"
        report = validate_dataset(dataset)
        self.assertTrue(report["valid"], report["errors"][:5])
        self.assertEqual(report["records"], 1094)
        self.assertEqual(report["unique_ids"], 1094)
        self.assertEqual(report["unique_questions"], 1094)
        self.assertEqual(report["item_groups"], 380)
        self.assertEqual(report["singleton_groups"], 0)

    def test_empty_dataset_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "empty.jsonl"
            dataset.write_text("", encoding="utf-8")
            report = validate_dataset(dataset)
        self.assertFalse(report["valid"])
        self.assertIn("dataset contains no records", report["errors"])

    def test_group_and_descriptive_fields_are_required(self) -> None:
        record = item("a")
        record.pop("item_group_id")
        record["eval"]["preferred_answer"] = ""
        errors = validate_item(record)
        self.assertIn("item_group_id must be a non-empty string", errors)
        self.assertIn("eval.preferred_answer must be a non-empty string", errors)

    def test_removed_metadata_is_not_part_of_the_release_schema(self) -> None:
        record = item("a")
        record["eval"]["difficulty"] = "unsupported"
        record["eval"]["answer_type"] = "unsupported"
        record["eval"]["slot"] = "unsupported"
        errors = validate_item(record)
        for field in ("difficulty", "answer_type", "slot"):
            self.assertIn(f"eval.{field} is not part of the release schema", errors)

    def test_duplicate_gold_values_are_rejected(self) -> None:
        record = item("a")
        record["eval"]["gold_answers"] = [{"value": "One"}, {"value": " one "}]
        self.assertIn("eval.gold_answers must not contain duplicate values", validate_item(record))

    def test_at_least_two_gold_values_are_required(self) -> None:
        record = item("a")
        record["eval"]["gold_answers"] = [{"value": "One"}]
        self.assertIn("eval.gold_answers must contain at least two items", validate_item(record))

    def test_duplicate_questions_and_singleton_groups_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            duplicate_dataset = Path(directory) / "duplicates.jsonl"
            left, right = item("left"), item("right")
            right["item_group_id"] = left["item_group_id"]
            right["eval"]["question"] = "  WHAT VALUES ARE REPORTED?  "
            duplicate_dataset.write_text(
                json.dumps(left) + "\n" + json.dumps(right) + "\n", encoding="utf-8"
            )
            duplicate_report = validate_dataset(duplicate_dataset)
            singleton_dataset = Path(directory) / "singleton.jsonl"
            singleton_dataset.write_text(json.dumps(item("single")) + "\n", encoding="utf-8")
            singleton_report = validate_dataset(singleton_dataset)
        self.assertFalse(duplicate_report["valid"])
        self.assertTrue(
            any("duplicate normalized question" in error for error in duplicate_report["errors"])
        )
        self.assertFalse(singleton_report["valid"])
        self.assertEqual(singleton_report["singleton_groups"], 1)

    def test_ids_must_be_canonical_uuids(self) -> None:
        record = item("a")
        record["benchmark_id"] = "RV-0001-q1"
        record["item_group_id"] = str(uuid.uuid4()).upper()
        errors = validate_item(record)
        self.assertIn("benchmark_id must be a canonical UUID string", errors)
        self.assertIn("item_group_id must be a canonical UUID string", errors)

    def test_released_ids_are_canonical_uuids(self) -> None:
        dataset = Path(__file__).resolve().parents[1] / "data" / "elephantbench.jsonl"
        for row in map(json.loads, dataset.read_text(encoding="utf-8").splitlines()):
            self.assertEqual(str(uuid.UUID(row["benchmark_id"])), row["benchmark_id"])
            self.assertEqual(str(uuid.UUID(row["item_group_id"])), row["item_group_id"])


if __name__ == "__main__":
    unittest.main()
