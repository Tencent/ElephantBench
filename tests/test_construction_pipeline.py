from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import threading
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

from elephantbench.client import OpenAIChatClient
from elephantbench.construction_pipeline import ner_tagging
from elephantbench.construction_pipeline.candidates import (
    CandidateConfig,
    build_candidates,
    evaluate_known_pair_recall,
)
from elephantbench.construction_pipeline.cli import build_parser, resolve_result_paths
from elephantbench.construction_pipeline.document_store import build_document_store
from elephantbench.construction_pipeline.export import export_records
from elephantbench.construction_pipeline.knowledge_tagging import (
    KnowledgeTagClient,
    load_taxonomy,
    normalize_tag_result,
    response_schema,
    validate_tag_result,
    visible_document,
)
from elephantbench.construction_pipeline.ner_tagging import (
    _input_jobs,
    _parse_devices,
    aggregate_mentions,
    normalize_entity,
)
from elephantbench.construction_pipeline.prepare import (
    PreparationConfig,
    prepare_anchors,
    prepare_document,
)
from elephantbench.construction_pipeline.relations import (
    _endpoint_parts,
    compact_candidate,
    iter_candidate_tasks,
    validate_relation,
)
from elephantbench.construction_pipeline.relations import (
    render_user_prompt as render_relation_prompt,
)
from elephantbench.construction_pipeline.source_io import (
    completed_document_ids,
    document_id,
    expand_inputs,
    iter_jsonl,
    open_text,
    selected_jsonl_records,
    write_jsonl_row,
)
from elephantbench.construction_pipeline.source_validation import (
    _load_by_subgraph as load_source_validation_input,
)
from elephantbench.construction_pipeline.source_validation import validate_source_result
from elephantbench.construction_pipeline.subgraphs import build_subgraphs
from elephantbench.construction_pipeline.synthesis import run_synthesis, validate_qa
from elephantbench.construction_pipeline.verification import (
    ToolEndpointClient,
    _load_source_validations,
    _load_synthesis,
    _quotation_failure_result,
    _seed_urls,
    _verification_input,
    validate_verification,
)
from elephantbench.construction_pipeline.web_tools import WebCache, WebTools, parse_brave_results
from elephantbench.io import read_jsonl


def anchor(
    doc_id: str,
    kp: str,
    *,
    number: str,
    tokens: list[str],
    domain: str,
) -> dict:
    discipline, field, subfield = kp.split("/")
    return {
        "doc_id": doc_id,
        "knowledge_point": {
            "key": "\t".join((discipline, field, subfield)),
            "discipline": discipline,
            "field": field,
            "subfield": subfield,
        },
        "confidence": 0.95,
        "subject": {"text": "Example Person", "norm": "example person", "label": "person"},
        "slot": "votes",
        "evidence": f"Example Person received {number} votes.",
        "dates": [],
        "numbers": [number],
        "tokens": tokens,
        "source_domain": domain,
    }


class RawTaggingTests(unittest.TestCase):
    def test_input_directory_discovery_is_recursive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "recycled_data_pool"
            nested.mkdir()
            shard = nested / "shard_00000000.jsonl"
            shard.write_text('{"text":"example","url":"https://example.test"}\n')
            self.assertEqual(expand_inputs([str(root)]), [shard.resolve()])

    @staticmethod
    def _taxonomy_path() -> Path:
        return (
            Path(__file__).resolve().parents[1]
            / "src"
            / "elephantbench"
            / "taxonomy"
            / "supergpqa_plus_taxonomy.json"
        )

    def test_kp_and_ner_share_stable_document_identity(self) -> None:
        record = {"url": "https://example.test/a", "text": "Example document."}
        expected = hashlib.sha1(b"https://example.test/a\nExample document.").hexdigest()
        self.assertEqual(document_id(record), expected)
        self.assertEqual(document_id(dict(record)), expected)

    def test_document_store_is_built_incrementally_from_source_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            rows = (
                [
                    {"url": "https://a.test", "text": "document a"},
                    {"url": "https://b.test", "text": "document b"},
                ],
                [
                    {"url": "https://a.test", "text": "document a"},
                    {"url": "https://c.test", "text": "document c"},
                ],
            )
            for index, shard_rows in enumerate(rows):
                path = source / f"shard_{index:08d}.jsonl.zstd"
                with open_text(path, "w") as handle:
                    for row in shard_rows:
                        write_jsonl_row(handle, row)

            store = root / "outputs" / "documents.sqlite"
            first = build_document_store([str(source)], store, max_shards=1, progress_every=0)
            self.assertEqual(first["processed_shards"], 1)
            self.assertEqual(first["document_rows"], 2)

            resumed = build_document_store([str(source)], store, progress_every=0)
            self.assertEqual(resumed["processed_shards"], 1)
            self.assertEqual(resumed["skipped_shards"], 1)
            self.assertEqual(resumed["completed_shards"], 2)
            self.assertEqual(resumed["rows_seen"], 4)
            self.assertEqual(resumed["document_rows"], 3)

            unchanged = build_document_store([str(source)], store, progress_every=0)
            self.assertEqual(unchanged["processed_shards"], 0)
            self.assertEqual(unchanged["skipped_shards"], 2)

            connection = sqlite3.connect(store)
            try:
                stored = connection.execute(
                    "SELECT canonical_doc_id,url,text FROM docs ORDER BY url"
                ).fetchall()
            finally:
                connection.close()
            self.assertEqual(
                [row[1] for row in stored], ["https://a.test", "https://b.test", "https://c.test"]
            )
            self.assertEqual(
                stored[0][0],
                hashlib.sha1(b"https://a.test\ndocument a").hexdigest(),
            )

    def test_packaged_taxonomy_schema_and_normalized_tag_feed_prepare(self) -> None:
        taxonomy = load_taxonomy(self._taxonomy_path())
        self.assertGreater(len(taxonomy["paths"]), 300)
        path = taxonomy["paths"][0]
        schema = response_schema(taxonomy)
        tag_array = schema["json_schema"]["schema"]["properties"]["tag_paths"]
        self.assertEqual((tag_array["minItems"], tag_array["maxItems"]), (1, 1))
        path_enum = schema["json_schema"]["schema"]["properties"]["tag_paths"]["items"][
            "properties"
        ]["path"]["enum"]
        self.assertIn(path, path_enum)
        normalized = normalize_tag_result(
            {
                "is_knowledge_bearing": True,
                "content_type": "news",
                "language": "en",
                "tag_paths": [
                    {
                        "path": path,
                        "confidence": 0.97,
                        "evidence_start_sentence": 1,
                        "evidence_end_sentence": 1,
                    }
                ],
                "reason": "The sentence states a factual result.",
            },
            taxonomy,
            {1: "Example Person received 10 votes in the final election tally."},
        )
        anchors, reason = prepare_document(
            {"document_id": "doc-a", **normalized},
            [
                {
                    "text": "Example Person",
                    "norm": "example person",
                    "label": "person",
                    "count": 1,
                    "first_start": 0,
                    "surface_forms": ["Example Person"],
                }
            ],
            PreparationConfig(min_evidence_chars=20),
        )
        self.assertIsNone(reason)
        self.assertTrue(anchors)

        invalid = {
            "is_knowledge_bearing": True,
            "content_type": "news",
            "language": "en",
            "tag_paths": [
                {
                    "path": path,
                    "confidence": 0.97,
                    "evidence_start_sentence": 1,
                    "evidence_end_sentence": 1,
                },
                {
                    "path": path,
                    "confidence": 0.96,
                    "evidence_start_sentence": 1,
                    "evidence_end_sentence": 1,
                },
            ],
            "reason": "Two labels are not allowed.",
        }
        self.assertIn(
            "tag_paths must contain exactly one item",
            validate_tag_result(invalid, taxonomy, {1: "Evidence."}),
        )

    def test_knowledge_tagging_preserves_the_full_document(self) -> None:
        marker = "UNIQUE_MIDDLE_EVIDENCE"
        document, sentences = visible_document(
            {"url": "https://example.test", "text": f"Head. {marker}. Tail."},
            max_sentence_chars=10_000,
        )
        self.assertIn(marker, document["text"])
        self.assertIn(marker, " ".join(sentences.values()))

    def test_jsonl_zstd_round_trip_resume_and_compressed_prepare(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stream = root / "rows.jsonl.zstd"
            with open_text(stream, "w") as handle:
                write_jsonl_row(handle, {"document_id": "a", "status": "success"})
                write_jsonl_row(handle, {"document_id": "b", "status": "error"})
            with open_text(stream, "a") as handle:
                write_jsonl_row(handle, {"document_id": "c", "status": "success"})
            self.assertEqual([row["document_id"] for _, row in iter_jsonl(stream)], ["a", "b", "c"])
            self.assertEqual(completed_document_ids(stream), {"a", "c"})

            kp_dir, ner_dir, output_dir = root / "kp", root / "ner", root / "anchors"
            kp_dir.mkdir()
            ner_dir.mkdir()
            with open_text(kp_dir / "shard_00000000.kp.jsonl.zstd", "w") as handle:
                write_jsonl_row(
                    handle,
                    {
                        "document_id": "doc-a",
                        "is_knowledge_bearing": True,
                        "language": "en",
                        "tags": [
                            {
                                "confidence": 0.95,
                                "discipline": "Public Affairs",
                                "field": "Government",
                                "subfield": "Elections",
                                "evidence": (
                                    "Example Person received 10 votes in the final election tally."
                                ),
                            }
                        ],
                    },
                )
            with open_text(ner_dir / "shard_00000000.ner.jsonl.zstd", "w") as handle:
                write_jsonl_row(
                    handle,
                    {
                        "document_id": "doc-a",
                        "entities": [
                            {
                                "text": "Example Person",
                                "norm": "example person",
                                "label": "person",
                                "count": 1,
                                "first_start": 0,
                                "surface_forms": ["Example Person"],
                            }
                        ],
                    },
                )
            report = prepare_anchors(
                kp_dir,
                ner_dir,
                output_dir,
                config=PreparationConfig(min_evidence_chars=20),
                workers=1,
            )
            self.assertGreater(report["anchors_written"], 0)

    def test_resume_repairs_a_truncated_zstd_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "interrupted.jsonl.zstd"
            with open_text(path, "w") as handle:
                write_jsonl_row(handle, {"document_id": "a", "status": "success"})
            first_frame_size = path.stat().st_size
            with open_text(path, "a") as handle:
                write_jsonl_row(
                    handle,
                    {"document_id": "b", "status": "success", "payload": "x" * 10_000},
                )
            complete_size = path.stat().st_size
            self.assertGreater(complete_size, first_frame_size)
            with path.open("r+b") as handle:
                handle.truncate(first_frame_size + (complete_size - first_frame_size) // 2)

            self.assertEqual(completed_document_ids(path), {"a"})
            with open_text(path, "a") as handle:
                write_jsonl_row(handle, {"document_id": "c", "status": "success"})
            self.assertEqual([row["document_id"] for _, row in iter_jsonl(path)], ["a", "c"])

    def test_resume_limit_selects_a_stable_source_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.jsonl"
            with open_text(path, "w") as handle:
                for document in "abcd":
                    write_jsonl_row(
                        handle,
                        {"document_id": document, "text": f"document {document}"},
                    )
            selected = list(selected_jsonl_records(path, {"a"}, limit=2))
            self.assertEqual([row["document_id"] for _, row in selected], ["b"])
            self.assertEqual(list(selected_jsonl_records(path, {"a", "b"}, limit=2)), [])

    def test_ner_normalization_and_aggregation_without_loading_torch(self) -> None:
        self.assertNotIn("torch", vars(ner_tagging))
        self.assertEqual(normalize_entity("U.S."), "united states")
        entities = aggregate_mentions(
            [
                {
                    "text": "Example Person",
                    "norm": "example person",
                    "label": "person",
                    "start": 5,
                    "end": 19,
                },
                {
                    "text": "EXAMPLE PERSON",
                    "norm": "example person",
                    "label": "person",
                    "start": 30,
                    "end": 44,
                },
            ],
            max_entities=10,
            max_surfaces=3,
        )
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]["count"], 2)

    def test_multi_device_ner_assigns_bounded_shard_jobs(self) -> None:
        inputs = [Path(f"shard_{index:08d}.jsonl.zstd") for index in range(8)]
        self.assertEqual(_parse_devices("cuda:0", "cuda:0,cuda:1"), ["cuda:0", "cuda:1"])
        jobs = _input_jobs(inputs, max_docs=17, max_docs_per_shard=0)
        self.assertEqual(sum(limit for _, limit in jobs), 17)
        self.assertEqual([limit for _, limit in jobs], [3, 2, 2, 2, 2, 2, 2, 2])
        per_shard = _input_jobs(inputs, max_docs=0, max_docs_per_shard=5)
        self.assertEqual({limit for _, limit in per_shard}, {5})
        with self.assertRaises(ValueError):
            _input_jobs(inputs, max_docs=1, max_docs_per_shard=1)

    def test_knowledge_tag_request_may_omit_model(self) -> None:
        response = MagicMock()
        response.read.return_value = json.dumps(
            {"choices": [{"message": {"content": "{}"}}], "usage": {}}
        ).encode()
        context = MagicMock()
        context.__enter__.return_value = response
        client = KnowledgeTagClient(
            "https://example.test/v1",
            "",
            "",
            timeout=10,
            retries=0,
            max_tokens=100,
        )
        with patch("urllib.request.urlopen", return_value=context) as urlopen:
            client.complete("system", "user", {})
        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertNotIn("model", payload)


class PreparationTests(unittest.TestCase):
    def test_primary_tag_subject_and_slot_are_deterministic(self) -> None:
        row = {
            "document_id": "doc-a",
            "language": "en",
            "tags": [
                {
                    "confidence": 0.8,
                    "discipline": "Low",
                    "field": "Low",
                    "subfield": "Low",
                    "evidence": "Example Person received 10 votes in the election.",
                },
                {
                    "confidence": 0.95,
                    "discipline": "Public Affairs",
                    "field": "Government",
                    "subfield": "Elections",
                    "evidence": "Example Person received 10 votes in the election tally.",
                },
            ],
        }
        entities = [
            {
                "text": "Example Person",
                "label": "person",
                "count": 2,
                "first_start": 0,
                "surface_forms": ["Example Person"],
            }
        ]
        anchors, reason = prepare_document(row, entities, PreparationConfig(min_evidence_chars=20))
        self.assertIsNone(reason)
        self.assertEqual({item["slot"] for item in anchors}, {"votes"})
        self.assertEqual(anchors[0]["subject"]["norm"], "example person")
        self.assertEqual(anchors[0]["knowledge_point"]["subfield"], "Elections")


class CandidatePipelineTests(unittest.TestCase):
    def test_relation_resume_stays_within_the_selected_candidate_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidates.jsonl"
            rows = [
                {
                    "pair_id": f"pair-{index}",
                    "doc_a": {"doc_id": f"a-{index}"},
                    "doc_b": {"doc_id": f"b-{index}"},
                }
                for index in range(3)
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            pending = list(iter_candidate_tasks(path, {"pair-0"}, max_pairs=2))
        self.assertEqual([row["pair_id"] for row in pending], ["pair-1"])

    def test_kp_and_cross_kp_routes_are_merged_and_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anchors_dir = root / "anchors"
            anchors_dir.mkdir()
            rows = [
                anchor(
                    "a",
                    "public/government/elections",
                    number="10",
                    tokens=["example", "person", "received", "votes"],
                    domain="a.test",
                ),
                anchor(
                    "b",
                    "public/government/elections",
                    number="12",
                    tokens=["example", "person", "received", "votes"],
                    domain="b.test",
                ),
                anchor(
                    "c",
                    "reference/people/biography",
                    number="11",
                    tokens=["example", "person", "election", "votes"],
                    domain="c.test",
                ),
            ]
            rows[2]["slot"] = "release"
            path = anchors_dir / "shard_00000000.anchors.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            output = root / "candidates.jsonl"
            report = build_candidates(
                anchors_dir,
                output,
                config=CandidateConfig(partitions=4),
                workers=2,
                work_dir=root / "work",
            )
            candidates = list(read_jsonl(output))
            by_pair = {(row["doc_a"]["doc_id"], row["doc_b"]["doc_id"]): row for row in candidates}
            self.assertIn(("a", "b"), by_pair)
            self.assertEqual(by_pair[("a", "b")]["retrieval_signals"], ["same_kp_subject_slot"])
            self.assertIn(("a", "c"), by_pair)
            self.assertEqual(
                by_pair[("a", "c")]["retrieval_signals"],
                ["tner_cross_kp_entity"],
            )
            self.assertEqual(report["merge"]["unique_pairs"], 3)
            self.assertEqual(report["retrieval"]["totals"]["unbounded_pairs"], 4)
            self.assertEqual(report["retrieval"]["totals"]["eligible_pairs_before_bounding"], 3)
            self.assertIn("full-document relation classification", report["next_stage"])

    def test_relation_task_preserves_pair_and_document_ids(self) -> None:
        task = compact_candidate(
            {
                "pair_id": "pair-1",
                "doc_a": {"doc_id": "a", "evidence": "Example received 10 votes."},
                "doc_b": {"doc_id": "b", "evidence": "Example received 12 votes."},
                "retrieval_signals": ["same_kp_subject_slot"],
                "subjects": [{"norm": "example"}],
                "slots": ["votes"],
            }
        )
        self.assertEqual(task["pair_id"], "pair-1")
        self.assertEqual((task["doc_a_id"], task["doc_b_id"]), ("a", "b"))
        self.assertEqual(task["metadata"]["slots"], ["votes"])

    def test_relation_schema_validation_and_endpoint(self) -> None:
        validate_relation(
            {
                "relation": "conflict",
                "same_subject": True,
                "same_attribute": True,
                "same_fact_context": True,
                "values_compatible": False,
                "subject": "Example",
                "attribute": "votes",
                "value_a": "10",
                "value_b": "12",
                "evidence_a": "received 10 votes",
                "evidence_b": "received 12 votes",
                "confidence": 0.9,
                "reason": "The same tally has incompatible values.",
            }
        )
        self.assertEqual(
            _endpoint_parts("https://example.test/qwen/v1"),
            ("https", "example.test", None, "/qwen/v1/chat/completions"),
        )

    def test_large_bucket_uses_bounded_top_k(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anchors_dir = root / "anchors"
            anchors_dir.mkdir()
            rows = [
                anchor(
                    f"d{index}",
                    "public/government/elections",
                    number=str(index),
                    tokens=["example", "person", "votes", f"token{index % 2}"],
                    domain=f"{index}.test",
                )
                for index in range(8)
            ]
            (anchors_dir / "shard_00000000.anchors.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            output = root / "candidates.jsonl"
            build_candidates(
                anchors_dir,
                output,
                config=CandidateConfig(
                    partitions=2,
                    all_pairs_max_docs=3,
                    support_top_k=1,
                    conflict_top_k=1,
                    candidate_pool_size=3,
                ),
                workers=2,
                work_dir=root / "work",
            )
            candidates = list(read_jsonl(output))
            self.assertGreater(len(candidates), 0)
            self.assertLess(len(candidates), 28)

    def test_known_pair_audit_reads_benchmark_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "candidates.jsonl"
            candidates.write_text(
                json.dumps(
                    {
                        "doc_a": {"doc_id": "a"},
                        "doc_b": {"doc_id": "b"},
                        "retrieval_signals": ["same_kp_subject_slot"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            benchmark = root / "benchmark.jsonl"
            benchmark.write_text(
                json.dumps(
                    {"provenance": {"conflict_edge": {"src_doc_id": "b", "dst_doc_id": "a"}}}
                )
                + "\n",
                encoding="utf-8",
            )
            report = evaluate_known_pair_recall(candidates, benchmark)
            self.assertEqual(report["known_pairs"], 1)
            self.assertEqual(report["recovered_pairs"], 1)
            self.assertEqual(report["recovered_by_signal"], {"same_kp_subject_slot": 1})


class PostClassificationPipelineTests(unittest.TestCase):
    def test_results_dir_controls_every_stage_default(self) -> None:
        parser = build_parser()
        root = Path("custom-results").resolve()
        commands = {
            "build-store": (
                ["build-store", "--input", "source.jsonl"],
                {"output": "documents.sqlite"},
            ),
            "tag-kp": (["tag-kp", "--input", "source.jsonl"], {"output_dir": "knowledge_tags"}),
            "tag-ner": (["tag-ner", "--input", "source.jsonl"], {"output_dir": "ner_tags"}),
            "prepare": (
                ["prepare"],
                {
                    "kp_dir": "knowledge_tags",
                    "ner_dir": "ner_tags",
                    "doc_store": "documents.sqlite",
                },
            ),
            "audit-preparation": (
                ["audit-preparation", "--known-pairs", "known.jsonl"],
                {
                    "kp_dir": "knowledge_tags",
                    "ner_dir": "ner_tags",
                    "doc_store": "documents.sqlite",
                },
            ),
            "classify": (
                ["classify"],
                {
                    "candidates": "pre_llm_candidates.jsonl",
                    "doc_store": "documents.sqlite",
                    "output": "relations.jsonl",
                },
            ),
            "subgraphs": (
                ["subgraphs"],
                {
                    "relations": ("relations.jsonl",),
                    "doc_store": "documents.sqlite",
                    "output": "conflict_subgraphs.jsonl",
                },
            ),
            "synthesize": (
                ["synthesize"],
                {"subgraphs": "conflict_subgraphs.jsonl", "output": "synthesized_qa.jsonl"},
            ),
            "validate-sources": (
                ["validate-sources"],
                {
                    "synthesis": "synthesized_qa.jsonl",
                    "subgraphs": "conflict_subgraphs.jsonl",
                    "output": "source_validation.jsonl",
                },
            ),
            "verify": (
                ["verify"],
                {
                    "synthesis": "synthesized_qa.jsonl",
                    "source_validation": "source_validation.jsonl",
                    "subgraphs": "conflict_subgraphs.jsonl",
                    "output": "web_verification.jsonl",
                    "cache_dir": "web_cache",
                },
            ),
            "export": (
                ["export"],
                {
                    "synthesis": "synthesized_qa.jsonl",
                    "source_validation": "source_validation.jsonl",
                    "verification": "web_verification.jsonl",
                    "output": "verified_benchmark.jsonl",
                },
            ),
        }
        for name, (command, expected) in commands.items():
            with self.subTest(command=name):
                args = parser.parse_args(command)
                resolve_result_paths(args, root)
                for attribute, relative in expected.items():
                    if isinstance(relative, tuple):
                        self.assertEqual(
                            getattr(args, attribute), [root / item for item in relative]
                        )
                    else:
                        self.assertEqual(getattr(args, attribute), root / relative)

    def test_packaged_construction_prompts_are_available(self) -> None:
        parser = build_parser()
        commands = (
            ["tag-kp", "--input", "input.jsonl"],
            ["classify", "--doc-store", "docs.sqlite"],
            ["synthesize"],
            ["validate-sources"],
            ["verify"],
        )
        for command in commands:
            with self.subTest(command=command[0]):
                prompt = parser.parse_args(command).prompt
                self.assertTrue(prompt.is_file(), prompt)

        tag_kp = parser.parse_args(["tag-kp", "--input", "input.jsonl"])
        self.assertTrue(tag_kp.taxonomy.is_file(), tag_kp.taxonomy)
        tag_ner = parser.parse_args(["tag-ner", "--input", "input.jsonl"])
        self.assertEqual(tag_ner.model_name, "tner/deberta-v3-large-ontonotes5")
        self.assertEqual(tag_ner.model_revision, ner_tagging.DEFAULT_MODEL_REVISION)
        classify = parser.parse_args(["classify"])
        self.assertIsNone(classify.top_k)
        self.assertIsNone(classify.min_p)
        self.assertIsNone(classify.repetition_penalty)
        self.assertFalse(classify.disable_thinking)
        candidates = parser.parse_args(["candidates"])
        self.assertEqual(candidates.pair_sample_rate, 1.0)

    def test_subgraphs_require_support_on_both_conflict_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relations = root / "relations.jsonl"
            rows = [
                {
                    "pair_id": "conflict",
                    "doc_a_id": "a",
                    "doc_b_id": "b",
                    "status": "success",
                    "relation": "conflict",
                    "judgment": {"confidence": 0.9},
                },
                {
                    "pair_id": "support-a",
                    "doc_a_id": "a",
                    "doc_b_id": "c",
                    "status": "success",
                    "relation": "support",
                    "judgment": {"confidence": 0.8},
                },
                {
                    "pair_id": "support-b",
                    "doc_a_id": "b",
                    "doc_b_id": "d",
                    "status": "success",
                    "relation": "support",
                    "judgment": {"confidence": 0.8},
                },
            ]
            relations.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            store = root / "docs.sqlite"
            connection = sqlite3.connect(store)
            connection.execute("CREATE TABLE docs(canonical_doc_id TEXT, url TEXT, text TEXT)")
            long_document = "head " + "x" * 60_000 + " FULL_SUBGRAPH_TAIL"
            connection.executemany(
                "INSERT INTO docs VALUES(?,?,?)",
                [
                    (
                        doc_id,
                        f"https://{doc_id}.test",
                        long_document if doc_id == "c" else f"document {doc_id}",
                    )
                    for doc_id in "abcd"
                ],
            )
            connection.commit()
            connection.close()
            output = root / "subgraphs.jsonl"
            report = build_subgraphs([relations], store, output)
            self.assertEqual(report["selected_subgraphs"], 1)
            graph = next(read_jsonl(output))
            self.assertEqual({doc["doc_id"] for doc in graph["documents"]}, set("abcd"))
            self.assertEqual(len(graph["support_edges"]), 2)
            document_c = next(doc for doc in graph["documents"] if doc["doc_id"] == "c")
            self.assertTrue(document_c["text"].endswith("FULL_SUBGRAPH_TAIL"))

    def test_relation_prompt_preserves_both_full_documents(self) -> None:
        marker_a = "A_MIDDLE_MARKER"
        marker_b = "B_MIDDLE_MARKER"
        prompt = render_relation_prompt(
            {"metadata": {}},
            "https://a.test",
            "head " + marker_a + " tail",
            "https://b.test",
            "head " + marker_b + " tail",
        )
        self.assertIn(marker_a, prompt)
        self.assertIn(marker_b, prompt)

    def test_synthesis_validation_checks_two_formulations_and_evidence(self) -> None:
        graph = {
            "documents": [
                {"doc_id": "a", "text": "The report gives 10 votes."},
                {"doc_id": "b", "text": "The report gives 12 votes."},
            ]
        }
        qa = {
            "keep": True,
            "rejection_reason": "",
            "subject": "Example",
            "attribute": "vote tally",
            "questions": [
                {
                    "formulation": "named_entity",
                    "question": "What vote tally was reported for Example?",
                },
                {
                    "formulation": "clue_based",
                    "question": (
                        "What vote tally was reported for the candidate in the example election?"
                    ),
                },
            ],
            "gold_answers": [
                {
                    "value": "10",
                    "supporting_doc_ids": ["a"],
                    "evidence_spans": [{"doc_id": "a", "quote": "The report gives 10 votes."}],
                },
                {
                    "value": "12",
                    "supporting_doc_ids": ["b"],
                    "evidence_spans": [{"doc_id": "b", "quote": "The report gives 12 votes."}],
                },
            ],
            "preferred_answer": "Reports give 10 and 12 votes.",
        }
        self.assertEqual(validate_qa(qa, graph), [])

    def test_full_document_validation_requires_every_answer_and_verbatim_evidence(self) -> None:
        qa = {"gold_answers": [{"value": "10"}, {"value": "12"}]}
        subgraph = {
            "documents": [
                {"doc_id": "a", "text": "The official count was 10 votes."},
                {"doc_id": "b", "text": "A corrected report listed 12 votes."},
            ]
        }
        result = {
            "keep": True,
            "verdict": "verified",
            "same_subject": True,
            "same_attribute": True,
            "same_fact_context": True,
            "qa_valid": True,
            "answer_verifications": [
                {
                    "value": "10",
                    "supported": True,
                    "evidence_spans": [
                        {"doc_id": "a", "quote": "The official count was 10 votes."}
                    ],
                    "reason": "Explicitly reported.",
                },
                {
                    "value": "12",
                    "supported": True,
                    "evidence_spans": [
                        {"doc_id": "b", "quote": "A corrected report listed 12 votes."}
                    ],
                    "reason": "Explicitly reported.",
                },
            ],
            "reason": "Both accounts are supported by the supplied documents.",
        }
        self.assertEqual(validate_source_result(result, qa, subgraph), [])
        result["answer_verifications"][1]["evidence_spans"][0]["quote"] = "12 votes"
        self.assertEqual(validate_source_result(result, qa, subgraph), [])
        result["answer_verifications"][1]["evidence_spans"][0]["quote"] = "thirteen votes"
        self.assertIn(
            "answer verification 1 evidence is not verbatim",
            validate_source_result(result, qa, subgraph),
        )

    def test_verification_request_omits_model_selection(self) -> None:
        response = MagicMock()
        response.read.return_value = json.dumps(
            {"choices": [{"message": {"role": "assistant", "content": "{}"}}]}
        ).encode()
        context = MagicMock()
        context.__enter__.return_value = response
        client = ToolEndpointClient(
            "https://example.test/v1", "", "", timeout=10, retries=0, max_tokens=100
        )
        with patch("urllib.request.urlopen", return_value=context) as urlopen:
            _, fields = client.complete([{"role": "user", "content": "test"}], tools_enabled=True)
        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertNotIn("model", payload)
        self.assertNotIn("model", fields)
        self.assertNotIn("chat_template_kwargs", payload)
        self.assertEqual(
            {tool["function"]["name"] for tool in payload["tools"]},
            {"WebSearch", "WebFetch"},
        )

    def test_synthesis_request_can_omit_model_selection(self) -> None:
        response = MagicMock()
        response.read.return_value = json.dumps(
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": '{"keep":false}'},
                        "finish_reason": "stop",
                    }
                ]
            }
        ).encode()
        context = MagicMock()
        context.__enter__.return_value = response
        client = OpenAIChatClient("https://example.test/v1", "", "", timeout=10, retries=0)
        with patch("urllib.request.urlopen", return_value=context) as urlopen:
            client.chat("system", "user", max_tokens=100)
        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertNotIn("model", payload)

    def test_synthesis_stage_accepts_an_endpoint_selected_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subgraphs = root / "subgraphs.jsonl"
            output = root / "synthesis.jsonl"
            prompt = root / "prompt.txt"
            subgraphs.write_text("", encoding="utf-8")
            prompt.write_text("Synthesize questions.", encoding="utf-8")
            args = build_parser().parse_args(
                [
                    "synthesize",
                    "--subgraphs",
                    str(subgraphs),
                    "--output",
                    str(output),
                    "--prompt",
                    str(prompt),
                    "--base-url",
                    "https://example.test/v1",
                    "--model",
                    "",
                ]
            )
            report = run_synthesis(args)
            self.assertEqual(report["input_subgraphs"], 0)
            self.assertFalse(report["request_includes_model_selection"])

    def test_downstream_loaders_use_the_latest_retry_for_each_subgraph(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            synthesis = root / "synthesis.jsonl"
            synthesis.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {"subgraph_id": "a", "status": "error"},
                        {
                            "subgraph_id": "a",
                            "status": "success",
                            "qa": {"keep": True},
                        },
                        {
                            "subgraph_id": "b",
                            "status": "success",
                            "qa": {"keep": False},
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate subgraph_id a"):
                load_source_validation_input(synthesis)
            selected = load_source_validation_input(synthesis, allow_retries=True)
            self.assertEqual(selected["a"]["status"], "success")
            rows, skipped = _load_synthesis(synthesis)
            self.assertEqual([row["subgraph_id"] for row in rows], ["a"])
            self.assertEqual(skipped, 1)

            validations = root / "validations.jsonl"
            validations.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {"subgraph_id": "a", "status": "error"},
                        {
                            "subgraph_id": "a",
                            "status": "success",
                            "validation": {"keep": True, "verdict": "verified"},
                        },
                        {
                            "subgraph_id": "b",
                            "status": "success",
                            "validation": {"keep": False, "verdict": "rejected"},
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            selected, skipped = _load_source_validations(validations)
            self.assertEqual(set(selected), {"a"})
            self.assertEqual(skipped, 1)

    def test_verification_request_can_select_model(self) -> None:
        response = MagicMock()
        response.read.return_value = json.dumps(
            {"choices": [{"message": {"role": "assistant", "content": "{}"}}]}
        ).encode()
        context = MagicMock()
        context.__enter__.return_value = response
        client = ToolEndpointClient(
            "https://example.test/v1",
            "",
            "verification-model",
            timeout=10,
            retries=0,
            max_tokens=100,
        )
        with patch("urllib.request.urlopen", return_value=context) as urlopen:
            _, fields = client.complete([{"role": "user", "content": "test"}], tools_enabled=False)
        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(payload["model"], "verification-model")
        self.assertIn("model", fields)

    def test_verification_requires_verbatim_fetched_evidence(self) -> None:
        qa = {"gold_answers": [{"value": "10"}, {"value": "12"}]}
        fetched = {
            "source-1": {
                "url": "https://external.test/one",
                "text": "The official result was 10 votes.",
            },
            "source-2": {
                "url": "https://external.test/two",
                "text": "A later report listed 12 votes.",
            },
        }
        result = {
            "keep": True,
            "verdict": "verified",
            "same_subject": True,
            "same_attribute": True,
            "same_fact_context": True,
            "qa_valid": True,
            "answer_verifications": [
                {
                    "value": "10",
                    "verified": True,
                    "sources": [{"source_id": "source-1", "quote": "official result was 10 votes"}],
                },
                {
                    "value": "12",
                    "verified": True,
                    "sources": [{"source_id": "source-2", "quote": "later report listed 12 votes"}],
                },
            ],
            "reason": "Both reports are independently available.",
        }
        self.assertEqual(validate_verification(result, qa, fetched, {"//seed.test/source"}), [])

        fetched["source-1"]["requested_url"] = "https://seed.test/source/"
        self.assertIn(
            "answer verification 0 reuses a seed document as independent evidence",
            validate_verification(result, qa, fetched, {"//seed.test/source"}),
        )

    def test_verification_receives_full_seed_documents(self) -> None:
        subgraph = {
            "seed_conflict": {"pair_id": "conflict"},
            "support_edges": [{"pair_id": "support"}],
            "documents": [
                {
                    "doc_id": "a",
                    "url": "https://seed.test/source",
                    "text": "head UNIQUE_SEED_MIDDLE tail",
                }
            ],
        }
        source_validation = {"keep": True, "verdict": "verified"}
        payload = _verification_input(
            {"gold_answers": [{"value": "10"}]}, subgraph, source_validation
        )
        self.assertIn("UNIQUE_SEED_MIDDLE", payload["seed_documents"][0]["text"])
        self.assertEqual(payload["source_validation"], source_validation)
        self.assertEqual(_seed_urls(subgraph), {"//seed.test/source"})

    def test_web_fetch_does_not_truncate_page_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            placeholder = "test-only-placeholder"
            web = WebTools(
                Path(directory),
                search_api_key=placeholder,
                allow_private_network=True,
            )
            body = ("<html><body>" + "A" * 40_000 + "END_MARKER</body></html>").encode()
            with patch.object(
                web, "_request", return_value=("https://example.test", body, "text/html")
            ):
                fetched = web.fetch("https://example.test")
        self.assertIn("END_MARKER", fetched["text"])
        self.assertNotIn("truncated", fetched)

    def test_web_cache_handles_concurrent_writes_to_the_same_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = WebCache(Path(directory))
            barrier = threading.Barrier(2)

            def produce() -> dict[str, str]:
                barrier.wait()
                return {"value": "cached"}

            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(
                    executor.map(lambda _: cache.get_or_create("fetch", "same", produce), range(2))
                )

            cached_files = list(Path(directory).glob("fetch_*.json"))
            self.assertEqual(len(cached_files), 1)
            self.assertEqual(json.loads(cached_files[0].read_text())["value"], "cached")
            self.assertEqual([result["value"] for result in results], ["cached", "cached"])

    def test_nonverbatim_exhaustion_becomes_insufficient_evidence(self) -> None:
        qa = {"gold_answers": [{"value": "10"}, {"value": "12"}]}
        result = _quotation_failure_result(qa)
        self.assertFalse(result["keep"])
        self.assertEqual(result["verdict"], "insufficient_evidence")
        self.assertEqual(validate_verification(result, qa, {}), [])

    def test_brave_search_parser_extracts_web_results(self) -> None:
        self.assertEqual(
            parse_brave_results(
                {
                    "web": {
                        "results": [
                            {
                                "title": "Example A",
                                "url": "https://example.test/a",
                                "description": "First snippet.",
                            }
                        ]
                    }
                }
            ),
            [
                {
                    "rank": 1,
                    "title": "Example A",
                    "url": "https://example.test/a",
                    "snippet": "First snippet.",
                }
            ],
        )

    def test_brave_search_uses_subscription_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            placeholder = "test-only-placeholder"
            web = WebTools(
                Path(directory),
                search_api_key=placeholder,
            )
            payload = json.dumps({"web": {"results": []}}).encode()
            with patch.object(
                web,
                "_request",
                return_value=(
                    "https://api.search.brave.com/res/v1/web/search",
                    payload,
                    "application/json",
                ),
            ) as request:
                result = web.search("example query")
        self.assertEqual(result["provider"], "brave")
        self.assertEqual(
            request.call_args.kwargs["extra_headers"]["X-Subscription-Token"],
            placeholder,
        )

    def test_export_writes_exactly_two_records_per_verified_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            synthesis = root / "synthesis.jsonl"
            source_validation = root / "source_validation.jsonl"
            verification = root / "verification.jsonl"
            output = root / "benchmark.jsonl"
            qa = {
                "keep": True,
                "questions": [
                    {"formulation": "named_entity", "question": "What was reported for Example?"},
                    {
                        "formulation": "clue_based",
                        "question": "What was reported for the candidate?",
                    },
                ],
                "gold_answers": [{"value": "10"}, {"value": "12"}],
                "preferred_answer": "Reports give 10 and 12.",
            }
            synthesis.write_text(
                json.dumps(
                    {"subgraph_id": "subgraph-pair-abcdef1234567890", "status": "success", "qa": qa}
                )
                + "\n",
                encoding="utf-8",
            )
            verification.write_text(
                json.dumps(
                    {
                        "subgraph_id": "subgraph-pair-abcdef1234567890",
                        "status": "success",
                        "verification": {"keep": True, "verdict": "verified"},
                        "request_fields": ["messages", "tools"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            source_validation.write_text(
                json.dumps(
                    {
                        "subgraph_id": "subgraph-pair-abcdef1234567890",
                        "status": "success",
                        "validation": {"keep": True, "verdict": "verified"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            report = export_records(synthesis, source_validation, verification, output)
            records = list(read_jsonl(output))
            self.assertEqual(report["questions_exported"], 2)
            self.assertTrue(all("difficulty" not in row["eval"] for row in records))
            self.assertEqual(len({row["item_group_id"] for row in records}), 1)
            self.assertTrue(all("scoring_rubric" not in row["eval"] for row in records))
            self.assertTrue(all(uuid.UUID(row["benchmark_id"]).version == 4 for row in records))
            self.assertTrue(all(uuid.UUID(row["item_group_id"]).version == 4 for row in records))


if __name__ == "__main__":
    unittest.main()
