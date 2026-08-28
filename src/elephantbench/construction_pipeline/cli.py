"""Command-line interface for the paper-aligned pre-LLM construction stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import audit_preparation_filters
from .candidates import CandidateConfig, audit_known_pair_blocking, build_candidates
from .document_store import add_document_store_arguments, run_document_store
from .export import add_export_arguments, run_export
from .knowledge_tagging import add_knowledge_tagging_arguments, run_knowledge_tagging
from .ner_tagging import add_ner_tagging_arguments, run_ner_tagging
from .prepare import PreparationConfig, prepare_anchors
from .relations import add_relation_arguments, run_relation_classification
from .source_validation import add_source_validation_arguments, run_source_validation
from .subgraphs import add_subgraph_arguments, run_subgraph_build
from .synthesis import add_synthesis_arguments, run_synthesis
from .verification import add_verification_arguments, run_verification

DEFAULT_RESULTS = Path("outputs/construction")


RESULT_PATH_DEFAULTS: dict[str, dict[str, str | tuple[str, ...]]] = {
    "build-store": {"output": "documents.sqlite"},
    "tag-kp": {"output_dir": "knowledge_tags"},
    "tag-ner": {"output_dir": "ner_tags"},
    "prepare": {
        "kp_dir": "knowledge_tags",
        "ner_dir": "ner_tags",
        "doc_store": "documents.sqlite",
    },
    "all": {
        "kp_dir": "knowledge_tags",
        "ner_dir": "ner_tags",
        "doc_store": "documents.sqlite",
    },
    "audit-preparation": {
        "kp_dir": "knowledge_tags",
        "ner_dir": "ner_tags",
        "doc_store": "documents.sqlite",
    },
    "classify": {
        "candidates": "pre_llm_candidates.jsonl",
        "doc_store": "documents.sqlite",
        "output": "relations.jsonl",
    },
    "subgraphs": {
        "relations": ("relations.jsonl",),
        "doc_store": "documents.sqlite",
        "output": "conflict_subgraphs.jsonl",
    },
    "synthesize": {
        "subgraphs": "conflict_subgraphs.jsonl",
        "output": "synthesized_qa.jsonl",
    },
    "validate-sources": {
        "synthesis": "synthesized_qa.jsonl",
        "subgraphs": "conflict_subgraphs.jsonl",
        "output": "source_validation.jsonl",
    },
    "verify": {
        "synthesis": "synthesized_qa.jsonl",
        "source_validation": "source_validation.jsonl",
        "subgraphs": "conflict_subgraphs.jsonl",
        "output": "web_verification.jsonl",
        "cache_dir": "web_cache",
    },
    "export": {
        "synthesis": "synthesized_qa.jsonl",
        "source_validation": "source_validation.jsonl",
        "verification": "web_verification.jsonl",
        "output": "verified_benchmark.jsonl",
    },
}


def resolve_result_paths(args: argparse.Namespace, results_dir: Path) -> None:
    """Apply one output root consistently across every construction stage."""
    for attribute, relative in RESULT_PATH_DEFAULTS.get(args.command, {}).items():
        if getattr(args, attribute) is None:
            if isinstance(relative, tuple):
                setattr(args, attribute, [results_dir / item for item in relative])
            else:
                setattr(args, attribute, results_dir / relative)


def _preparation_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--kp-dir", type=Path, help="D_low SuperGPQA++ JSONL shards")
    parser.add_argument("--ner-dir", type=Path, help="D_low T-NER JSONL shards")
    parser.add_argument(
        "--doc-store",
        type=Path,
        help="optional SQLite doc_index/docs store used only to recover source domains",
    )
    parser.add_argument("--min-confidence", type=float, default=0.9)
    parser.add_argument("--min-evidence-chars", type=int, default=60)
    parser.add_argument("--max-subjects", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--language", default="en")
    parser.add_argument("--require-knowledge-bearing", action="store_true")
    parser.add_argument("--subject-aliases", action="store_true")
    parser.add_argument("--max-shards", type=int, default=0, help="0 uses every matched shard")


def _candidate_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--partitions", type=int, default=256)
    parser.add_argument("--all-pairs-max-docs", type=int, default=20)
    parser.add_argument(
        "--support-top-k",
        type=int,
        default=0,
        help="support-like neighbors retained per document in large buckets (default: 0)",
    )
    parser.add_argument(
        "--conflict-top-k",
        type=int,
        default=1,
        help="conflict-like neighbors retained per document in large buckets (default: 1)",
    )
    parser.add_argument("--rare-tokens-per-doc", type=int, default=8)
    parser.add_argument("--max-token-posting", type=int, default=200)
    parser.add_argument("--candidate-pool-size", type=int, default=200)
    parser.add_argument("--max-bucket-docs", type=int, default=5000)
    parser.add_argument("--bucket-overlap-docs", type=int, default=64)
    parser.add_argument("--max-cross-cluster-docs", type=int, default=200000)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.95)
    parser.add_argument(
        "--known-pairs", type=Path, help="optional verified pair JSONL for recall audit"
    )
    parser.add_argument(
        "--rebuild-retrieval",
        action="store_true",
        help="reuse anchor partitions but rebuild retrieval and all downstream files",
    )
    parser.add_argument(
        "--rebuild-pair-partitions",
        action="store_true",
        help="reuse retrieved pairs but rebuild sampling, deduplication, and merged output",
    )
    parser.add_argument(
        "--pair-sample-rate",
        type=float,
        default=1.0,
        help=(
            "deterministic pair-ID sampling rate applied before global deduplication "
            "(default: 1.0, retaining all bounded candidates)"
        ),
    )


def _prepare_config(args: argparse.Namespace) -> PreparationConfig:
    return PreparationConfig(
        min_confidence=args.min_confidence,
        min_evidence_chars=args.min_evidence_chars,
        max_subjects=args.max_subjects,
        max_tokens=args.max_tokens,
        language=args.language,
        require_knowledge_bearing=args.require_knowledge_bearing,
        subject_aliases=args.subject_aliases,
    )


def _candidate_config(args: argparse.Namespace) -> CandidateConfig:
    return CandidateConfig(
        partitions=args.partitions,
        all_pairs_max_docs=args.all_pairs_max_docs,
        support_top_k=args.support_top_k,
        conflict_top_k=args.conflict_top_k,
        rare_tokens_per_doc=args.rare_tokens_per_doc,
        max_token_posting=args.max_token_posting,
        candidate_pool_size=args.candidate_pool_size,
        max_bucket_docs=args.max_bucket_docs,
        bucket_overlap_docs=args.bucket_overlap_docs,
        max_cross_cluster_docs=args.max_cross_cluster_docs,
        near_duplicate_threshold=args.near_duplicate_threshold,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Construct ElephantBench from document signals through verified paired questions"
        )
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_store = subparsers.add_parser(
        "build-store", help="build the full-document SQLite store from raw shards"
    )
    add_document_store_arguments(build_store)

    tag_kp = subparsers.add_parser(
        "tag-kp", help="tag raw documents with the bundled SuperGPQA++ taxonomy"
    )
    add_knowledge_tagging_arguments(tag_kp)

    tag_ner = subparsers.add_parser("tag-ner", help="extract T-NER entities from raw documents")
    add_ner_tagging_arguments(tag_ner)

    prepare = subparsers.add_parser(
        "prepare", help="join KP/T-NER shards into subject-slot anchors"
    )
    _preparation_options(prepare)

    candidates = subparsers.add_parser(
        "candidates", help="retrieve, cap, and deduplicate pre-LLM document pairs"
    )
    candidates.add_argument("--anchors-dir", type=Path)
    _candidate_options(candidates)

    run_all = subparsers.add_parser("all", help="run preparation and candidate retrieval")
    _preparation_options(run_all)
    _candidate_options(run_all)

    inspect = subparsers.add_parser("inspect", help="print the latest count-only report")
    inspect.add_argument("--report", type=Path)

    audit = subparsers.add_parser(
        "audit-blocking",
        help="measure known-pair recall before Top-K retrieval or model calls",
    )
    audit.add_argument("--known-pairs", type=Path, required=True)
    audit.add_argument("--partitions", type=int, default=256)

    audit_prepare = subparsers.add_parser(
        "audit-preparation",
        help="measure known-pair recall after subject-slot preparation filters",
    )
    _preparation_options(audit_prepare)
    audit_prepare.add_argument("--known-pairs", type=Path, required=True)

    classify = subparsers.add_parser(
        "classify",
        help="classify full-document pairs as support, conflict, or none",
    )
    add_relation_arguments(classify)

    subgraphs = subparsers.add_parser(
        "subgraphs",
        help="expand conflict relations with support neighbors",
    )
    add_subgraph_arguments(subgraphs)

    synthesize = subparsers.add_parser(
        "synthesize",
        help="generate matched named-entity and clue-based questions",
    )
    add_synthesis_arguments(synthesize)

    validate_sources = subparsers.add_parser(
        "validate-sources",
        help="validate every synthesized answer against the full source documents",
    )
    add_source_validation_arguments(validate_sources)

    verify = subparsers.add_parser(
        "verify",
        help="verify every synthesized answer with bounded public-web tools",
    )
    add_verification_arguments(verify)

    export = subparsers.add_parser(
        "export",
        help="export every verified conflict group as two benchmark records",
    )
    add_export_arguments(export)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results_dir = args.results_dir.resolve()
    resolve_result_paths(args, results_dir)
    anchors_dir = results_dir / "prepared_anchors"
    output = results_dir / "pre_llm_candidates.jsonl"
    if args.command == "inspect":
        report_path = (args.report or results_dir / "candidate_report.json").resolve()
        print(report_path.read_text(encoding="utf-8"), end="")
        return 0
    if args.command == "audit-blocking":
        report = audit_known_pair_blocking(
            results_dir / "candidate_work" / "anchor_partitions",
            args.known_pairs,
            results_dir / "blocking_audit.json",
            partitions=args.partitions,
            workers=args.workers,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.command == "audit-preparation":
        report = audit_preparation_filters(
            args.kp_dir,
            args.ner_dir,
            args.known_pairs,
            results_dir / "preparation_audit.json",
            config=_prepare_config(args),
            workers=args.workers,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.command == "tag-kp":
        report = run_knowledge_tagging(args)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0 if int(report.get("error") or 0) == 0 else 1
    if args.command == "build-store":
        report = run_document_store(args)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.command == "tag-ner":
        report = run_ner_tagging(args)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.command == "classify":
        report = run_relation_classification(args)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.command == "subgraphs":
        report = run_subgraph_build(args)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.command == "synthesize":
        report = run_synthesis(args)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0 if int(report.get("error") or 0) == 0 else 1
    if args.command == "validate-sources":
        report = run_source_validation(args)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0 if int(report.get("error") or 0) == 0 else 1
    if args.command == "verify":
        report = run_verification(args)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0 if int(report.get("error") or 0) == 0 else 1
    if args.command == "export":
        report = run_export(args)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.command in {"prepare", "all"}:
        report = prepare_anchors(
            args.kp_dir,
            args.ner_dir,
            anchors_dir,
            config=_prepare_config(args),
            workers=args.workers,
            doc_store=args.doc_store,
            overwrite=args.overwrite,
            max_shards=args.max_shards,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.command in {"candidates", "all"}:
        selected_anchors = (getattr(args, "anchors_dir", None) or anchors_dir).resolve()
        report = build_candidates(
            selected_anchors,
            output,
            config=_candidate_config(args),
            workers=args.workers,
            work_dir=results_dir / "candidate_work",
            known_pairs=args.known_pairs,
            overwrite=args.overwrite,
            rebuild_retrieval=args.rebuild_retrieval,
            rebuild_pair_partitions=args.rebuild_pair_partitions,
            pair_sample_rate=args.pair_sample_rate,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
