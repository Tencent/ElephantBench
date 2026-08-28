"""Streaming document I/O shared by the raw-corpus tagging stages."""

from __future__ import annotations

import contextlib
import glob
import hashlib
import io
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TextIO


def _zstandard() -> Any:
    try:
        import zstandard
    except ImportError as exc:
        raise RuntimeError(
            "reading or writing .zstd shards requires the 'construction' extra: "
            "pip install 'elephantbench[construction]'"
        ) from exc
    return zstandard


@contextlib.contextmanager
def open_text(path: Path, mode: str = "r") -> Iterator[TextIO]:
    """Open plain JSONL or concatenated-frame JSONL.zstd as UTF-8 text."""
    if mode not in {"r", "w", "a"}:
        raise ValueError(f"unsupported text mode: {mode}")
    if path.suffix != ".zstd":
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open(mode, encoding="utf-8", newline="") as handle:
            yield handle
        return

    zstandard = _zstandard()
    if mode == "r":
        raw = path.open("rb")
        stream = zstandard.ZstdDecompressor().stream_reader(raw, read_across_frames=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = path.open("ab" if mode == "a" else "wb")
        stream = zstandard.ZstdCompressor(level=3, threads=0).stream_writer(raw)
    text = io.TextIOWrapper(stream, encoding="utf-8", newline="")
    try:
        yield text
    finally:
        text.flush()
        text.close()


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with open_text(path, "r") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, value


def write_jsonl_row(handle: TextIO, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def extract_text(record: dict[str, Any]) -> str:
    source = record.get("source_record")
    if isinstance(source, dict) and isinstance(source.get("text"), str):
        return source["text"]
    for key in ("text", "content"):
        value = record.get(key)
        if isinstance(value, str):
            return value
    return ""


def extract_url(record: dict[str, Any]) -> str:
    source = record.get("source_record")
    if isinstance(source, dict) and isinstance(source.get("url"), str):
        return source["url"]
    value = record.get("url")
    return value if isinstance(value, str) else ""


def document_id(record: dict[str, Any]) -> str:
    existing = record.get("document_id") or record.get("doc_id")
    if isinstance(existing, str) and existing:
        return existing
    raw = f"{extract_url(record)}\n{extract_text(record)}"
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()


def expand_inputs(values: list[str]) -> list[Path]:
    """Resolve repeated files, directories, and shell-style globs deterministically."""
    selected: dict[str, Path] = {}
    for value in values:
        path = Path(value)
        if path.is_dir():
            matches = sorted(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file()
                and (candidate.name.endswith(".jsonl") or candidate.name.endswith(".jsonl.zstd"))
            )
        else:
            globbed = [Path(item) for item in sorted(glob.glob(value))]
            matches = globbed or ([path] if path.is_file() else [])
        if not matches:
            raise FileNotFoundError(f"input does not match a JSONL shard: {value}")
        for match in matches:
            selected[str(match.resolve())] = match.resolve()
    return [selected[key] for key in sorted(selected)]


def shard_output_name(input_path: Path, stage: str, compress: bool) -> str:
    name = input_path.name
    for suffix in (".jsonl.zstd", ".jsonl"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return f"{name}.{stage}.jsonl" + (".zstd" if compress else "")


def _scan_completed_document_ids(path: Path) -> tuple[set[str], bool]:
    completed: set[str] = set()
    truncated = False
    try:
        with open_text(path, "r") as handle:
            for line in handle:
                if not line.strip():
                    continue
                if not line.endswith("\n"):
                    truncated = True
                    break
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    truncated = True
                    break
                if not isinstance(row, dict):
                    raise ValueError(f"{path}: expected a JSON object")
                doc_id = str(row.get("document_id") or "")
                if doc_id and row.get("status") == "success":
                    completed.add(doc_id)
    except Exception:
        truncated = True
    return completed, truncated


def _repair_truncated_jsonl(path: Path) -> None:
    temporary = path.with_name(path.name + ".repairing" + path.suffix)
    if temporary.exists():
        temporary.unlink()
    try:
        with open_text(path, "r") as source, open_text(temporary, "w") as output:
            while True:
                try:
                    line = source.readline()
                except Exception:
                    break
                if not line:
                    break
                if not line.strip():
                    continue
                if not line.endswith("\n"):
                    break
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    break
                if not isinstance(row, dict):
                    break
                write_jsonl_row(output, row)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _has_incomplete_zstd_frame(path: Path) -> bool:
    zstandard = _zstandard()
    decompressor = None
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                pending = chunk
                while pending:
                    if decompressor is None:
                        decompressor = zstandard.ZstdDecompressor().decompressobj()
                    decompressor.decompress(pending)
                    if decompressor.eof:
                        pending = decompressor.unused_data
                        decompressor = None
                    else:
                        pending = b""
    except Exception:
        return True
    return decompressor is not None


def completed_document_ids(path: Path) -> set[str]:
    """Load successful IDs, repairing an interrupted final JSONL frame if necessary."""
    if not path.exists():
        return set()
    completed, truncated = _scan_completed_document_ids(path)
    if not truncated and path.suffix == ".zstd":
        truncated = _has_incomplete_zstd_frame(path)
    if truncated:
        _repair_truncated_jsonl(path)
    return completed


def selected_jsonl_records(
    path: Path, completed: set[str], limit: int
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield unfinished records within a stable first-N source selection."""
    selected = 0
    for line_number, record in iter_jsonl(path):
        if limit > 0 and selected >= limit:
            return
        selected += 1
        if document_id(record) in completed:
            continue
        yield line_number, record
