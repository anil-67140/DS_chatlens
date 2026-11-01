"""Safe ingestion of uploaded WhatsApp exports (.txt or .zip containing one .txt).

Security notes: archives are read in memory only (no extraction to disk, so no path traversal),
uncompressed size is capped (zip-bomb guard), and files containing NUL bytes are rejected.
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .config import AnalysisConfig
from .exceptions import IngestionError

_ENCODINGS = ("utf-8-sig", "cp1252")


@dataclass(frozen=True)
class IngestResult:
    text: str
    encoding: str
    source_name: str
    n_bytes: int
    replaced_chars: int


def _decode(raw: bytes) -> tuple[str, str, int]:
    if b"\x00" in raw[:4096] and not raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        raise IngestionError("Binary content detected (NUL bytes); not a text export.")
    # UTF-16 is only trusted when a BOM is present: without one, almost any even-length byte
    # string "decodes" as UTF-16 and silently produces garbage.
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return raw.decode("utf-16"), "utf-16", 0
        except UnicodeError:
            pass
    for enc in _ENCODINGS:
        try:
            return raw.decode(enc), enc, 0
        except UnicodeError:
            continue
    text = raw.decode("utf-8", errors="replace")
    return text, "utf-8 (lossy)", text.count("\ufffd")


def read_chat(source: str | Path | bytes, filename: str | None = None,
              config: AnalysisConfig | None = None) -> IngestResult:
    """Read and validate an export from a path or raw bytes."""
    cfg = config or AnalysisConfig()
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise IngestionError(f"File not found: {path.name}")
        if path.stat().st_size > cfg.max_upload_mb * 1024 * 1024:
            raise IngestionError(f"File exceeds the {cfg.max_upload_mb} MB limit.")
        raw, name = path.read_bytes(), path.name
    else:
        raw, name = source, filename or "upload"
        if len(raw) > cfg.max_upload_mb * 1024 * 1024:
            raise IngestionError(f"File exceeds the {cfg.max_upload_mb} MB limit.")
    if not raw:
        raise IngestionError("The file is empty.")

    if raw[:4] == b"PK\x03\x04":
        raw = _read_zip(raw, cfg)
    elif not name.lower().endswith(".txt") and filename is not None:
        raise IngestionError("Only .txt exports or .zip archives containing one are accepted.")

    text, enc, replaced = _decode(raw)
    return IngestResult(text=text, encoding=enc, source_name=name, n_bytes=len(raw),
                        replaced_chars=replaced)


def _read_zip(raw: bytes, cfg: AnalysisConfig) -> bytes:
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as exc:
        raise IngestionError("Corrupt zip archive.") from exc
    txt = [i for i in zf.infolist() if i.filename.lower().endswith(".txt") and not i.is_dir()]
    if len(txt) != 1:
        raise IngestionError(f"Archive must contain exactly one .txt file (found {len(txt)}).")
    info = txt[0]
    if info.file_size > cfg.max_zip_uncompressed_mb * 1024 * 1024:
        raise IngestionError("Archive content exceeds the uncompressed size limit.")
    with zf.open(info) as fh:
        data = fh.read(cfg.max_zip_uncompressed_mb * 1024 * 1024 + 1)
    if len(data) > cfg.max_zip_uncompressed_mb * 1024 * 1024:
        raise IngestionError("Archive content exceeds the uncompressed size limit.")
    return data
