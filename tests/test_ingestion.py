import io
import zipfile

import pytest

from chatlens import AnalysisConfig, IngestionError, read_chat


def test_reads_utf8_bytes():
    r = read_chat(b"[01/01/2024, 10:00:00] A: hi", filename="chat.txt")
    assert "hi" in r.text and r.replaced_chars == 0


def test_rejects_empty():
    with pytest.raises(IngestionError):
        read_chat(b"", filename="a.txt")


def test_rejects_oversize():
    cfg = AnalysisConfig(max_upload_mb=1)
    with pytest.raises(IngestionError, match="limit"):
        read_chat(b"a" * (1024 * 1024 + 1), filename="a.txt", config=cfg)


def test_rejects_wrong_extension():
    with pytest.raises(IngestionError, match=r"\.txt"):
        read_chat(b"hello", filename="malware.exe")


def test_rejects_binary():
    with pytest.raises(IngestionError, match="Binary"):
        read_chat(b"abc\x00\x01\x02def", filename="a.txt")


def _zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for n, d in files.items():
            z.writestr(n, d)
    return buf.getvalue()


def test_zip_with_single_txt():
    r = read_chat(_zip({"_chat.txt": b"[01/01/2024, 10:00:00] A: hi"}), filename="x.zip")
    assert "hi" in r.text


def test_zip_with_two_txt_rejected():
    with pytest.raises(IngestionError, match="exactly one"):
        read_chat(_zip({"a.txt": b"x", "b.txt": b"y"}), filename="x.zip")


def test_zip_bomb_rejected():
    cfg = AnalysisConfig(max_zip_uncompressed_mb=1)
    bomb = _zip({"_chat.txt": b"a" * (3 * 1024 * 1024)})
    assert len(bomb) < 100_000  # highly compressed
    with pytest.raises(IngestionError, match="uncompressed"):
        read_chat(bomb, filename="x.zip", config=cfg)


def test_zip_path_traversal_name_never_touches_disk(tmp_path):
    r = read_chat(_zip({"../../evil.txt": b"[01/01/2024, 10:00:00] A: hi"}), filename="x.zip")
    assert "hi" in r.text
    assert not (tmp_path.parent / "evil.txt").exists()


def test_cp1252_fallback():
    r = read_chat("[01/01/2024, 10:00:00] A: café".encode("cp1252"), filename="a.txt")
    assert "café" in r.text and r.encoding == "cp1252"
