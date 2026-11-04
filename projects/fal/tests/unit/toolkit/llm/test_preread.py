from __future__ import annotations

import os

from fal.toolkit.llm import preread_weights


def test_preread_counts_all_bytes_across_chunks(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(os.urandom(1000))
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.safetensors").write_bytes(os.urandom(2500))
    (tmp_path / "config.json").write_text("{}")

    stats = preread_weights(tmp_path, workers=4, chunk_bytes=300)

    assert stats.files == 2
    assert stats.bytes == 3500
    assert stats.gb_per_s >= 0


def test_preread_single_file_and_no_match(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"x" * 10)
    assert preread_weights(f).bytes == 10
    assert preread_weights(tmp_path, patterns=("*.nomatch",)).files == 0
