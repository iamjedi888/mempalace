"""Tests for the closet layer, mine_lock, entity metadata, BM25 hybrid search,
and diary ingest.

Content derived from Milla's omnibus test file; trimmed to only the features
present in this PR stack (#784 lock, #788 closets, this PR's entity/BM25/diary).
Strip-noise tests live with #785; tunnel tests live with the tunnels PR.
"""

import os
import tempfile
import threading
import time

from mempalace.palace import (
    CLOSET_CHAR_LIMIT,
    build_closet_lines,
    get_closets_collection,
    get_collection,
    mine_lock,
    upsert_closet_lines,
)
from mempalace.miner import _extract_entities_for_metadata
from mempalace.searcher import _bm25_score, _hybrid_rank


# ── mine_lock ────────────────────────────────────────────────────────────


class TestMineLock:
    def test_lock_acquires_and_releases(self):
        with mine_lock("/tmp/test_lock_file.txt"):
            lock_dir = os.path.expanduser("~/.mempalace/locks")
            assert os.path.isdir(lock_dir)

    def test_lock_blocks_concurrent_access(self):
        results = []

        def worker(name):
            start = time.time()
            with mine_lock("/tmp/same_file_lock_test.txt"):
                results.append((name, time.time() - start))
                time.sleep(0.2)

        t1 = threading.Thread(target=worker, args=("a",))
        t2 = threading.Thread(target=worker, args=("b",))
        t1.start()
        time.sleep(0.05)
        t2.start()
        t1.join()
        t2.join()

        # Second thread should have waited
        wait_times = sorted(results, key=lambda x: x[1])
        assert wait_times[1][1] > 0.1, "Second thread should block"


# ── closet lines ─────────────────────────────────────────────────────────


class TestBuildClosetLines:
    def test_returns_list_of_lines(self):
        lines = build_closet_lines(
            "/tmp/test.py", ["drawer_001"], "We built the auth system", "code", "general"
        )
        assert isinstance(lines, list)
        assert len(lines) >= 1

    def test_each_line_has_pointer(self):
        lines = build_closet_lines(
            "/tmp/test.py",
            ["drawer_001", "drawer_002"],
            "We built the auth system and tested the login flow",
            "code",
            "general",
        )
        for line in lines:
            assert "→" in line, f"Line missing pointer: {line}"

    def test_fallback_when_no_topics(self):
        lines = build_closet_lines(
            "/tmp/test.py", ["drawer_001"], "short text", "wing", "room"
        )
        assert len(lines) >= 1
        assert "→" in lines[0]


# ── upsert_closet_lines ─────────────────────────────────────────────────


class TestUpsertClosetLines:
    def test_writes_closets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            col = get_closets_collection(tmpdir)
            lines = [
                "topic one|Entity1|→drawer_001",
                "topic two|Entity2|→drawer_002",
            ]
            n = upsert_closet_lines(col, "test_closet", lines, {"wing": "test"})
            assert n >= 1
            assert col.count() >= 1

    def test_never_splits_mid_topic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            col = get_closets_collection(tmpdir)
            # Create lines that together exceed CLOSET_CHAR_LIMIT
            lines = [f"topic_{i}|{'x' * 200}|→drawer_{i}" for i in range(20)]
            n = upsert_closet_lines(col, "test_closet", lines, {"wing": "test"})
            assert n >= 2, "Should create multiple closets"

            # Verify each closet has complete lines
            all_data = col.get(include=["documents"])
            for doc in all_data["documents"]:
                for line in doc.strip().split("\n"):
                    assert "→" in line, f"Split topic found: {line}"

    def test_respects_char_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            col = get_closets_collection(tmpdir)
            lines = [f"topic_{i}|entities|→drawer_{i}" for i in range(50)]
            upsert_closet_lines(col, "test_closet", lines, {"wing": "test"})

            all_data = col.get(include=["documents"])
            for doc in all_data["documents"]:
                assert len(doc) <= CLOSET_CHAR_LIMIT + 100  # small buffer for existing content


# ── entity metadata ──────────────────────────────────────────────────────


class TestEntityMetadata:
    def test_extracts_capitalized_names(self):
        text = "Ben reviewed the code. Ben approved it. Igor flagged two issues. Igor fixed them."
        entities = _extract_entities_for_metadata(text)
        assert "Ben" in entities
        assert "Igor" in entities

    def test_empty_for_no_entities(self):
        text = "this is all lowercase with no proper nouns at all"
        entities = _extract_entities_for_metadata(text)
        assert entities == ""

    def test_semicolon_separated(self):
        text = "Alice and Bob met Charlie. Alice said hello. Bob agreed. Charlie laughed."
        entities = _extract_entities_for_metadata(text)
        assert ";" in entities


# ── BM25 hybrid search ──────────────────────────────────────────────────


class TestBM25:
    def test_bm25_score_positive_for_match(self):
        score = _bm25_score("database migration", "We migrated the database to Postgres")
        assert score > 0

    def test_bm25_score_zero_for_no_match(self):
        score = _bm25_score("quantum physics", "We built a web application in React")
        assert score == 0.0

    def test_hybrid_rank_reorders(self):
        results = [
            {"text": "database schema design for Postgres", "distance": 0.5},
            {"text": "unrelated topic about cooking", "distance": 0.3},
        ]
        ranked = _hybrid_rank(results, "database Postgres schema")
        # The database result should rank higher despite worse vector distance
        assert "database" in ranked[0]["text"]


# ── diary ingest ─────────────────────────────────────────────────────────


class TestDiaryIngest:
    def test_ingest_creates_drawers_and_closets(self):
        with tempfile.TemporaryDirectory() as palace_dir:
            diary_dir = tempfile.mkdtemp()
            # Write a test diary
            with open(os.path.join(diary_dir, "2026-04-13.md"), "w") as f:
                f.write("# 2026-04-13\n\n## 10:00 PDT — Test\n\nBuilt the auth system.\n")

            from mempalace.diary_ingest import ingest_diaries

            result = ingest_diaries(diary_dir, palace_dir, force=True)
            assert result["days_updated"] >= 1

            # Check drawer exists
            drawers = get_collection(palace_dir)
            count = drawers.count()
            assert count >= 1

    def test_ingest_skips_unchanged(self):
        with tempfile.TemporaryDirectory() as palace_dir:
            diary_dir = tempfile.mkdtemp()
            with open(os.path.join(diary_dir, "2026-04-13.md"), "w") as f:
                f.write("# 2026-04-13\n\n## 10:00 — Test\n\nContent.\n")

            from mempalace.diary_ingest import ingest_diaries

            ingest_diaries(diary_dir, palace_dir, force=True)
            result = ingest_diaries(diary_dir, palace_dir)  # second run, no force
            assert result["days_updated"] == 0
