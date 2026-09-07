"""Tests for the portable vector store (no models, no network)."""

from __future__ import annotations

import numpy as np
import pytest

from nrc_rag.index.vectors import VectorStore


def _vs(tmp_path):
    return VectorStore(tmp_path / "vectors.npz")


def _rows(n: int, d: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_upsert_query_and_ordering(tmp_path):
    vs = _vs(tmp_path)
    v = _rows(5)
    vs.upsert([f"D:p1:c{i}" for i in range(5)], ["D"] * 5, ["text"] * 5, v)
    assert vs.count() == 5
    hits = vs.query(v[2], 3)
    assert hits[0][0] == "D:p1:c2"
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)
    assert [s for _, s in hits] == sorted([s for _, s in hits], reverse=True)


def test_vectors_are_normalised_on_write(tmp_path):
    vs = _vs(tmp_path)
    raw = np.array([[3.0, 4.0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)  # norm 5
    vs.upsert(["A:p1:c1"], ["A"], ["text"], raw)
    assert float(np.linalg.norm(vs.matrix[0])) == pytest.approx(1.0, abs=1e-6)


def test_round_trip_through_disk_is_exact(tmp_path):
    vs = _vs(tmp_path)
    v = _rows(20)
    ids = [f"D{i % 3}:p1:c{i}" for i in range(20)]
    docs = [f"D{i % 3}" for i in range(20)]
    kinds = ["text" if i % 4 else "figure" for i in range(20)]
    vs.upsert(ids, docs, kinds, v)
    vs.save()

    again = VectorStore(tmp_path / "vectors.npz")
    assert again.ids == ids and again.doc_ids == docs and again.kinds == kinds
    assert np.array_equal(again.matrix, vs.matrix)


def test_filters_restrict_results(tmp_path):
    vs = _vs(tmp_path)
    v = _rows(9)
    ids = [f"D{i // 3}:p1:c{i}" for i in range(9)]
    docs = [f"D{i // 3}" for i in range(9)]
    kinds = ["figure" if i % 3 == 0 else "text" for i in range(9)]
    vs.upsert(ids, docs, kinds, v)

    assert all(c.startswith("D1") for c, _ in vs.query(v[0], 5, doc_ids={"D1"}))
    assert {c for c, _ in vs.query(v[0], 9, kinds={"figure"})} == {"D0:p1:c0", "D1:p1:c3", "D2:p1:c6"}
    assert vs.query(v[0], 5, doc_ids={"nope"}) == []


def test_upsert_replaces_without_duplicating(tmp_path):
    vs = _vs(tmp_path)
    v = _rows(3)
    vs.upsert(["a", "b", "c"], ["D"] * 3, ["text"] * 3, v)
    new = _rows(1, seed=7)
    vs.upsert(["b"], ["D"], ["figure"], new)
    assert vs.count() == 3
    assert vs.kinds[vs.ids.index("b")] == "figure"
    assert vs.query(new[0], 1)[0][0] == "b"


def test_delete_by_id_and_by_document(tmp_path):
    vs = _vs(tmp_path)
    v = _rows(6)
    vs.upsert([f"x{i}" for i in range(6)], ["A", "A", "A", "B", "B", "B"], ["text"] * 6, v)
    vs.delete_ids(["x0"])
    assert vs.count() == 5 and "x0" not in vs.ids
    vs.delete_doc("B")
    assert vs.count() == 2 and set(vs.doc_ids) == {"A"}
    assert vs.matrix.shape[0] == 2


def test_dimension_change_is_rejected(tmp_path):
    vs = _vs(tmp_path)
    vs.upsert(["a"], ["D"], ["text"], _rows(1, d=8))
    with pytest.raises(ValueError, match="dimension changed"):
        vs.upsert(["b"], ["D"], ["text"], _rows(1, d=16))


def test_empty_store_is_safe(tmp_path):
    vs = _vs(tmp_path)
    assert vs.count() == 0
    assert vs.query([0.1] * 8, 5) == []
    vs.delete_ids(["nothing"])
    vs.save()
    assert VectorStore(tmp_path / "vectors.npz").count() == 0


def test_query_more_than_available(tmp_path):
    vs = _vs(tmp_path)
    v = _rows(2)
    vs.upsert(["a", "b"], ["D", "D"], ["text", "text"], v)
    assert len(vs.query(v[0], 50)) == 2


def test_document_paths_resolve_when_index_was_built_elsewhere(tmp_path):
    """An index is portable: absolute paths recorded on the build machine will not
    exist on the host that restores it, so sources are re-found by accession number."""
    from nrc_rag.index.store import IndexStore

    data = tmp_path / "Data" / "2025"
    data.mkdir(parents=True)
    pdf = data / "ML25000A001.pdf"
    pdf.write_bytes(b"%PDF-1.4 not a real pdf")

    store = IndexStore(tmp_path / "index", data_dir=tmp_path / "Data")
    with store.conn:
        store.conn.execute(
            "INSERT INTO documents (doc_id, path, sha256) VALUES (?,?,?)",
            ("ML25000A001", "/mount/src/app/Data/2025/ML25000A001.pdf", "abc"),
        )
    assert store.list_documents()[0].path == str(pdf)

    # a path that does exist is left alone
    assert store.resolve_doc_path("ML25000A001", str(pdf)) == str(pdf)
    # an unknown document falls back to what was stored rather than raising
    assert store.resolve_doc_path("NOPE", "/gone/x.pdf") == "/gone/x.pdf"
