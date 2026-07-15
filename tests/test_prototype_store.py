import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.prototype_store import PrototypeStore


def _unit_vec(v):
    v = np.array(v, dtype=float)
    return v / np.linalg.norm(v)


def test_add_class_builds_mean_prototype():
    store = PrototypeStore()
    embeddings = [_unit_vec([1, 0, 0]), _unit_vec([1, 0.1, 0])]
    store.add_class("widget", embeddings)
    assert "widget" in store.prototypes
    assert store.reference_counts["widget"] == 2


def test_add_class_rejects_empty_list():
    store = PrototypeStore()
    try:
        store.add_class("widget", [])
        assert False, "should have raised"
    except ValueError:
        pass


def test_predict_picks_nearest_prototype():
    store = PrototypeStore()
    store.add_class("class_a", [_unit_vec([1, 0, 0])])
    store.add_class("class_b", [_unit_vec([0, 1, 0])])

    # Query close to class_a's prototype
    query = _unit_vec([0.9, 0.1, 0])
    result = store.predict(query)
    assert result["predicted_class"] == "class_a"
    assert result["confidence"] > 0.8


def test_predict_no_classes_registered():
    store = PrototypeStore()
    result = store.predict(_unit_vec([1, 0, 0]))
    assert "error" in result


def test_predict_top_k_ordering():
    store = PrototypeStore()
    store.add_class("class_a", [_unit_vec([1, 0, 0])])
    store.add_class("class_b", [_unit_vec([0.9, 0.1, 0])])
    store.add_class("class_c", [_unit_vec([0, 0, 1])])

    query = _unit_vec([1, 0, 0])
    result = store.predict(query, top_k=3)
    scores = [item["score"] for item in result["top_k"]]
    assert scores == sorted(scores, reverse=True)  # descending order


def test_remove_class():
    store = PrototypeStore()
    store.add_class("widget", [_unit_vec([1, 0, 0])])
    assert store.remove_class("widget") is True
    assert "widget" not in store.prototypes
    assert store.remove_class("widget") is False  # already removed


def test_save_and_load_roundtrip(tmp_path):
    store = PrototypeStore()
    store.add_class("widget", [_unit_vec([1, 0, 0]), _unit_vec([1, 0.2, 0])])

    path = str(tmp_path / "protos.json")
    store.save(path)

    loaded = PrototypeStore()
    loaded.load(path)

    assert set(loaded.prototypes.keys()) == {"widget"}
    assert loaded.reference_counts["widget"] == 2
    np.testing.assert_allclose(loaded.prototypes["widget"], store.prototypes["widget"], atol=1e-6)


def test_list_classes():
    store = PrototypeStore()
    store.add_class("a", [_unit_vec([1, 0, 0])])
    store.add_class("b", [_unit_vec([0, 1, 0]), _unit_vec([0, 0.9, 0.1])])
    listing = store.list_classes()
    assert listing["a"]["reference_images"] == 1
    assert listing["b"]["reference_images"] == 2
