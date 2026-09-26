import json

import torch

from tacs.data_selection.select_from_scores import select_indices, write_metadata


def test_select_indices_top_k():
    scores = torch.tensor([0.2, 0.9, 0.4])
    index_map = [("a", 0), ("b", 0), ("a", 1)]
    selected = select_indices(scores, index_map, top_k=2)
    assert selected[0][0:2] == ("b", 0)
    assert selected[1][0:2] == ("a", 1)
    assert abs(selected[0][2] - 0.9) < 1e-6
    assert abs(selected[1][2] - 0.4) < 1e-6


def test_select_indices_rejects_ambiguous_budget():
    try:
        select_indices(torch.tensor([1.0]), [("a", 0)], top_k=1, top_pct=0.5)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_write_metadata(tmp_path):
    out = tmp_path / "meta.json"
    write_metadata(str(out), [("src", 3, 0.75)])
    assert json.loads(out.read_text()) == [{"source": "src", "index": 3, "score": 0.75}]
