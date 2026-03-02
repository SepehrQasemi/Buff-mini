from __future__ import annotations

from buffmini.alpha_v2.search_v2 import SearchConfigV2, bounded_search


def test_stage21_search_bounded_evaluations() -> None:
    space = [{"x": i} for i in range(100)]

    def evaluate(candidate: dict) -> dict:
        return {"score": float(candidate["x"]), "valid": True, "reason": "VALID"}

    out = bounded_search(candidate_space=space, evaluate_fn=evaluate, cfg=SearchConfigV2(max_evaluations=10, beam_width=5, seed=42))
    assert int(out["evaluated_count"]) == 10
    assert len(out["top_candidates"]) <= 5


def test_stage21_search_deterministic_order() -> None:
    space = [{"x": i} for i in range(50)]

    def evaluate(candidate: dict) -> dict:
        return {"score": float(candidate["x"] % 7), "valid": True, "reason": "VALID"}

    cfg = SearchConfigV2(max_evaluations=30, beam_width=8, seed=7)
    a = bounded_search(candidate_space=space, evaluate_fn=evaluate, cfg=cfg)
    b = bounded_search(candidate_space=space, evaluate_fn=evaluate, cfg=cfg)
    assert a["top_candidates"] == b["top_candidates"]


def test_stage21_search_pruning_reasons_collected() -> None:
    space = [{"x": i} for i in range(20)]

    def evaluate(candidate: dict) -> dict:
        if candidate["x"] % 2 == 0:
            return {"score": -1.0, "valid": False, "reason": "EVEN_PRUNED"}
        return {"score": float(candidate["x"]), "valid": True, "reason": "VALID"}

    out = bounded_search(candidate_space=space, evaluate_fn=evaluate, cfg=SearchConfigV2(max_evaluations=20, beam_width=5, seed=1))
    assert int(out["pruned_count"]) > 0
    assert int(out["prune_reasons"].get("EVEN_PRUNED", 0)) > 0

