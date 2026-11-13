"""Unit tests for SystemPromptProjection."""

from magitech_harness.agent.system_prompt import SystemPromptProjection


def _projection(nodes: list[dict]) -> SystemPromptProjection:
    return SystemPromptProjection(lambda: nodes)


def _node(seq: int, text: str | None) -> dict:
    return {"seq": seq, "text": text}


def test_first_prompt_reserves_node_zero_even_when_empty() -> None:
    commits = _projection([]).project("", in_history=False, starts_series=False)
    assert len(commits) == 1
    assert commits[0].message["content"] == []
    assert commits[0].intent == {"surface_op": "append"}
    commits = _projection([]).project("Prompt.", in_history=False, starts_series=False)
    assert commits[0].message["content"] == [{"type": "text", "text": "Prompt."}]


def test_unchanged_rendering_returns_no_commits() -> None:
    commits = _projection([_node(0, "Prompt.")]).project(
        "Prompt.", in_history=False, starts_series=False
    )
    assert commits == []


def test_changed_rendering_rewrites_head() -> None:
    commits = _projection([_node(0, "Old.")]).project(
        "New.", in_history=False, starts_series=False
    )
    assert len(commits) == 1
    assert commits[0].message["content"] == [{"type": "text", "text": "New."}]
    assert commits[0].intent == {
        "surface_op": {"op": "replace", "startSeq": 0, "endSeq": 0},
        "source_event_seqs": [0],
    }


def test_normalization_empties_later_nodes_before_rewriting_head() -> None:
    commits = _projection([_node(0, "Old."), _node(5, "Tail."), _node(9, "")]).project(
        "New.", in_history=False, starts_series=False
    )
    assert len(commits) == 2
    assert commits[0].message["content"] == []
    assert commits[0].intent["surface_op"] == {
        "op": "replace",
        "startSeq": 5,
        "endSeq": 5,
    }
    assert commits[0].intent["source_event_seqs"] == [5]
    assert commits[1].message["content"] == [{"type": "text", "text": "New."}]
    assert commits[1].intent["surface_op"]["startSeq"] == 0


def test_empty_rendering_clears_all_active_nodes() -> None:
    commits = _projection([_node(0, "Old."), _node(5, "Tail.")]).project(
        "", in_history=False, starts_series=False
    )
    assert len(commits) == 2
    assert all(commit.message["content"] == [] for commit in commits)
    assert [commit.intent["surface_op"]["startSeq"] for commit in commits] == [5, 0]


def test_empty_head_with_no_active_later_node_records_no_prompt() -> None:
    commits = _projection([_node(0, ""), _node(5, "")]).project(
        "", in_history=False, starts_series=False
    )
    assert commits == []


def test_in_history_continuing_series_appends_changed_text() -> None:
    commits = _projection([_node(0, "Old.")]).project(
        "New.", in_history=True, starts_series=False
    )
    assert len(commits) == 1
    assert commits[0].intent == {"surface_op": "append"}
    assert commits[0].message["content"] == [{"type": "text", "text": "New."}]


def test_in_history_series_start_with_latest_holding_rendering_is_noop() -> None:
    commits = _projection([_node(0, "Old."), _node(5, "New.")]).project(
        "New.", in_history=True, starts_series=True
    )
    assert commits == []


def test_in_history_series_start_with_different_latest_normalizes() -> None:
    commits = _projection([_node(0, "Old."), _node(5, "Tail.")]).project(
        "New.", in_history=True, starts_series=True
    )
    assert len(commits) == 2
    assert commits[0].intent["surface_op"]["startSeq"] == 5
    assert commits[1].intent["surface_op"]["startSeq"] == 0


def test_multi_block_node_has_no_comparable_text() -> None:
    commits = _projection([_node(0, None)]).project(
        "New.", in_history=False, starts_series=False
    )
    assert len(commits) == 1
    assert commits[0].message["content"] == [{"type": "text", "text": "New."}]


def test_latest_is_last_nonempty_node_not_head() -> None:
    commits = _projection([_node(0, ""), _node(5, "Tail.")]).project(
        "Tail.", in_history=True, starts_series=True
    )
    assert commits == []
    commits = _projection([_node(0, ""), _node(5, "Tail.")]).project(
        "Other.", in_history=True, starts_series=True
    )
    assert len(commits) == 2
