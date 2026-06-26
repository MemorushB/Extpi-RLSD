from tools.extpi_rlsd.build_frontier import eligible_frontier


def _strict_row(**overrides):
    row = {
        "p_plain": 0.5,
        "p_PI": 0.75,
        "qwen8b_pi_verified": True,
        "student_completion_truncated": False,
    }
    row.update(overrides)
    return row


def test_strict_frontier_accepts_verified_non_truncated_row():
    assert eligible_frontier(_strict_row()) is True


def test_strict_frontier_rejects_truncated_row():
    assert eligible_frontier(_strict_row(student_completion_truncated=True)) is False


def test_strict_frontier_rejects_missing_verified_field():
    row = _strict_row()
    row.pop("qwen8b_pi_verified")
    assert eligible_frontier(row) is False
