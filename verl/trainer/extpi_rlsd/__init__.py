"""ExtPI-RLSD helpers.

This package intentionally keeps the first MVP implementation narrow: the
math, prompt assembly, exact-token scoring helpers, and manifest collection
live here so they can be tested independently before deeper trainer wiring.
"""

from .rlsd import RLSDConfig, compute_opd_pg_advantages, compute_rlsd_advantages
from .scorer import (
    TeacherScoreOutput,
    TeacherScorer,
    TeacherScoreRequest,
    assert_tokenizer_compatible,
    build_causal_score_inputs,
    gather_response_log_probs,
)

__all__ = [
    "RLSDConfig",
    "TeacherScorer",
    "TeacherScoreRequest",
    "TeacherScoreOutput",
    "assert_tokenizer_compatible",
    "build_causal_score_inputs",
    "compute_opd_pg_advantages",
    "compute_rlsd_advantages",
    "gather_response_log_probs",
]
