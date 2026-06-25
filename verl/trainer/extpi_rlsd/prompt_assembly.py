"""Prompt assembly for ExtPI-RLSD recipes."""

from __future__ import annotations

from dataclasses import dataclass

STUDENT_USER_TEMPLATE = """Solve the following mathematics problem. Give a concise, self-contained derivation.
End with the final answer in \\boxed{{...}}.

{problem}"""

PI_TEACHER_USER_TEMPLATE = """Solve the following mathematics problem. Give a concise, self-contained derivation.
End with the final answer in \\boxed{{...}}.

{problem}

A verified reference solution from another solver is provided below. Use it only as
private guidance. Do not quote it or mention that a reference was provided.

<reference_solution>
{qwen8b_pi_trace}
</reference_solution>"""

CLOSED_TEACHER_USER_TEMPLATE = """Solve the following mathematics problem. Give a concise, self-contained derivation.
End with the final answer in \\boxed{{...}}.

Do not provide hidden chain-of-thought. Provide only visible reasoning that can be audited.

{problem}"""


@dataclass(frozen=True)
class PromptBundle:
    """Serialized student and scorer prompts for one problem."""

    student_prompt: str
    direct_opd_teacher_prompt: str
    pi_teacher_prompt: str | None


def _render_chat(tokenizer, user_text: str, *, enable_thinking: bool | None = False) -> str:
    messages = [{"role": "user", "content": user_text}]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def build_student_prompt(tokenizer, problem: str, *, enable_thinking: bool = False) -> str:
    """Build the PI-free student rollout prompt."""

    return _render_chat(tokenizer, STUDENT_USER_TEMPLATE.format(problem=problem), enable_thinking=enable_thinking)


def build_direct_opd_teacher_prompt(tokenizer, problem: str, *, enable_thinking: bool = False) -> str:
    """Build the direct OPD teacher prompt, identical to the student prompt."""

    return build_student_prompt(tokenizer, problem, enable_thinking=enable_thinking)


def build_pi_teacher_prompt(
    tokenizer,
    problem: str,
    qwen8b_pi_trace: str,
    *,
    enable_thinking: bool = False,
) -> str:
    """Build the scorer-only privileged prompt."""

    return _render_chat(
        tokenizer,
        PI_TEACHER_USER_TEMPLATE.format(problem=problem, qwen8b_pi_trace=qwen8b_pi_trace),
        enable_thinking=enable_thinking,
    )


def build_prompt_bundle(
    tokenizer,
    problem: str,
    qwen8b_pi_trace: str | None,
    *,
    student_thinking: bool = False,
    teacher_thinking: bool = False,
) -> PromptBundle:
    """Build all prompts needed by one ExtPI-RLSD sample."""

    student_prompt = build_student_prompt(tokenizer, problem, enable_thinking=student_thinking)
    direct_prompt = build_direct_opd_teacher_prompt(tokenizer, problem, enable_thinking=teacher_thinking)
    pi_prompt = None
    if qwen8b_pi_trace:
        pi_prompt = build_pi_teacher_prompt(
            tokenizer,
            problem,
            qwen8b_pi_trace,
            enable_thinking=teacher_thinking,
        )
    return PromptBundle(
        student_prompt=student_prompt,
        direct_opd_teacher_prompt=direct_prompt,
        pi_teacher_prompt=pi_prompt,
    )


def assert_no_pi_in_student_prompt(student_prompt: str, pi_trace: str | None) -> None:
    """Fail if privileged text leaks into the student rollout prompt."""

    if "<reference_solution>" in student_prompt or "</reference_solution>" in student_prompt:
        raise ValueError("Student prompt contains PI reference tags")
    if pi_trace and pi_trace.strip() and pi_trace.strip() in student_prompt:
        raise ValueError("Student prompt contains privileged trace text")
