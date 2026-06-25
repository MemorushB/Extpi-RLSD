from verl.trainer.extpi_rlsd.prompt_assembly import assert_no_pi_in_student_prompt, build_prompt_bundle


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        assert tokenize is False
        suffix = "<think>" if enable_thinking else ""
        return messages[0]["content"] + suffix + ("<assistant>" if add_generation_prompt else "")


def test_pi_trace_does_not_enter_student_prompt():
    tokenizer = FakeTokenizer()
    bundle = build_prompt_bundle(tokenizer, "What is 1+1?", "A privileged trace says 2.")

    assert "privileged trace" not in bundle.student_prompt
    assert "privileged trace" in bundle.pi_teacher_prompt
    assert_no_pi_in_student_prompt(bundle.student_prompt, "A privileged trace says 2.")


def test_privacy_assertion_fails_on_reference_tag():
    try:
        assert_no_pi_in_student_prompt("hello <reference_solution>x</reference_solution>", "x")
    except ValueError:
        return
    raise AssertionError("Expected PI leak assertion to fail")
