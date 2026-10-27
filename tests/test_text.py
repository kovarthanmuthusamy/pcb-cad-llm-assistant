from pcb_assistant.text import final_answer_text, remove_think_blocks


def test_remove_think_blocks_drops_reasoning_only() -> None:
    raw = "<think>step one\nstep two</think>\nUse a 50 ohm trace.  "
    assert remove_think_blocks(raw) == "Use a 50 ohm trace."


def test_remove_think_blocks_handles_several_blocks() -> None:
    assert remove_think_blocks("<think>a</think>x<think>b</think>y") == "xy"


def test_final_answer_text_keeps_text_after_reasoning() -> None:
    assert final_answer_text("<think>hmm</think>\nThe answer.") == "The answer."


def test_final_answer_text_without_reasoning_is_unchanged() -> None:
    assert final_answer_text("Plain answer") == "Plain answer"


def test_final_answer_text_cuts_trailing_action_json() -> None:
    raw = 'Here is the result.\n{"action": "Final Answer", "action_input": "Here is the result."}'
    assert final_answer_text(raw) == "Here is the result."


def test_final_answer_text_keeps_json_in_the_middle() -> None:
    raw = 'Call {"action": "RAG", "action_input": "x"} first, then read the answer.'
    assert final_answer_text(raw) == raw
