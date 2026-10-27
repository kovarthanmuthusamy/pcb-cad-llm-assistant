from pathlib import Path

import pytest

from pcb_assistant.cli import (
    build_agent_input,
    chat_loop,
    collect_image_paths,
    describe_images,
    parse_args,
)
from pcb_assistant.llm import BASE_INSTRUCTION, PROFILES, ProviderProfile, build_llm


class FakeAssistant:
    def __init__(self, profile: ProviderProfile, replies: list[str] | None = None) -> None:
        self.profile = profile
        self.inputs: list[str] = []
        self.replies = replies if replies is not None else ["the answer"]

    def last_message(self) -> str:
        return "earlier message"

    def ask(self, text: str) -> str:
        self.inputs.append(text)
        reply = self.replies.pop(0)
        if reply == "RAISE":
            raise RuntimeError("model unreachable")
        return reply


class Script:
    """Scripted ``input()`` replacement that also records the prompts."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)


def test_provider_profiles_preserve_the_per_provider_behaviour() -> None:
    assert set(PROFILES) == {"ollama", "openai", "google"}
    assert [PROFILES[name].max_iterations for name in ("ollama", "openai", "google")] == [4, 4, 6]
    assert PROFILES["openai"].include_history_in_input
    assert not PROFILES["ollama"].include_history_in_input
    assert PROFILES["ollama"].system_instruction == BASE_INSTRUCTION
    assert PROFILES["openai"].system_instruction.startswith(BASE_INSTRUCTION + ",dont miss")


def test_build_llm_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        build_llm("mistral-cloud")


def test_parse_args_defaults_and_validation() -> None:
    args = parse_args([])
    assert (args.provider, args.log_level) == ("ollama", "WARNING")
    assert parse_args(["--provider", "google"]).provider == "google"
    with pytest.raises(SystemExit):
        parse_args(["--provider", "nope"])


def test_build_agent_input_without_images_or_history() -> None:
    assert build_agent_input("what is TDR?", []) == "\nUser: what is TDR?"


def test_build_agent_input_with_images_and_history() -> None:
    text = build_agent_input("explain", ["a plot", "a circuit"], history="hi", include_history=True)
    assert text == (
        "previous history = hi\n"
        "Attached images:\nImage 1: a plot\nImage 2: a circuit\n\n"
        "\nUser: explain"
    )


def test_collect_image_paths_stops_on_empty_answer_and_skips_missing_files(tmp_path: Path) -> None:
    image = tmp_path / "plot.png"
    image.write_bytes(b"png")
    messages: list[str] = []
    read = Script(str(tmp_path / "missing.png"), str(image), "")
    assert collect_image_paths(read, messages.append) == [str(image)]
    assert messages == [f"Image not found: {tmp_path / 'missing.png'}. Try again."]


def test_collect_image_paths_asks_at_most_three_times(tmp_path: Path) -> None:
    image = tmp_path / "plot.png"
    image.write_bytes(b"png")
    read = Script(*[str(image)] * 5)
    assert len(collect_image_paths(read, lambda message: None)) == 3
    assert len(read.prompts) == 3


def test_describe_images_replaces_a_failing_image_with_a_placeholder() -> None:
    messages: list[str] = []

    def describe(path: str) -> str:
        if path == "bad.png":
            raise OSError("cannot open")
        return f"description of {path}"

    result = describe_images(["ok.png", "bad.png"], describe, messages.append)
    assert result == ["description of ok.png", "Image description unavailable"]
    assert messages == ["Error describing image bad.png: cannot open"]


def test_chat_loop_answers_until_exit() -> None:
    assistant = FakeAssistant(PROFILES["ollama"], ["first answer", "second answer"])
    output: list[str] = []
    read = Script("hello", "", "again", "", "EXIT")
    chat_loop(assistant, read, output.append)
    assert assistant.inputs == ["\nUser: hello", "\nUser: again"]
    assert "first answer" in output and "second answer" in output
    assert output[-1] == "Exiting chatbot."


def test_chat_loop_describes_attached_images(tmp_path: Path) -> None:
    image = tmp_path / "plot.png"
    image.write_bytes(b"png")
    assistant = FakeAssistant(PROFILES["openai"])
    read = Script("what is this?", str(image), "", "exit")
    chat_loop(assistant, read, lambda message: None, describe=lambda path: "a TDR plot")
    assert assistant.inputs == [
        "previous history = earlier message\nAttached images:\nImage 1: a TDR plot\n\n"
        "\nUser: what is this?"
    ]


def test_chat_loop_survives_a_failing_turn() -> None:
    assistant = FakeAssistant(PROFILES["ollama"], ["RAISE", "recovered"])
    output: list[str] = []
    chat_loop(assistant, Script("one", "", "two", "", "exit"), output.append)
    assert "Error: model unreachable" in output
    assert "recovered" in output


def test_chat_loop_ends_cleanly_at_end_of_input() -> None:
    output: list[str] = []
    chat_loop(FakeAssistant(PROFILES["ollama"]), Script(), output.append)
    assert output[-1].strip() == "Exiting chatbot."
