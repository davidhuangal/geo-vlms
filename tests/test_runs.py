import json

import pytest

from geo_vlms.example import Example
from geo_vlms.provenance import dataset_sha256
from geo_vlms.runs import (
    drop_truncated_tail,
    finished_ids,
    note_resume,
    validate_resume,
)


def make_examples(n: int) -> list[Example]:
    """Build n small distinct examples."""
    return [
        Example(
            id=f"example_{i}",
            image_path=f"/{i}.jpeg",
            prompt=f"prompt for example {i}",
        )
        for i in range(n)
    ]


@pytest.fixture
def examples() -> list[Example]:
    return make_examples(3)


class StubBackend:
    """Backend with a fixed self-description."""

    def __init__(self, description: dict | None = None):
        self._description = description or {"kind": "stub", "name": "stub/model"}

    def generate(self, prompt, image_paths, max_new_tokens):
        return "canned response"

    def describe(self):
        return dict(self._description)


@pytest.fixture
def prev_meta(examples) -> dict:
    """Sidecar meta matching the `examples` fixture."""
    return {
        "args": {"model": "stub/model", "max_new_tokens": 64},
        "dataset": {"sha256": dataset_sha256(examples)},
        "backend": StubBackend().describe(),
    }


def matching_args() -> dict:
    return {"model": "stub/model", "max_new_tokens": 64}


def test_validate_resume_matching_config_passes(prev_meta, examples):
    validate_resume(prev_meta, examples, matching_args(), StubBackend())


def test_validate_resume_dataset_mismatch_raises(prev_meta):
    with pytest.raises(ValueError, match="Dataset does not match"):
        validate_resume(prev_meta, make_examples(2), matching_args(), StubBackend())


def test_validate_resume_model_mismatch_raises(prev_meta, examples):
    args = matching_args() | {"model": "other/model"}
    with pytest.raises(ValueError, match="--model=other/model"):
        validate_resume(prev_meta, examples, args, StubBackend())


def test_validate_resume_max_new_tokens_mismatch_raises(prev_meta, examples):
    args = matching_args() | {"max_new_tokens": 128}
    with pytest.raises(ValueError, match="--max-new-tokens=128"):
        validate_resume(prev_meta, examples, args, StubBackend())


def test_validate_resume_backend_mismatch_raises(prev_meta, examples):
    other = StubBackend({"kind": "stub", "name": "other/backend"})
    with pytest.raises(ValueError, match="Backend does not match"):
        validate_resume(prev_meta, examples, matching_args(), other)


def test_validate_resume_ignores_base_url(prev_meta, examples):
    # Same backend reached through a different address should still resume
    prev_meta["backend"]["base_url"] = "http://old-tunnel:8080/v1"
    moved = StubBackend(
        {"kind": "stub", "name": "stub/model", "base_url": "http://new-tunnel:9090/v1"}
    )
    validate_resume(prev_meta, examples, matching_args(), moved)


def test_drop_truncated_tail_clean_file_untouched(tmp_path):
    out = tmp_path / "records.jsonl"
    content = '{"id": "a"}\n{"id": "b"}\n'
    out.write_text(content)

    assert drop_truncated_tail(out) is False
    assert out.read_text() == content


def test_drop_truncated_tail_partial_last_line_dropped(tmp_path):
    out = tmp_path / "records.jsonl"
    out.write_text('{"id": "a"}\n{"id": "b"}\n{"id": "c", "outp')

    assert drop_truncated_tail(out) is True
    assert out.read_text() == '{"id": "a"}\n{"id": "b"}\n'


def test_drop_truncated_tail_single_partial_line_resets_file(tmp_path):
    out = tmp_path / "records.jsonl"
    out.write_text('{"id": "a", "outp')

    assert drop_truncated_tail(out) is True
    assert out.read_text() == ""


def test_drop_truncated_tail_missing_final_newline_restored(tmp_path):
    out = tmp_path / "records.jsonl"
    out.write_text('{"id": "a"}\n{"id": "b"}')

    assert drop_truncated_tail(out) is False
    assert out.read_text() == '{"id": "a"}\n{"id": "b"}\n'


def test_drop_truncated_tail_empty_file_untouched(tmp_path):
    out = tmp_path / "records.jsonl"
    out.write_text("")

    assert drop_truncated_tail(out) is False
    assert out.read_text() == ""


def test_finished_ids_reads_unique_ids(tmp_path):
    out = tmp_path / "records.jsonl"
    records = [{"id": "a"}, {"id": "b"}, {"id": "b"}]
    out.write_text("".join(json.dumps(r) + "\n" for r in records))

    assert finished_ids(out) == {"a", "b"}


def test_finished_ids_empty_file_gives_empty_set(tmp_path):
    out = tmp_path / "records.jsonl"
    out.write_text("")

    assert finished_ids(out) == set()


def test_note_resume_appends_first_entry():
    meta = {"args": {}}
    result = note_resume(meta, "cmd --resume", "2026-08-11T00:00:00+00:00")

    assert result["resumes"] == [
        {"command": "cmd --resume", "started_at": "2026-08-11T00:00:00+00:00"}
    ]


def test_note_resume_appends_to_existing_entries():
    first = {"command": "cmd1", "started_at": "t1"}
    meta = {"resumes": [first]}
    result = note_resume(meta, "cmd2", "t2")

    assert result["resumes"] == [first, {"command": "cmd2", "started_at": "t2"}]
