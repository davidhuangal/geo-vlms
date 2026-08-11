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


@pytest.fixture
def prev_meta(examples) -> dict:
    """Sidecar meta matching the `examples` fixture."""
    return {
        "args": {"model": "stub/model", "max_new_tokens": 64},
        "dataset": {"sha256": dataset_sha256(examples)},
    }


def matching_args() -> dict:
    return {"model": "stub/model", "max_new_tokens": 64}


def test_validate_resume_matching_config_passes(prev_meta, examples):
    validate_resume(prev_meta, examples, matching_args())


def test_validate_resume_dataset_mismatch_raises(prev_meta):
    with pytest.raises(ValueError, match="Dataset does not match"):
        validate_resume(prev_meta, make_examples(2), matching_args())


def test_validate_resume_model_mismatch_raises(prev_meta, examples):
    args = matching_args() | {"model": "other/model"}
    with pytest.raises(ValueError, match="--model=other/model"):
        validate_resume(prev_meta, examples, args)


def test_validate_resume_max_new_tokens_mismatch_raises(prev_meta, examples):
    args = matching_args() | {"max_new_tokens": 128}
    with pytest.raises(ValueError, match="--max-new-tokens=128"):
        validate_resume(prev_meta, examples, args)


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
