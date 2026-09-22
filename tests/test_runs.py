import json

import pytest

from geo_vlms.backends.base import Generation
from geo_vlms.example import Example
from geo_vlms.provenance import dataset_sha256
from geo_vlms.runs import (
    RecordWriter,
    check_backend,
    check_keys,
    drop_truncated_tail,
    finished_ids,
    note_resume,
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

    def generate(self, prompt, images, max_new_tokens, top_logprobs=None):
        return Generation(text="canned response")

    def describe(self):
        return dict(self._description)


@pytest.fixture
def prev_meta(examples) -> dict:
    """Sidecar meta matching the `examples` fixture."""
    return {
        "args": {"model_name": "stub/model", "max_new_tokens": 64},
        "dataset": {"sha256": dataset_sha256(examples)},
        "backend": StubBackend().describe(),
    }


RESUME_KEYS = ["args.model_name", "args.max_new_tokens", "dataset.sha256"]


def test_check_keys_matching_passes(prev_meta):
    check_keys(prev_meta, json.loads(json.dumps(prev_meta)), RESUME_KEYS)


def test_check_keys_dataset_mismatch_raises(prev_meta):
    curr = json.loads(json.dumps(prev_meta))
    curr["dataset"]["sha256"] = dataset_sha256(make_examples(2))
    with pytest.raises(ValueError, match=r"dataset\.sha256"):
        check_keys(prev_meta, curr, RESUME_KEYS)


def test_check_keys_model_mismatch_raises(prev_meta):
    curr = json.loads(json.dumps(prev_meta))
    curr["args"]["model_name"] = "other/model"
    with pytest.raises(ValueError, match=r"args\.model_name='other/model'"):
        check_keys(prev_meta, curr, RESUME_KEYS)


def test_check_keys_missing_key_raises(prev_meta):
    curr = json.loads(json.dumps(prev_meta))
    del curr["args"]["max_new_tokens"]
    with pytest.raises(ValueError, match="<missing>"):
        check_keys(prev_meta, curr, RESUME_KEYS)


def test_check_backend_mismatch_raises(prev_meta):
    curr = {"backend": {"kind": "stub", "name": "other/backend"}}
    with pytest.raises(ValueError, match="Backend does not match"):
        check_backend(prev_meta, curr)


def test_check_backend_ignores_base_url(prev_meta):
    # Same backend reached through a different address should still resume
    prev_meta["backend"]["base_url"] = "http://old-tunnel:8080/v1"
    curr = {
        "backend": {
            "kind": "stub",
            "name": "stub/model",
            "base_url": "http://new-tunnel:9090/v1",
        }
    }
    check_backend(prev_meta, curr)


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


def test_finished_ids_custom_key(tmp_path):
    out = tmp_path / "records.jsonl"
    records = [{"det_id": "x"}, {"det_id": "y"}]
    out.write_text("".join(json.dumps(r) + "\n" for r in records))

    assert finished_ids(out, key="det_id") == {"x", "y"}


def test_record_writer_writes_and_appends(tmp_path):
    out = tmp_path / "records.jsonl"
    with RecordWriter(out) as writer:
        writer.write({"id": "a"})
        # Flushed before the context closes
        assert out.read_text() == '{"id": "a"}\n'
        writer.write({"id": "b"})

    with RecordWriter(out, append=True) as writer:
        writer.write({"id": "c"})

    assert out.read_text() == '{"id": "a"}\n{"id": "b"}\n{"id": "c"}\n'

    with RecordWriter(out) as writer:
        writer.write({"id": "d"})

    assert out.read_text() == '{"id": "d"}\n'


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
