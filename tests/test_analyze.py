import json

import pytest

from geo_vlms.commands.analyze import main


@pytest.fixture
def records(tmp_path):
    path = tmp_path / "run.jsonl"
    record = {"id": "0", "expected": 3, "metadata": {"category": "ship"}}
    path.write_text(json.dumps(record | {"output": "3"}) + "\n")
    return path


def write_meta(records, task):
    meta = {"args": {"task": task}}
    records.with_suffix(".meta.json").write_text(json.dumps(meta))


def test_task_comes_from_meta(records, capsys):
    write_meta(records, "counting")

    main(["--records", str(records)])

    assert "exact_match" in capsys.readouterr().out


def test_matching_task_is_accepted(records, capsys):
    write_meta(records, "counting")

    main(["--records", str(records), "--task", "counting"])

    assert "exact_match" in capsys.readouterr().out


def test_mismatched_task_exits(records, capsys):
    write_meta(records, "counting")

    with pytest.raises(SystemExit) as exit_info:
        main(["--records", str(records), "--task", "existence"])

    assert exit_info.value.code == 2
    assert capsys.readouterr().err.startswith(
        "--task existence does not match counting in "
    )


def test_missing_meta_needs_task(records, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--records", str(records)])

    assert exit_info.value.code == 2
    assert capsys.readouterr().err.endswith("run.meta.json not found; pass --task\n")


def test_missing_meta_warns_with_task(records, capsys):
    main(["--records", str(records), "--task", "counting"])

    captured = capsys.readouterr()
    assert captured.err.startswith("Warning: ")
    assert "scoring as counting" in captured.err
    assert "exact_match" in captured.out


def test_meta_without_task_exits(records, capsys):
    records.with_suffix(".meta.json").write_text("{}")

    with pytest.raises(SystemExit) as exit_info:
        main(["--records", str(records)])

    assert exit_info.value.code == 2
    assert capsys.readouterr().err.endswith("run.meta.json has no args.task\n")
