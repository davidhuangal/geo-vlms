import json

import pytest

from geo_vlms.commands.analyze import main


@pytest.fixture
def records(tmp_path):
    path = tmp_path / "run.jsonl"
    record = {"id": "0", "expected": 3, "metadata": {"category": "ship"}}
    path.write_text(json.dumps(record | {"output": "3", "prompt_tokens": 9}) + "\n")
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


def test_metric_typo_suggests_and_lists(records, capsys):
    write_meta(records, "counting")

    with pytest.raises(SystemExit) as exit_info:
        main(["--records", str(records), "--metrics", "exact_mtch"])

    assert exit_info.value.code == 2
    first, second = capsys.readouterr().err.splitlines()
    assert first == "Unknown metric 'exact_mtch'. Did you mean 'exact_match'?"
    assert second.startswith("Metrics: ")
    assert "exact_match" in second
    assert "output" not in second


def test_groupby_typo_suggests(records, capsys):
    write_meta(records, "counting")

    with pytest.raises(SystemExit):
        main(["--records", str(records), "--groupby", "categry"])

    err = capsys.readouterr().err
    assert err.startswith("Unknown column 'categry'. Did you mean 'category'?\n")


def test_unknown_name_without_close_match(records, capsys):
    write_meta(records, "counting")

    with pytest.raises(SystemExit):
        main(["--records", str(records), "--metrics", "zzz", "exact_mtch"])

    lines = capsys.readouterr().err.splitlines()
    assert lines[0] == "Unknown metric 'zzz'."
    assert lines[1].startswith("Unknown metric 'exact_mtch'.")
    assert lines[2].startswith("Metrics: ")


def test_any_numeric_column_is_a_metric(records, capsys):
    write_meta(records, "counting")

    main(["--records", str(records), "--metrics", "prompt_tokens"])

    assert "prompt_tokens" in capsys.readouterr().out
