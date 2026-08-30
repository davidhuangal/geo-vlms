import runpy
import sys
from pathlib import Path

import pytest

RUN_SCRIPT = Path(__file__).parents[1] / "scripts" / "run.py"
COMMON = ["-t", "counting", "-m", "org/model", "-b", "huggingface", "-o", "out.jsonl"]


def _parse(monkeypatch, *extra: str):
    monkeypatch.setattr(sys, "argv", ["run.py", *COMMON, *extra])
    return runpy.run_path(str(RUN_SCRIPT), run_name="test")["parse_args"]()


def test_defaults_follow_dataset(monkeypatch):
    vhr10 = _parse(monkeypatch, "-d", "vhr10")
    dior = _parse(monkeypatch, "-d", "dior")

    assert (vhr10.data_dir, vhr10.split) == ("data/vhr10", None)
    assert (dior.data_dir, dior.split) == ("data/dior", "test")


@pytest.mark.parametrize(
    "dataset, flags",
    [
        ("dior", ["--num-pos", "1"]),
        ("dior", ["--no-neg"]),
        ("vhr10", ["--split", "val"]),
        ("vhr10", ["--num-images", "1"]),
    ],
)
def test_rejects_flags_of_other_dataset(dataset, flags, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _parse(monkeypatch, "-d", dataset, *flags)

    assert "only apply to --dataset" in capsys.readouterr().err
